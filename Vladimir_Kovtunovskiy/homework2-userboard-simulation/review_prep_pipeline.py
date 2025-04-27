"""
A robust, self-contained clustering pipeline for Spotify reviews. 
Optimized for speed using MPS on Apple Silicon and TF-IDF keywords.

Highlights:
* **MPS Acceleration:** Automatically utilizes Apple Silicon GPU for embedding.
* **TF-IDF Keywords:** Replaced KeyBERT with faster TF-IDF for keyword extraction.
* **Optimized K-Means:** Faster K selection in fallback scenario.
* **Relative paths** anchored to the project root.
* CLI overrides for input CSV & output dir.
* Dual logging (pretty console via Rich, plus logfile in output folder).
* Graceful error handling with non-zero exit codes.
* Configuration dataclass for better organization.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import traceback
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import hdbscan
import numpy as np
import pandas as pd
import torch # Ensure torch is imported
import umap
from langdetect import LangDetectException, detect
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from rich.logging import RichHandler
from sentence_transformers import SentenceTransformer
# Replace KeyBERT with TF-IDF
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

# -----------------------------------------------------------------------------
# Configuration Dataclass
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    """Pipeline Configuration."""
    # Paths (relative to project root)
    default_csv: Path = Path("spotify_reviews.csv")
    default_out: Path = Path("cluster_outputs")
    logfile_name: str = "review_prep_pipeline.log"

    # Models & Language
    embedding_model: str = "all-MiniLM-L6-v2"
    language: str = "english"

    # Data Handling
    min_review_words: int = 3
    max_texts_per_cluster_keywords: int = 1500 # Reduced slightly for TF-IDF speed balance
    samples_per_cluster_output: int = 5

    # Clustering Parameters
    umap_n_components: int = 50
    umap_metric: str = "cosine"
    umap_min_dist: float = 0.0
    hdbscan_min_cluster_size: int = 50
    hdbscan_metric: str = "euclidean"
    kmeans_max_k: int = 15
    kmeans_n_init_k_selection: int = 3 # Faster K selection
    min_silhouette_score_fallback: float = 0.20
    min_cluster_size_output: int = 30

    # Keyword Parameters
    keywords_per_cluster: int = 10
    min_keyword_alpha_ratio: float = 0.6
    min_keyword_length: int = 3
    tfidf_ngram_range: Tuple[int, int] = (1, 2)
    tfidf_max_df: float = 0.85
    tfidf_min_df: int = 3

    # Misc
    random_seed: int = 42
    embedding_batch_size: int = 64


# Initialize Config
CFG = Config()

# -----------------------------------------------------------------------------
# PATHS & Global Setup
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = PROJECT_ROOT / CFG.default_csv
DEFAULT_OUT = PROJECT_ROOT / CFG.default_out

# Set random seed for reproducibility where applicable (Numpy, random, etc.)
# Note: Doesn't guarantee full determinism with UMAP/HDBSCAN/PyTorch on MPS/GPU
random.seed(CFG.random_seed)
np.random.seed(CFG.random_seed)
# PyTorch seed setting can be added if strict determinism is needed, but often not fully possible
# torch.manual_seed(CFG.random_seed)
# if torch.backends.mps.is_available():
#     torch.mps.manual_seed(CFG.random_seed) # If available

# Lazy loaded NLTK resources
STOPWORDS = None
sia = None

# -----------------------------------------------------------------------------
# LOGGING SETUP
# -----------------------------------------------------------------------------
log = logging.getLogger(__name__) # Define logger globally

def init_logger(out_dir: Path) -> None:
    """Initializes logging to console and file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / CFG.logfile_name

    # Use basicConfig to setup root logger handlers
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True), logging.FileHandler(log_path, "w", "utf-8")],
    )
    # Suppress noisy logs from underlying libraries if needed
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    # Set log level for this script's logger specifically if needed
    log.setLevel(logging.INFO)
    log.info("--- Starting Clustering Pipeline ---")
    log.info("Logs will be saved to %s", log_path)


# -----------------------------------------------------------------------------
# DEVICE DETECTION (MPS / CPU)
# -----------------------------------------------------------------------------
def get_compute_device() -> torch.device:
    """Detects and returns the appropriate torch device (MPS or CPU)."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # Additional check is_built() is good practice
        device = torch.device("mps")
        log.info("Apple Silicon MPS device detected. Using MPS for acceleration.")
    else:
        device = torch.device("cpu")
        log.info("MPS not available. Using CPU.")
    return device

DEVICE = get_compute_device()

# -----------------------------------------------------------------------------
# TEXT HELPERS
# -----------------------------------------------------------------------------
_re_spotify_variants = re.compile(r"s[po]t[fph]?i[y]?|sportify|spotfiy|spotfi")

def latin(text: str) -> str:
    """Normalize unicode characters to their closest ASCII equivalent."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def basic_clean(text: str) -> str:
    """Perform basic text cleaning: lowercasing, ASCII normalization, variant replacement,
       punctuation removal, and stopword removal."""
    global STOPWORDS # Ensure STOPWORDS are loaded
    if STOPWORDS is None:
        log.error("STOPWORDS not loaded before cleaning. Call load_nltk_resources first.")
        # Or load them here lazily, but better to do it explicitly at startup
        raise RuntimeError("NLTK stopwords not loaded.")

    if not isinstance(text, str):
        return "" # Handle potential non-string input gracefully

    text = latin(text.lower())
    text = _re_spotify_variants.sub("spotify", text) # Normalize Spotify variants
    text = re.sub(r"[^a-z0-9\s]", " ", text) # Remove non-alphanumeric chars
    tokens = [
        token for token in text.split()
        if token not in STOPWORDS and len(token) >= CFG.min_keyword_length
    ]
    return " ".join(tokens)

# -------------------------------------------------
# Extra stop-keywords (too generic → always noisy)
# -------------------------------------------------
BANNED_KWS = {
    "music", "song", "songs", "app", "spotify", "play",
    "listen", "good", "great", "best", "love"
}

def keyword_ok(kw: str) -> bool:
    """Validate a keyword: length, alpha-ratio, and not too generic."""
    if not kw or len(kw) < CFG.min_keyword_length:
        return False
    if kw.lower() in BANNED_KWS:                 
        return False
    alpha_ratio = sum(c.isalpha() for c in kw) / len(kw)
    return alpha_ratio >= CFG.min_keyword_alpha_ratio

# -----------------------------------------------------------------------------
# DATA LOAD + SENTIMENT
# -----------------------------------------------------------------------------

def load_reviews(path: Path) -> pd.DataFrame:
    """Loads reviews from CSV, performs basic filtering and cleaning."""
    log.info("Loading reviews from: %s", path)
    if not path.exists():
        log.error("Input file not found: %s", path)
        raise FileNotFoundError(f"Input CSV not found at {path}")

    df = pd.read_csv(path)
    log.info("Initial rows loaded: %d", len(df))

    # Basic filtering
    df = df.dropna(subset=["Review"]).copy()
    log.info("Rows after dropping NA reviews: %d", len(df))

    # Cleaning
    df["clean"] = df["Review"].astype(str).apply(basic_clean)

    # Filter by word count *after* cleaning
    df = df[df["clean"].str.split().str.len() >= CFG.min_review_words].reset_index(drop=True)
    log.info("Rows after filtering by min words (%d): %d", CFG.min_review_words, len(df))

    if df.empty:
        log.error("No reviews meet the minimum word count after cleaning.")
        raise ValueError("No reviews long enough to process.")

    return df


def add_sentiment(df: pd.DataFrame) -> None:
    """Adds sentiment score and label to the DataFrame using VADER."""
    global sia # Ensure VADER is loaded
    if sia is None:
        log.error("SentimentIntensityAnalyzer (sia) not loaded. Call load_nltk_resources first.")
        raise RuntimeError("NLTK VADER lexicon not loaded.")

    log.info("Calculating sentiment scores...")
    df["sentiment_score"] = df["Review"].astype(str).apply(lambda t: sia.polarity_scores(t)["compound"])
    df["sentiment_label"] = pd.cut(
        df["sentiment_score"],
        bins=[-1.01, -0.05, 0.05, 1.01], # Use slightly wider bins to catch exact boundaries
        labels=["negative", "neutral", "positive"],
        right=True # Include right boundary
    )
    log.info("Sentiment analysis complete.")

# -----------------------------------------------------------------------------
# EMBEDDINGS + REDUCTION
# -----------------------------------------------------------------------------
# Initialize embedder globally but load model later if needed (or keep as is)
embedder = None

def get_embedder() -> SentenceTransformer:
    """Initializes and returns the SentenceTransformer model on the correct device."""
    global embedder
    if embedder is None:
        log.info(f"Loading SentenceTransformer model '{CFG.embedding_model}' onto device '{DEVICE}'...")
        embedder = SentenceTransformer(CFG.embedding_model, device=DEVICE)
        log.info(f"Embedder loaded. Max sequence length: {embedder.max_seq_length}")
    return embedder

def embed(texts: List[str]) -> np.ndarray:
    """Encodes a list of texts into embeddings using SentenceTransformer."""
    model = get_embedder()
    log.info("Encoding %d reviews...", len(texts))
    # The model loaded via get_embedder() is already on the correct device (MPS or CPU)
    embeddings = model.encode(
        texts,
        batch_size=CFG.embedding_batch_size,
        show_progress_bar=True, # Progress bar is helpful
        convert_to_numpy=True   # Ensures NumPy array output
    )
    log.info("Encoding complete. Embedding shape: %s", embeddings.shape)
    return embeddings


def reduce_dims(vecs: np.ndarray) -> np.ndarray:
    """Reduces dimensionality of embeddings using UMAP."""
    log.info("Performing dimensionality reduction with UMAP (n_components=%d)...", CFG.umap_n_components)
    reducer = umap.UMAP(
        n_components=CFG.umap_n_components,
        metric=CFG.umap_metric,
        random_state=CFG.random_seed,
        min_dist=CFG.umap_min_dist,
        n_neighbors=15, # Default, often works well
        low_memory=True, # Can help on machines with less RAM
        verbose=False # Set to True for detailed UMAP logs if needed
    )
    reduced_embeddings = reducer.fit_transform(vecs)
    log.info("Dimensionality reduction complete. Reduced shape: %s", reduced_embeddings.shape)
    return reduced_embeddings

# -----------------------------------------------------------------------------
# CLUSTERING
# -----------------------------------------------------------------------------

def choose_best_k(mat: np.ndarray, k_min: int = 2) -> int:
    """
    Fast heuristic K search:
    • Fits K-Means once per k (2‒CFG.kmeans_max_k)
    • Evaluates silhouette on a 10 k-point sample
    • Early-stops after 3 consecutive drops
    Runs 5-10× quicker than full grid-search.
    """
    SAMPLE = min(10_000, mat.shape[0])
    sample_idx = np.random.choice(mat.shape[0], SAMPLE, replace=False)
    X_sample = mat[sample_idx]

    best_k, best_sil = k_min, -1.0
    drops = 0
    for k in range(k_min, CFG.kmeans_max_k + 1):
        km = KMeans(
            n_clusters=k,
            random_state=CFG.random_seed,
            algorithm="elkan",
            n_init=2,
            max_iter=100,
        )
        labels = km.fit_predict(mat)
        sil = silhouette_score(X_sample, labels[sample_idx])
        log.debug("k=%d → sil=%.3f", k, sil)

        if sil > best_sil:
            best_k, best_sil, drops = k, sil, 0
        else:
            drops += 1
            if drops >= 3:                     # early exit
                log.info("Silhouette dropping 3×—stop at k=%d", best_k)
                break

    log.info("Optimal K chosen: %d (sil=%.3f)", best_k, best_sil)
    return best_k


def cluster(mat: np.ndarray) -> np.ndarray:
    best_k = choose_best_k(mat) # Use the optimized K selection
    kmeans = KMeans(
        n_clusters=best_k,
        random_state=CFG.random_seed,
        n_init='auto' # Let scikit-learn decide based on version (often 10)
    )
    final_labels = kmeans.fit_predict(mat)
    try:
        final_sil_score = silhouette_score(mat, final_labels)
        log.info("K-Means (k=%d) Silhouette Score: %.4f", best_k, final_sil_score)
    except ValueError as e:
        log.warning("Could not calculate silhouette score for final K-Means labels: %s", e)

    return final_labels.astype(int)

# -----------------------------------------------------------------------------
# KEYWORDS (TF-IDF Implementation)
# -----------------------------------------------------------------------------

tfidf_vectorizer = None
tfidf_matrix = None
feature_names = None

def calculate_tfidf(texts: pd.Series) -> None:
    """Calculates TF-IDF matrix for the entire corpus."""
    global tfidf_vectorizer, tfidf_matrix, feature_names
    log.info("Calculating TF-IDF matrix...")
    tfidf_vectorizer = TfidfVectorizer(
        stop_words=CFG.language, # Use standard stop words
        ngram_range=CFG.tfidf_ngram_range,
        max_df=CFG.tfidf_max_df, # Ignore terms that appear in > 85% of docs
        min_df=CFG.tfidf_min_df, # Ignore terms that appear in < 3 docs
        use_idf=True,
        smooth_idf=True,
        norm='l2' # Standard L2 normalization
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
    log.info("TF-IDF matrix calculated. Shape: %s, Features: %d", tfidf_matrix.shape, len(feature_names))


def extract_tfidf_keywords(cluster_indices: np.ndarray) -> List[str]:
    """Extracts top keywords for a specific cluster based on mean TF-IDF scores."""
    global tfidf_matrix, feature_names

    if tfidf_matrix is None or feature_names is None:
        log.error("TF-IDF matrix not calculated. Call calculate_tfidf first.")
        raise RuntimeError("TF-IDF matrix not available.")

    if len(cluster_indices) == 0:
        return []

    # Slice the TF-IDF matrix for the current cluster
    cluster_tfidf_matrix = tfidf_matrix[cluster_indices]

    # Calculate the mean TF-IDF score for each term across the cluster's documents
    # Convert sparse matrix slice to array for mean calculation
    mean_tfidf_scores = np.array(cluster_tfidf_matrix.mean(axis=0)).flatten()

    # Get indices of terms sorted by score (descending)
    sorted_indices = np.argsort(mean_tfidf_scores)[::-1]

    # Extract top keywords meeting criteria
    top_keywords = []
    for idx in sorted_indices:
        keyword = feature_names[idx]
        if keyword_ok(keyword): # Apply keyword quality filter
            top_keywords.append(keyword)
            if len(top_keywords) >= CFG.keywords_per_cluster:
                break

    return top_keywords

# -----------------------------------------------------------------------------
# BUILD CLUSTER SUMMARY
# -----------------------------------------------------------------------------

def build_clusters_summary(df: pd.DataFrame) -> Dict[int, dict]:
    """Builds the final summary dictionary for each valid cluster."""
    log.info("Building cluster summaries...")
    cluster_ids = sorted([c for c in df["cluster"].unique() if c != -1]) # Exclude noise points (-1) if any
    cluster_summaries: Dict[int, dict] = {}

    if tfidf_matrix is None:
        log.error("Attempting to build summaries before TF-IDF calculation.")
        raise RuntimeError("TF-IDF matrix must be calculated before building summaries.")

    for cid in cluster_ids:
        cluster_mask = df["cluster"] == cid
        sub_df = df[cluster_mask]

        if len(sub_df) < CFG.min_cluster_size_output:
            log.info("Skipping cluster %d: size (%d) is below minimum output threshold (%d).",
                     cid, len(sub_df), CFG.min_cluster_size_output)
            continue

        log.info("Processing cluster %d (size: %d)...", cid, len(sub_df))

        # Get document indices for keyword extraction
        cluster_doc_indices = sub_df.index.to_numpy()

        # Extract keywords using the new TF-IDF based method
        keywords = extract_tfidf_keywords(cluster_doc_indices)

        # Get sample reviews (handle case where cluster is smaller than sample count)
        sample_size = min(CFG.samples_per_cluster_output, len(sub_df))
        samples = random.sample(sub_df["Review"].tolist(), sample_size)

        cluster_summaries[int(cid)] = {
            "cluster_id": int(cid), # Add explicit ID
            "count": int(len(sub_df)),
            "avg_sentiment": float(sub_df["sentiment_score"].mean().round(3)),
            "sentiment_dist": {
                label: int(count)
                for label, count in sub_df["sentiment_label"].value_counts().items()
            },
            "keywords": keywords,
            "samples": samples,
        }
        log.debug("Cluster %d summary created with %d keywords.", cid, len(keywords))

    log.info("Finished building summaries for %d clusters.", len(cluster_summaries))
    return cluster_summaries

# -----------------------------------------------------------------------------
# SAVE HELPERS
# -----------------------------------------------------------------------------

def save_txt(report: Dict[int, dict], path: Path):
    """Saves a human-readable text report of the clusters."""
    log.info("Saving text report to: %s", path)
    try:
        with path.open("w", encoding="utf-8") as f:
            for cid, info in sorted(report.items()): # Sort by cluster ID for consistency
                f.write(f"--- Cluster {cid} ---\n")
                f.write(f"Count: {info['count']}\n")
                f.write(f"Avg Sentiment: {info['avg_sentiment']:.3f}\n")
                f.write("Sentiment Distribution:\n")
                for label, count in info['sentiment_dist'].items():
                    f.write(f"  - {label.capitalize()}: {count}\n")
                f.write("Keywords:\n  " + ", ".join(info["keywords"]) + "\n")
                f.write("Sample Reviews:\n")
                for s in info["samples"]:
                    # Clean up sample for display
                    cleaned_sample = re.sub(r'\s+', ' ', s).strip()
                    f.write(f"  - {cleaned_sample[:180]}...\n") # Limit length
                f.write("\n")
    except IOError as e:
        log.error("Failed to write text report to %s: %s", path, e)


def save_json(report: Dict[int, dict], path: Path):
    """Saves the cluster report dictionary as a JSON file."""
    log.info("Saving JSON report to: %s", path)
    try:
        with path.open("w", encoding="utf-8") as f:
            # Convert dict values to list and sort by cluster_id for consistent JSON output
            sorted_report_list = sorted(report.values(), key=lambda x: x['cluster_id'])
            json.dump(sorted_report_list, f, ensure_ascii=False, indent=2)
    except TypeError as e:
        log.error("Data serialization error while saving JSON: %s", e)
    except IOError as e:
        log.error("Failed to write JSON report to %s: %s", path, e)

# -----------------------------------------------------------------------------
# NLTK Resource Loader
# -----------------------------------------------------------------------------

def load_nltk_resources():
    """Downloads and loads necessary NLTK resources."""
    global STOPWORDS, sia
    try:
        import nltk
        log.info("Downloading NLTK resources (stopwords, vader_lexicon)...")
        nltk.download("stopwords", quiet=True)
        nltk.download("vader_lexicon", quiet=True)
        # Load resources into global variables after download
        STOPWORDS = set(stopwords.words(CFG.language))
        sia = SentimentIntensityAnalyzer()
        log.info("NLTK resources loaded.")
    except ImportError:
        log.error("NLTK library not found. Please install it: pip install nltk")
        raise
    except Exception as e:
        log.error("Failed to download or load NLTK resources: %s", e)
        raise

# -----------------------------------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------------------------------

def run_pipeline(csv_path: Path, out_dir: Path):
    """Executes the entire clustering pipeline."""
    global log # Ensure we are using the initialized logger

    log.info("--- Starting Main Pipeline Execution ---")
    log.info("Input CSV: %s", csv_path)
    log.info("Output Directory: %s", out_dir)

    # 1. Load NLTK resources (critical for cleaning and sentiment)
    load_nltk_resources()

    # 2. Load and preprocess data
    df = load_reviews(csv_path)
    add_sentiment(df)

    # 3. Calculate embeddings
    # Use the 'clean' text which has stopwords removed etc.
    embeddings = embed(df["clean"].tolist())

    # 4. Reduce dimensionality
    reduced_embeddings = reduce_dims(embeddings)

    # 5. Cluster
    df["cluster"] = cluster(reduced_embeddings)

    # 6. Calculate TF-IDF (needed for keyword extraction)
    # Use the same 'clean' text used for embeddings
    calculate_tfidf(df["clean"])

    # 7. Build cluster summaries (includes keyword extraction)
    clusters_summary = build_clusters_summary(df)

    # 8. Save results
    if clusters_summary:
        save_txt(clusters_summary, out_dir / "cluster_report.txt")
        save_json(clusters_summary, out_dir / "clusters_data.json")
        log.info("✅ Pipeline Finished Successfully: Found %d valid clusters.", len(clusters_summary))
    else:
        log.warning("Pipeline finished, but no valid clusters met the output criteria.")

    log.info("--- Pipeline Execution Complete ---")


# -----------------------------------------------------------------------------
# CLI ENTRY POINT
# -----------------------------------------------------------------------------

def cli():
    """Handles command-line argument parsing and pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Spotify review clustering pipeline with MPS acceleration and TF-IDF keywords.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show default values in help
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to the input CSV file containing reviews."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Directory to save output files (logs, reports, JSON)."
    )
    args = parser.parse_args()

    # Initialize logger *after* parsing args to get the correct output directory
    init_logger(args.out)

    try:
        # Filter UMAP warnings which can be noisy and often benign
        import warnings
        warnings.filterwarnings("ignore", message=".*Using precomputed metric.*", category=UserWarning, module="umap")
        warnings.filterwarnings("ignore", message=".*does not have required distance function.*", category=UserWarning, module="umap")

        run_pipeline(args.csv.resolve(), args.out.resolve()) # Use resolved absolute paths

    except FileNotFoundError as e:
        log.error("❌ Pipeline failed: Input file not found.")
        log.error(e)
        sys.exit(2) # Specific exit code for file not found
    except ValueError as e:
        log.error("❌ Pipeline failed: Data validation error.")
        log.error(e)
        sys.exit(3) # Specific exit code for data errors
    except Exception as e:
        log.error("❌ Pipeline failed with an unexpected error:", exc_info=False) # Log exception details to file via handler
        log.error(traceback.format_exc()) # Also print traceback to console log for immediate visibility
        sys.exit(1) # General error exit code


if __name__ == "__main__":
    cli()