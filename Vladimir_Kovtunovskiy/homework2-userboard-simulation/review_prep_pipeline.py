"""
Phase‑1 Extended Pipeline
=========================
Adds:
1. **UMAP** dimensionality‑reduction.
2. Automatic **K selection** (silhouette) for K‑Means.
3. **KeyBERT** keywords (higher quality than RAKE).
4. **Cluster sanity report** written to `cluster_report.txt` (keywords, avg sentiment, samples).
5. **LangGraph export** – dumps `clusters_data.json` with cluster metadata.

How to run
----------
```bash
pip install pandas numpy nltk sentence_transformers umap-learn scikit-learn keybert
python phase1_extended_pipeline.py
```

Outputs
-------
* `cluster_report.txt` – human‑readable cluster overview.
* `clusters_data.json` – machine‑readable for LangGraph nodes.
"""

import pandas as pd
import numpy as np
import re
import random
import json
import logging
from pathlib import Path
from typing import List, Dict

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
CSV_PATH = "Vladimir_Kovtunovskiy/homework2-userboard-simulation/Data/spotify_reviews.csv"
OUTPUT_DIR = Path("Vladimir_Kovtunovskiy/homework2-userboard-simulation/cluster_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
TXT_REPORT = OUTPUT_DIR / "cluster_report.txt"
JSON_EXPORT = OUTPUT_DIR / "clusters_data.json"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
STOPWORDS = set(stopwords.words("english"))
N_KEYWORDS = 10
N_SAMPLES = 3  # random sample reviews per cluster
RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def basic_clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)


def sentiment_columns(df: pd.DataFrame, col: str = "Review") -> None:
    sia = SentimentIntensityAnalyzer()
    df["sentiment_score"] = df[col].map(lambda t: sia.polarity_scores(t)["compound"])
    bins = [-1, -0.05, 0.05, 1]
    labels = ["negative", "neutral", "positive"]
    df["sentiment_label"] = pd.cut(df["sentiment_score"], bins=bins, labels=labels)


def load_reviews(path: str, col: str = "Review", min_words: int = 3) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.dropna(subset=[col], inplace=True)
    df = df[df[col].str.split().str.len() >= min_words]
    df[col] = df[col].astype(str)
    df["clean"] = df[col].map(basic_clean)
    df = df[df["clean"].str.strip().astype(bool)].reset_index(drop=True)
    logging.info("Loaded %d cleaned reviews", len(df))
    return df


def create_embeddings(texts: List[str], model_name: str = EMBED_MODEL_NAME, batch_size: int = 64):
    logging.info("Encoding reviews with %s", model_name)
    model = SentenceTransformer(model_name)
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True)


def reduce_dims(embeddings: np.ndarray, n_components: int = 50):
    reducer = umap.UMAP(n_components=n_components, metric="cosine", random_state=RANDOM_STATE)
    return reducer.fit_transform(embeddings)


def choose_best_k(embeds: np.ndarray, k_min: int = 2, k_max: int = 15):
    best_k, best_score = None, -1
    scores = {}
    for k in range(k_min, k_max + 1):
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(embeds)
        score = silhouette_score(embeds, labels)
        scores[k] = score
        if score > best_score:
            best_k, best_score = k, score
    logging.info("Silhouette scores: %s", scores)
    logging.info("Best k = %d (score %.3f)", best_k, best_score)
    return best_k


def cluster_embeddings(embeds: np.ndarray, k: int):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    return km.fit_predict(embeds)


def cluster_keywords(df: pd.DataFrame, kw_model: KeyBERT, cluster_id: int, n: int = N_KEYWORDS) -> List[str]:
    corpus = " ".join(df[df["cluster"] == cluster_id]["clean"].tolist())
    keywords = kw_model.extract_keywords(corpus, top_n=n, stop_words="english", use_maxsum=True)
    return [kw for kw, _ in keywords]


def build_cluster_report(df: pd.DataFrame, kw_model: KeyBERT) -> Dict[int, dict]:
    report = {}
    cluster_ids = sorted(df["cluster"].unique())

    for cid_np in cluster_ids:
        cid = int(cid_np)
        sub = df[df["cluster"] == cid_np]

        sentiment_dist_np = sub["sentiment_label"].value_counts().to_dict()
        sentiment_dist = {k: int(v) for k, v in sentiment_dist_np.items()}

        report[cid] = {
            "count": int(len(sub)),
            "avg_sentiment": float(sub["sentiment_score"].mean().round(3)),
            "sentiment_dist": sentiment_dist,
            "keywords": cluster_keywords(df, kw_model, cid_np),
            "samples": random.sample(sub["Review"].tolist(), min(N_SAMPLES, len(sub)))
        }
    return report


def save_txt_report(report: Dict[int, dict], path: Path):
    with path.open("w", encoding="utf-8") as f:
        for cid, info in report.items():
            f.write(f"Cluster {cid} | n={info['count']} | avg_sent={info['avg_sentiment']}\n")
            f.write(" keywords: " + ", ".join(info["keywords"]) + "\n")
            for s in info["samples"]:
                f.write("   - " + s[:200].replace("\n", " ") + "\n")
            f.write("\n")
    logging.info("Wrote cluster report to %s", path)


def save_json(report: Dict[int, dict], path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logging.info("Wrote JSON export to %s", path)

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(RANDOM_STATE)
    nltk.download("vader_lexicon", quiet=False)
    nltk.download("stopwords", quiet=False)

    # 1. Load + sentiment
    df = load_reviews(CSV_PATH)
    sentiment_columns(df)

    # 2. Embeddings (+ UMAP)
    embeds = create_embeddings(df["clean"].tolist())
    red = reduce_dims(embeds)

    # 3. Choose K & cluster
    k_opt = choose_best_k(red)
    df["cluster"] = cluster_embeddings(red, k_opt)

    # 4. Keyword + report
    kw_model = KeyBERT(EMBED_MODEL_NAME)
    cluster_info = build_cluster_report(df, kw_model)

    # 5. Persist
    save_txt_report(cluster_info, TXT_REPORT)
    save_json(cluster_info, JSON_EXPORT)

    logging.info("Pipeline complete — %d clusters ready for LangGraph", len(cluster_info))
