"""Utility functions for the summarization agent."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import nltk
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Any, Optional

# Load environment variables
load_dotenv()

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')

def load_data(file_path: str, sample_size: Optional[int] = None, random_seed: int = 42) -> pd.DataFrame:
    """Load and prepare the review data.
    
    Args:
        file_path: Path to the CSV file
        sample_size: Number of reviews to sample (None for all)
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame containing the review data
    """
    # Load the data
    df = pd.read_csv(file_path)
    
    # Clean the data
    df = df.dropna(subset=['Text'])
    
    # Take a random sample if specified
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=random_seed)
    
    return df

def preprocess_text(text: str) -> str:
    """Preprocess text for summarization.
    
    Args:
        text: The input text to preprocess
        
    Returns:
        Preprocessed text
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def create_output_dir(dir_name: str) -> str:
    """Create output directory if it doesn't exist.
    
    Args:
        dir_name: Directory name
        
    Returns:
        Path to the output directory
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

def calculate_metrics(original_text: str, summary_text: str) -> Dict[str, float]:
    """Calculate evaluation metrics for a summary.
    
    Args:
        original_text: The original text
        summary_text: The summary text
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Calculate compression ratio (summary length / original length)
    compression_ratio = len(summary_text) / len(original_text)
    
    # Calculate token count difference
    original_tokens = nltk.word_tokenize(original_text)
    summary_tokens = nltk.word_tokenize(summary_text)
    token_ratio = len(summary_tokens) / len(original_tokens)
    
    # Calculate sentence count difference
    original_sentences = nltk.sent_tokenize(original_text)
    summary_sentences = nltk.sent_tokenize(summary_text)
    sentence_ratio = len(summary_sentences) / len(original_sentences)
    
    return {
        'compression_ratio': compression_ratio,
        'token_ratio': token_ratio,
        'sentence_ratio': sentence_ratio,
        'original_length': len(original_text),
        'summary_length': len(summary_text),
        'original_tokens': len(original_tokens),
        'summary_tokens': len(summary_tokens),
        'original_sentences': len(original_sentences),
        'summary_sentences': len(summary_sentences)
    }

def visualize_comparison(extractive_metrics: Dict[str, float], 
                        abstractive_metrics: Dict[str, float],
                        output_path: str) -> str:
    """Visualize comparison between extractive and abstractive summaries.
    
    Args:
        extractive_metrics: Metrics for extractive summary
        abstractive_metrics: Metrics for abstractive summary
        output_path: Path to save the visualization
        
    Returns:
        Path to the saved visualization file
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot compression metrics
    metrics = ['compression_ratio', 'token_ratio', 'sentence_ratio']
    extractive_values = [extractive_metrics[m] for m in metrics]
    abstractive_values = [abstractive_metrics[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax1.bar(x - width/2, extractive_values, width, label='Extractive')
    ax1.bar(x + width/2, abstractive_values, width, label='Abstractive')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Compression', 'Token Ratio', 'Sentence Ratio'])
    ax1.set_ylabel('Ratio')
    ax1.set_title('Compression Metrics')
    ax1.legend()
    
    # Plot length metrics
    length_metrics = ['original_tokens', 'summary_tokens', 'original_sentences', 'summary_sentences']
    extractive_lengths = [extractive_metrics[m] for m in length_metrics]
    abstractive_lengths = [abstractive_metrics[m] for m in length_metrics]
    
    x = np.arange(len(length_metrics))
    
    ax2.bar(x - width/2, extractive_lengths, width, label='Extractive')
    ax2.bar(x + width/2, abstractive_lengths, width, label='Abstractive')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Orig Tokens', 'Sum Tokens', 'Orig Sent', 'Sum Sent'])
    ax2.set_ylabel('Count')
    ax2.set_title('Length Metrics')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save the figure
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    
    return output_path
