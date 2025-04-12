"""Configuration settings for the summarization agent."""

# Data settings
DATA_FILE = "Reviews.csv"
SAMPLE_SIZE = 1000  # Number of reviews to process for demonstration
RANDOM_SEED = 42

# Extractive summarizer settings
EXTRACTIVE_RATIO = 0.2  # Proportion of original text to keep in extractive summary
MIN_LENGTH = 40  # Minimum length of sentences to consider

# Abstractive summarizer settings
ABSTRACTIVE_MODEL = "gpt-4o"  # OpenAI model to use for abstractive summarization
MAX_TOKENS = 500  # Maximum tokens for generated abstractive summary

# Visualization settings
FIGURE_SIZE = (12, 8)
DPI = 300
OUTPUT_DIR = "outputs"

# OpenAI API settings
TIMEOUT = 60  # API request timeout in seconds
