# Summarization Agent with Multi-Modal LLMs

This project implements an intelligent summarization agent that uses both extractive and abstractive summarization techniques powered by multi-modal LLMs. The agent processes text data (product reviews) and provides comprehensive summaries along with visual feedback.

## Features

- **Extractive Summarizer**: Extracts key sentences from the original text while preserving the original meaning
- **Abstractive Summarizer**: Generates a new summary in the model's own words using OpenAI's multi-modal capabilities
- **Comparison Report**: Analyzes and compares the performance of both summarization methods
- **Multi-Modal Feedback**: Provides visual representation of the summarization results

## Setup

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Install NLTK and download the 'punkt' tokenizer data:
   ```
   python -m nltk.downloader punkt
   ```

3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

4. Run the main script:
   ```
   python main.py
   ```

## Project Structure

- `main.py`: Entry point for the summarization agent
- `extractive_summarizer.py`: Implementation of the extractive summarization module
- `abstractive_summarizer.py`: Implementation of the abstractive summarization module
- `comparison.py`: Module for comparing and analyzing the summarization results
- `utils.py`: Utility functions for data processing and visualization
- `config.py`: Configuration settings for the project

## Dataset

The project uses a product reviews dataset (`Reviews.csv`) for summarization tasks.

## Usage

### Interactive Mode: Summarize Your Own Text
To summarize your own text, use the following command:
```
python main.py --mode interactive --text "Your text to summarize here"
```

### Batch Mode: Process Multiple Samples
To process multiple samples from the `Reviews.csv` file, use the following command:
```
python main.py --mode batch --samples 10
```
