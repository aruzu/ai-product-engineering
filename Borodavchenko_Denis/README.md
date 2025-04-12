# Text Summarization Analysis Tool

A Python application that analyzes and compares extractive and abstractive text summarization techniques for both textual and image content.

## Overview

This project provides a framework for analyzing different text summarization approaches:

- **Extractive Summarization**: Uses a graph-based TextRank algorithm to select important sentences from the original text
- **Abstractive Summarization**: Leverages OpenAI's GPT models to generate concise summaries in new words

The tool can process both textual content and images containing text, generating comparative reports on the effectiveness of each summarization method.

## How It Works

### Architecture

The system uses a LangGraph-based agent architecture to orchestrate the summarization workflow:

1. **Input Processing**: The system accepts both text and images as input
   - Text is processed directly
   - Images are converted to base64 and sent to the LLM for text extraction

2. **Agent Workflow**:
   - An agent powered by OpenAI's GPT models manages the process
   - The agent routes inputs to appropriate summarization tools
   - A state machine (StateGraph) manages the workflow transitions

3. **Report Generation**: After both summarization methods are applied, the system generates a comparative report

### Extractive Summarization

The extractive summarizer uses the TextRank algorithm:

1. The text is split into sentences
2. A similarity matrix is built by calculating cosine similarity between each pair of sentences
3. A graph is constructed where:
   - Nodes represent sentences
   - Edges represent similarity scores between sentences
4. PageRank algorithm is applied to rank sentences by importance
5. Top-ranked sentences are selected and combined to form the summary

This approach selects actual sentences from the original text without modifying them.

### Abstractive Summarization

The abstractive summarizer leverages OpenAI's GPT models:

1. The text is sent to OpenAI's API with a prompt requesting a concise summary
2. The model generates a completely new summary using its own language and structure
3. The generated summary may contain phrases and sentences not present in the original text

This approach creates a more fluid and potentially creative summary.

### Comparison Process

The system generates reports that analyze both summarization methods, focusing on:

1. **Coherence and Readability**: How well the summary flows and is understood
2. **Information Retention**: How much key information is preserved
3. **Length and Conciseness**: How efficiently the summary conveys information
4. **Style Differences**: How the extractive and abstractive approaches differ in tone and structure

## Features

- Process text from CSV files or direct input
- Extract text from images for summarization
- Compare extractive and abstractive summarization techniques
- Generate detailed reports on summarization performance
- Utilizes LangGraph for orchestrating the summarization workflow

## Requirements

- Python 3.8+
- OpenAI API key
- NLTK and related packages
- Pandas
- Pillow
- LangChain
- LangGraph
- KaggleHub

## Installation

1. Clone the repository
2. Create a virtual environment:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install the required packages:
```
pip install langchain-openai langchain-core langgraph openai pillow pandas nltk kagglehub networkx numpy python-dotenv
```
4. Download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```
5. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

Run the main script:

```
python main.py
```

The script will:
1. Download product review data from Kaggle
2. Process both text reviews and image-based reviews
3. Generate summaries using both extractive and abstractive techniques
4. Create reports comparing the summarization methods
5. Save reports as markdown files

## Project Structure

- `main.py`: Main entry point for the application
- `agent.py`: Implements the LangGraph agent for orchestrating the summarization workflow
- `tools.py`: Defines tools for extractive and abstractive summarization
- `extractive.py`: Implements TextRank-based extractive summarization
- `abstractive.py`: Implements GPT-based abstractive summarization
- `review_1.jpg` & `review_2.jpg`: Sample image-based reviews for testing

## License

MIT

## Contributing

Contributions are welcome. Please feel free to submit a Pull Request. 