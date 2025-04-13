# Summarization Agent

## Project Overview
This project implements an agent that compares two types of text summarization approaches:
1. Extractive Summarization (using NLTK)
2. Abstractive Summarization (using OpenAI's o3-mini model)

The objective is to analyze and compare the performance of these two different summarization techniques on Amazon product reviews, focusing on:
- Execution time
- Summary length
- Summary clarity and readability
- Naturalness of the output

## Tools and Technologies Used

### Core Technologies
- **Python 3.8+**: Primary programming language
- **OpenAI Agents SDK**: For orchestrating the summarization comparison process
- **NLTK**: For extractive summarization using text processing capabilities
- **OpenAI API**: For abstractive summarization using the o3-mini model

### Libraries and Tools
- **pandas**: For data handling and manipulation
- **tqdm**: For progress tracking during processing
- **kaggle**: For dataset access and management
- **python-dotenv**: For environment variable management

### Dataset
- **Amazon Product Reviews Dataset**: Used as the source data for summarization comparison
  - Contains real-world product reviews
  - Provides diverse text samples for testing
  - Available through Kaggle

## Setup Instructions

### Prerequisites
1. Python 3.8 or higher
2. OpenAI API key
3. Kaggle account (for dataset access)

### Installation Steps

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

4. Configure Kaggle credentials:
   - Go to your Kaggle account settings
   - Create a new API token
   - Place the `kaggle.json` file in `~/.kaggle/` directory

### Running the Project

Execute the main script:
```bash
python main.py
```

The script will:
1. Load 3 random products from the Amazon reviews dataset
2. Generate summaries using both approaches
3. Compare the results
4. Save the results to `summarization_results.txt`

### Output
The script provides real-time output showing:
- Extractive summary
- Abstractive summary
- Execution times for both approaches
- Comparison metrics

Results are saved to `summarization_results.txt` for further analysis.

## Project Structure
```
.
├── main.py                 # Main script for running the comparison
├── summarization_tools.py  # Contains summarization functions
├── dataset_handler.py      # Handles dataset loading and processing
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Features

- Uses OpenAI Agents SDK for orchestration
- Extractive summarization using NLTK's text processing capabilities
- Abstractive summarization using OpenAI's GPT-3.5-turbo
- Comparison of both approaches based on:
  - Execution time
  - Summary length
  - Summary clarity
- Uses the Amazon Product Reviews dataset from Kaggle
- Caches dataset locally to avoid repeated downloads

## Architecture

The implementation uses the OpenAI Agents SDK to create a comparison agent that:
1. Uses function tools for both summarization approaches
2. Orchestrates the comparison process
3. Provides detailed analysis of the results

The agent is configured with:
- Clear instructions for comparison
- Access to both summarization tools
- GPT-3.5-turbo as the underlying model 