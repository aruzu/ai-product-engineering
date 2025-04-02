from extractive_summarizer import extractive_summarize
from abstractive_summarizer import abstractive_summarize
import time

def get_article_text(file_path):
    """Read text from the local file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except Exception as e:
        return f"Error reading file: {str(e)}"

def compare_summaries(text):
    """Compare extractive and abstractive summaries side by side."""
    print("\n=== Original Text ===")
    print(text[:500] + "...\n")
    
    print("=== Extractive Summary (NLTK) ===")
    start_time = time.time()
    extractive_summary = extractive_summarize(text, num_sentences=5)
    extractive_time = time.time() - start_time
    print(extractive_summary)
    print(f"\nProcessing time: {extractive_time:.2f} seconds")
    
    print("\n=== Abstractive Summary (OpenAI) ===")
    start_time = time.time()
    abstractive_summary = abstractive_summarize(text, max_length=150)
    abstractive_time = time.time() - start_time
    print(abstractive_summary)
    print(f"\nProcessing time: {abstractive_time:.2f} seconds")
    
    print("\n=== Key Differences ===")
    print("1. Extractive Summary:")
    print("   - Uses exact sentences from the original text")
    print("   - Based on word frequency scoring")
    print("   - Faster processing time")
    print("   - May contain redundant information")
    
    print("\n2. Abstractive Summary:")
    print("   - Creates new sentences")
    print("   - More natural language")
    print("   - Better coherence")
    print("   - Slower processing time")
    print("   - Requires API key and internet connection")

if __name__ == "__main__":
    file_path = "oreilly_endofprogramming.txt"
    text = get_article_text(file_path)
    compare_summaries(text) 