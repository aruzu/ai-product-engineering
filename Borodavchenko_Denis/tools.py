from extractive import ExtractiveSummarizer
from abstractive import AbstractiveSummarizer
import os
import dotenv

dotenv.load_dotenv()

key = os.getenv("OPENAI_API_KEY")

extractive_summarizer = ExtractiveSummarizer()
abstractive_summarizer = AbstractiveSummarizer(key)

def extractive_summarization(text: str) -> str:
    print(f"Extractive summarization: {text[:20]}")
    return extractive_summarizer.summarize(text)

def abstractive_summarization(text: str) -> str:
    print(f"Abstractive summarization: {text[:20]}")
    return abstractive_summarizer.summarize(text)