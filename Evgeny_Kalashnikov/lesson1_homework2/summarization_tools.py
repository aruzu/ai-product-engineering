from agents import function_tool
import nltk
import ssl
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import time
from openai import OpenAI
from dotenv import load_dotenv
import os

# Fix SSL certificate verification for NLTK
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Initialize NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Initialize OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

@function_tool
def extractive_summarize(text: str, num_sentences: int) -> str:
    """Summarize text using extractive summarization (NLTK).
    
    Args:
        text: The text to summarize
        num_sentences: Number of sentences to include in the summary
    """
    start_time = time.time()
    
    # Tokenize sentences
    sentences = sent_tokenize(text)
    # Tokenize words and remove stopwords
    words = [word.lower() for word in word_tokenize(text) 
            if word.isalnum() and word.lower() not in stop_words]
    
    # Calculate word frequencies
    word_freq = FreqDist(words)
    
    # Calculate sentence scores
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                if i not in sentence_scores:
                    sentence_scores[i] = word_freq[word]
                else:
                    sentence_scores[i] += word_freq[word]
    
    # Get top sentences
    top_sentences = sorted(sentence_scores.items(), 
                         key=lambda x: x[1], reverse=True)[:num_sentences]
    top_sentences = sorted(top_sentences, key=lambda x: x[0])
    
    # Generate summary
    summary = ' '.join([sentences[i] for i, _ in top_sentences])
    
    return {
        'summary': summary,
        'execution_time': time.time() - start_time,
        'num_sentences': len(summary.split('.')),
        'num_words': len(summary.split())
    }

@function_tool
def abstractive_summarize(text: str) -> str:
    """Summarize text using abstractive summarization (OpenAI)."""
    start_time = time.time()
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates concise summaries."},
                {"role": "user", "content": f"Please provide a concise summary of the following text in 2-3 sentences:\n\n{text}"}
            ],
            max_tokens=150,
            temperature=0.7
        )
        
        summary = response.choices[0].message.content.strip()
        
        return {
            'summary': summary,
            'execution_time': time.time() - start_time,
            'num_sentences': len(summary.split('.')),
            'num_words': len(summary.split())
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'execution_time': time.time() - start_time
        } 