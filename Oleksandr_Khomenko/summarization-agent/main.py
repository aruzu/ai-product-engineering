"""Main script for the Summarization Agent with Multi-Modal LLMs."""

import os
import time
import pandas as pd
import argparse
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# Import the custom modules
from extractive_summarizer import ExtractiveSummarizer
from abstractive_summarizer import AbstractiveSummarizer
from comparison import SummaryComparison
import utils
import config

# Import OpenAI Agents SDK
from openai import OpenAI

# Import NLTK
import nltk

# Load environment variables
load_dotenv()

class SummarizationAgent:
    """Agent for text summarization using both extractive and abstractive methods."""
    
    def __init__(self):
        """Initialize the summarization agent components."""
        self.extractive_summarizer = ExtractiveSummarizer(
            ratio=config.EXTRACTIVE_RATIO,
            min_length=config.MIN_LENGTH
        )
        
        self.abstractive_summarizer = AbstractiveSummarizer(
            model=config.ABSTRACTIVE_MODEL,
            max_tokens=config.MAX_TOKENS
        )
        
        self.comparison = SummaryComparison(output_dir=config.OUTPUT_DIR)
        
        # Initialize OpenAI client for agent functionality
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create output directory
        utils.create_output_dir(config.OUTPUT_DIR)
        
    def process_text(self, text: str, sample_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a single text for summarization and comparison.
        
        Args:
            text: The text to summarize
            sample_id: Optional identifier for the sample
            
        Returns:
            Dictionary containing all summarization results and comparison data
        """
        print(f"Processing sample {sample_id if sample_id else ''}...")
        
        # Clean and preprocess the text
        clean_text = utils.preprocess_text(text)
        
        # Generate extractive summary
        print("Generating extractive summary...")
        extractive_result = self.extractive_summarizer.get_best_summary(clean_text)
        
        # Generate abstractive summary
        print("Generating abstractive summary...")
        abstractive_result = self.abstractive_summarizer.summarize(clean_text)
        
        # Compare the summaries
        print("Comparing summaries...")
        comparison_result = self.comparison.compare_summaries(
            clean_text, extractive_result, abstractive_result
        )
        
        # Get multi-modal feedback
        print("Generating multi-modal feedback...")
        feedback_result = self.abstractive_summarizer.get_multi_modal_feedback(
            clean_text, extractive_result['summary'], abstractive_result['summary']
        )
        
        # Generate report
        print("Generating comparison report...")
        report_path = self.comparison.generate_report(
            comparison_result, sample_id=sample_id if sample_id else "sample"
        )
        
        # Combine all results
        results = {
            'original_text': clean_text,
            'extractive_summary': extractive_result,
            'abstractive_summary': abstractive_result,
            'comparison': comparison_result,
            'feedback': feedback_result,
            'report_path': report_path,
            'sample_id': sample_id
        }
        
        return results
    
    def process_batch(self, texts: List[str], sample_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Process a batch of texts for summarization and comparison.
        
        Args:
            texts: List of texts to summarize
            sample_ids: Optional list of identifiers for the samples
            
        Returns:
            List of dictionaries containing all summarization results and comparison data
        """
        results = []
        
        for i, text in enumerate(texts):
            sample_id = sample_ids[i] if sample_ids and i < len(sample_ids) else f"sample_{i}"
            result = self.process_text(text, sample_id)
            results.append(result)
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
        
        # Generate aggregate report
        if len(results) > 1:
            comparisons = [result['comparison'] for result in results]
            self.comparison.save_aggregate_report(comparisons)
            
        return results
    
    def agent_api_summary(self, text: str) -> Dict[str, Any]:
        """Generate a summary using the OpenAI Agents API.
        
        This demonstrates how to use the OpenAI Agents SDK to create a summarization agent.
        
        Args:
            text: The text to summarize
            
        Returns:
            Dictionary containing the agent API generated summary and metadata
        """
        # Use the OpenAI Chat Completions API with system instructions to simulate an agent
        # In a full implementation, this would use the Assistants API with specialized tools
        response = self.client.chat.completions.create(
            model=config.ABSTRACTIVE_MODEL,
            messages=[
                {"role": "system", "content": "You are an AI assistant specialized in summarization. You'll be given a text to summarize. Provide both an extractive summary (key sentences from the original) and an abstractive summary (in your own words)."},
                {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
            ],
            max_tokens=config.MAX_TOKENS * 2,
            temperature=0.3,
        )
        
        return {
            'agent_summary': response.choices[0].message.content,
            'model': config.ABSTRACTIVE_MODEL,
            'tokens_used': response.usage.total_tokens if hasattr(response, 'usage') else None
        }

def process_csv_data(file_path: str, sample_size: int) -> None:
    """Process data from a CSV file for summarization.
    
    Args:
        file_path: Path to the CSV file
        sample_size: Number of samples to process
    """
    # Load the data
    df = utils.load_data(file_path, sample_size, config.RANDOM_SEED)
    
    # Initialize the agent
    agent = SummarizationAgent()
    
    # Process the texts
    texts = df['Text'].tolist()
    sample_ids = [f"review_{i}" for i in range(len(texts))]
    
    print(f"Processing {len(texts)} samples...")
    results = agent.process_batch(texts, sample_ids)
    
    # Save the individual results
    for i, result in enumerate(results):
        sample_id = result['sample_id']
        
        # Save both summaries to files
        with open(f"{config.OUTPUT_DIR}/{sample_id}_extractive.txt", "w") as f:
            f.write(result['extractive_summary']['summary'])
            
        with open(f"{config.OUTPUT_DIR}/{sample_id}_abstractive.txt", "w") as f:
            f.write(result['abstractive_summary']['summary'])
            
        # Save the feedback
        with open(f"{config.OUTPUT_DIR}/{sample_id}_feedback.txt", "w") as f:
            f.write(result['feedback']['feedback'])
    
    print(f"Processing completed. Results saved to {config.OUTPUT_DIR}/")

def main():
    """Main function to run the summarization agent."""
    parser = argparse.ArgumentParser(description="Summarization Agent with Multi-Modal LLMs")
    parser.add_argument(
        "--mode", 
        choices=["sample", "interactive", "batch"], 
        default="sample",
        help="Mode to run the agent in"
    )
    parser.add_argument(
        "--samples", 
        type=int, 
        default=5,
        help="Number of samples to process in batch mode"
    )
    parser.add_argument(
        "--text", 
        type=str,
        help="Text to summarize in interactive mode"
    )
    
    args = parser.parse_args()
    
    # Create an instance of the agent
    agent = SummarizationAgent()
    
    if args.mode == "interactive" and args.text:
        # Process a single text provided via command line
        result = agent.process_text(args.text)
        
        print("\nEXTRACTIVE SUMMARY:")
        print(result['extractive_summary']['summary'])
        
        print("\nABSTRACTIVE SUMMARY:")
        print(result['abstractive_summary']['summary'])
        
        print("\nFEEDBACK:")
        print(result['feedback']['feedback'])
        
        print(f"\nReport saved to {result['report_path']}")
        
    elif args.mode == "batch":
        # Process multiple samples from the CSV file
        process_csv_data(config.DATA_FILE, args.samples)
        
    else:  # Sample mode - showcase the agent with a sample text
        sample_text = """The new smartphone exceeded my expectations in every way. The camera quality is exceptional, 
        capturing detailed photos even in low light conditions. Battery life is impressive, lasting a full day with 
        heavy use. The display is vibrant with true-to-life colors and excellent brightness. The processor handles 
        everything smoothly, from everyday tasks to demanding games without any lag. The build quality feels premium, 
        with a sleek design that's comfortable to hold. The new AI features are truly useful rather than just gimmicks, 
        especially the real-time translation and enhanced photo editing capabilities. The only minor downsides are the 
        lack of expandable storage and the removal of the headphone jack, but these are small compromises given the 
        overall excellence of this device. I highly recommend this phone to anyone looking for a high-performance, 
        feature-rich smartphone with an amazing camera system."""
        
        result = agent.process_text(sample_text, "sample_demo")
        
        print("\nORIGINAL TEXT:")
        print(sample_text)
        
        print("\nEXTRACTIVE SUMMARY:")
        print(result['extractive_summary']['summary'])
        
        print("\nABSTRACTIVE SUMMARY:")
        print(result['abstractive_summary']['summary'])
        
        print("\nFEEDBACK:")
        print(result['feedback']['feedback'])
        
        print(f"\nReport saved to {result['report_path']}")
        
        # Demonstrate the OpenAI Agents API approach
        print("\nOPENAI AGENTS API SUMMARY:")
        agent_result = agent.agent_api_summary(sample_text)
        print(agent_result['agent_summary'])

if __name__ == "__main__":
    main()
