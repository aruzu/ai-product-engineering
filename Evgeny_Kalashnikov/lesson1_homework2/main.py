from agents import Agent, Runner
from summarization_tools import extractive_summarize, abstractive_summarize
from dataset_handler import DatasetHandler
import json
from tqdm import tqdm
from typing import Dict, Any
import os

class SummarizationAgent:
    def __init__(self):
        self.agent = Agent(
            name="Summarization Comparison Agent",
            instructions="""
            You should compare two types of summarization approaches:
            1. Extractive summarization using NLTK (use 3 sentences for the summary)
            2. Abstractive summarization using OpenAI
            
            For each text, you should:
            1. Generate summaries using both approaches
            2. Compare the results based on:
                - Execution time
                - Summary length
                - Summary clarity
            3. Provide a detailed comparison of the results
            
            When using the extractive_summarize tool, always set num_sentences=3.
            
            Format your response as follows:
            Extractive Summary:
            [summary text]
            
            Abstractive Summary:
            [summary text]
            
            Comparison:
            [detailed comparison]
            """,
            tools=[extractive_summarize, abstractive_summarize],
            model="o3-mini"
        )
        self.dataset_handler = DatasetHandler()
    
    async def compare_summaries(self, text: str) -> str:
        runner = Runner()
        result = await runner.run(
            self.agent,
            f"Please compare summaries for the following text:\n\n{text}"
        )
        return result.final_output
    
    async def run_comparison(self, num_products: int = 3):
        print("Loading product reviews...")
        product_reviews = self.dataset_handler.get_random_products(num_products)
        
        # Create output file
        output_file = 'summarization_results.txt'
        with open(output_file, 'w') as f:
            f.write("Summarization Comparison Results\n")
            f.write("=" * 50 + "\n\n")
        
        for product_id, reviews in tqdm(product_reviews.items(), desc="Processing products"):
            print(f"\nProcessing product {product_id}...")
            num_reviews = len(reviews.split('.'))  # Rough estimate of number of reviews
            print(f"Number of reviews for this product: {num_reviews}")
            
            results = await self.compare_summaries(reviews)
            
            # Format output
            output = f"""
Product ID: {product_id}
Number of reviews: {num_reviews}

{results}

{'=' * 50}
"""
            # Print to console
            print(output)
            
            # Write to file
            with open(output_file, 'a') as f:
                f.write(output)
        
        print(f"\nResults have been saved to {output_file}")

if __name__ == "__main__":
    import asyncio
    agent = SummarizationAgent()
    asyncio.run(agent.run_comparison()) 