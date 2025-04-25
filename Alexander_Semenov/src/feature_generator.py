"""
Module for generating features from product reviews using OpenAI API.
"""

import pandas as pd
from typing import List, Dict
from src.llm_client import call_openai_api
from src.logger_config import setup_logger

# Prompt template for feature generation
FEATURE_GENERATION_PROMPT = """
Below is a list of user reviews about the product. Analyze them and identify 2–3 features that would help address the most common or most critical issues mentioned in the reviews.
Make sure that:
1. Each feature describes how it solves a specific problem, not just restates it.
2. Wording is concise but specific.
3. Features should not be abstract — describe how they could be implemented or how they would help the user.

Format each feature exactly as follows:: 
FEATURE: <feature name>
PROBLEM: <problem/need description>
SOLUTION: <how feature addresses the problem>

Reviews to analyze:
{reviews_text}
"""

class FeatureGenerator:
    """
    Responsible for analyzing reviews and generating product features.
    Uses OpenAI API to process review texts and extract meaningful features.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the Feature Generator.
        
        Args:
            api_key (str): OpenAI API key for authentication
        """
        self.api_key = api_key
        self.logger = setup_logger(__name__)

    def generate_features(self, reviews_df: pd.DataFrame) -> List[Dict[str, str]]:
        """
        Analyze reviews and generate a list of product features.
        
        Args:
            reviews_df (pd.DataFrame): DataFrame containing product reviews
            
        Returns:
            List[Dict[str, str]]: List of dictionaries containing extracted features
                Each dict has 'name', 'problem', and 'solution' keys
        """
        self.logger.info("Starting feature generation...")
        
        if reviews_df.empty:
            self.logger.warning("Empty DataFrame provided to generate_features")
            return []
            
        # Format reviews into a single string
        formatted_reviews = []
        for idx, row in reviews_df.iterrows():
            review_text = f"Review {idx + 1}:\n"
            if 'Summary' in reviews_df.columns and pd.notna(row['Summary']):
                review_text += f"Summary: {row['Summary']}\n"
            review_text += f"Text: {row['Text']}\n"
            formatted_reviews.append(review_text)
        
        formatted_input = "\n".join(formatted_reviews)
        
        # Format the full prompt
        full_prompt = FEATURE_GENERATION_PROMPT.format(reviews_text=formatted_input)
        
        # Call OpenAI API
        llm_response = call_openai_api(
            prompt=full_prompt,
            api_key=self.api_key
        )
        
        if llm_response is None:
            self.logger.error("Failed to get response from OpenAI API")
            return []
            
        # Parse features from the response
        features = []
        feature_blocks = llm_response.split('FEATURE:')[1:]  # Skip first empty split
        
        for block in feature_blocks:
            try:
                lines = block.strip().split('\n')
                feature_name = lines[0].strip()
                
                problem_start = block.find('PROBLEM:')
                solution_start = block.find('SOLUTION:')
                
                if problem_start != -1 and solution_start != -1:
                    problem = block[problem_start + 8:solution_start].strip()
                    solution = block[solution_start + 9:].strip()
                    
                    feature = {
                        'name': feature_name,
                        'problem': problem,
                        'solution': solution
                    }
                    features.append(feature)
                    
            except Exception as e:
                self.logger.warning(f"Error parsing feature block: {str(e)}")
                continue
        
        self.logger.info(f"Successfully extracted {len(features)} features")
        return features