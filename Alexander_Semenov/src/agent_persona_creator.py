"""
Module for creating user personas based on product reviews using OpenAI API.
"""

import pandas as pd
import logging
from typing import List, Dict
from src.llm_client import call_openai_api
from src.logger_config import setup_logger

class PersonaCreatorAgent:
    """Agent responsible for creating user personas from review data."""
    
    def __init__(self, api_key: str):
        """
        Initialize the Persona Creator Agent.
        
        Args:
            api_key (str): OpenAI API key for authentication
        """
        self.api_key = api_key
        self.logger = setup_logger(__name__)
    
    def _prepare_reviews_summary(self, reviews_df: pd.DataFrame, max_reviews: int = 100) -> str:
        """
        Prepare a summary of reviews for persona creation.
        
        Args:
            reviews_df (pd.DataFrame): DataFrame containing product reviews
            max_reviews (int): Maximum number of reviews to process
            
        Returns:
            str: Formatted string containing review summaries
            
        Raises:
            ValueError: If reviews_df is empty or required columns are missing
        """
        if reviews_df.empty:
            self.logger.warning("Empty DataFrame provided to _prepare_reviews_summary")
            return ""
        
        # Select required columns
        needed_columns = ['Score', 'Text']
        if not all(col in reviews_df.columns for col in needed_columns):
            missing_cols = [col for col in needed_columns if col not in reviews_df.columns]
            error_msg = f"Missing required columns: {', '.join(missing_cols)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Sample reviews if needed
        if len(reviews_df) > max_reviews:
            self.logger.info(
                f"Limiting reviews from {len(reviews_df)} to {max_reviews} for summary"
            )
            reviews_sample = reviews_df.sample(n=max_reviews, random_state=42)
        else:
            reviews_sample = reviews_df
        
        # Prepare summary text
        summary_parts = []
        
        # Add overall statistics
        avg_score = reviews_df['Score'].mean()
        score_dist = reviews_df['Score'].value_counts().sort_index()
        summary_parts.append(
            f"Overall Statistics:\n"
            f"- Average rating: {avg_score:.2f}\n"
            f"- Rating distribution:\n"
            + "\n".join(f"  {score} stars: {count} reviews" 
                       for score, count in score_dist.items())
        )
        
        # Add sampled reviews
        summary_parts.append("\nSample Reviews:")
        for idx, row in reviews_sample.iterrows():
            summary_parts.append(
                f"\nReview {idx + 1}:\n"
                f"Rating: {row['Score']} stars\n"
                f"Text: {row['Text']}"
            )
        
        return "\n\n".join(summary_parts)
    
    def create_personas(
        self, 
        reviews_df: pd.DataFrame, 
        num_personas: int = 4
    ) -> List[Dict[str, str]]:
        """
        Create user personas based on review data.
        
        Args:
            reviews_df (pd.DataFrame): DataFrame containing product reviews
            num_personas (int): Number of personas to generate
            
        Returns:
            List[Dict[str, str]]: List of generated personas, each containing
                                 'name', 'background', 'needs', and 'goals' keys
        """
        self.logger.info("Agent 2: Starting persona creation...")
        
        if reviews_df.empty:
            self.logger.warning("Empty DataFrame provided to create_personas")
            return []
        
        try:
            # Prepare review summary
            reviews_summary = self._prepare_reviews_summary(reviews_df)
            if not reviews_summary:
                self.logger.error("Failed to prepare reviews summary")
                return []
                
            self.logger.info(
                f"Prepared review summary for LLM processing "
                f"(length: {len(reviews_summary)} chars)"
            )
            
            # Placeholder return until we implement LLM processing
            return []
            
        except Exception as e:
            self.logger.error(f"Error during persona creation: {str(e)}")
            return []