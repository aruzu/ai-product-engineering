"""
Module for generating features from product reviews using OpenAI API.
"""

import pandas as pd
import logging
from typing import List, Optional


class FeatureGeneratorAgent:
    """
    Agent responsible for analyzing reviews and generating product features.
    Uses OpenAI API to process review texts and extract meaningful features.
    """
    
    def __init__(self, api_key: str, max_reviews: int = 100):
        """
        Initialize the Feature Generator Agent.
        
        Args:
            api_key (str): OpenAI API key for authentication
            max_reviews (int): Maximum number of reviews to process, defaults to 100
        """
        self.api_key = api_key
        self.max_reviews = max_reviews
        self.logger = logging.getLogger(__name__)
    
    def _prepare_llm_input(self, reviews_df: pd.DataFrame) -> str:
        """
        Prepare reviews data for LLM input by formatting and limiting the number of reviews.
        
        Args:
            reviews_df (pd.DataFrame): DataFrame containing product reviews
            
        Returns:
            str: Formatted string containing review summaries and texts
            
        Raises:
            ValueError: If reviews_df is empty or required columns are missing
        """
        if reviews_df.empty:
            self.logger.warning("Empty DataFrame provided to _prepare_llm_input")
            return ""
            
        # Select required columns, handle missing Summary column
        if 'Summary' in reviews_df.columns:
            selected_df = reviews_df[['Summary', 'Text']].copy()
        else:
            selected_df = reviews_df[['Text']].copy()
            selected_df['Summary'] = ''  # Empty summaries if column doesn't exist
        
        # Limit number of reviews if needed
        if len(selected_df) > self.max_reviews:
            self.logger.info(
                f"Limiting reviews from {len(selected_df)} to {self.max_reviews}"
            )
            selected_df = selected_df.sample(n=self.max_reviews, random_state=42)
        
        # Format reviews into a single string
        formatted_reviews = []
        for idx, row in selected_df.iterrows():
            if row['Summary']:  # Only include summary if it exists
                formatted_reviews.append(
                    f"Review {idx + 1}:\nSummary: {row['Summary']}\n"
                    f"Text: {row['Text']}\n"
                )
            else:
                formatted_reviews.append(
                    f"Review {idx + 1}:\nText: {row['Text']}\n"
                )
        
        return "\n".join(formatted_reviews)
    
    def generate_features(self, reviews_df: pd.DataFrame) -> List[str]:
        """
        Analyze reviews and generate a list of product features.
        
        Args:
            reviews_df (pd.DataFrame): DataFrame containing product reviews
            
        Returns:
            List[str]: List of extracted product features
            
        Raises:
            ValueError: If reviews_df is empty
        """
        self.logger.info("Agent 1: Starting feature generation...")
        
        if reviews_df.empty:
            self.logger.warning("Empty DataFrame provided to generate_features")
            return []
        
        # Prepare input for the LLM
        formatted_input = self._prepare_llm_input(reviews_df)
        self.logger.info(
            f"Prepared input for LLM processing (length: {len(formatted_input)} chars)"
        )
        
        # TODO: Implement actual feature generation using OpenAI API
        # For now return empty list as placeholder
        self.logger.info("Agent 1: Completed data preparation...")
        return []