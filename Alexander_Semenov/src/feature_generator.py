"""
Module for generating features from product reviews using OpenAI API.
"""

import re
import pandas as pd
import logging
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
    
    def _prepare_llm_input(self, reviews_df: pd.DataFrame) -> str:
        """
        Prepare reviews data for LLM input by formatting.
        
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
    
    def _parse_features(self, llm_response: str) -> List[Dict[str, str]]:
        """
        Parse the LLM response to extract structured feature information.
        
        Args:
            llm_response (str): Raw response from the LLM
            
        Returns:
            List[Dict[str, str]]: List of dictionaries containing parsed features
                Each dict has 'name', 'problem', and 'solution' keys
        """
        if not llm_response:
            self.logger.warning("Empty LLM response provided to _parse_features")
            return []
        
        features = []
        
        # Split response into feature blocks
        feature_blocks = re.split(r'\nFEATURE:', llm_response)
        
        # Remove any empty blocks and process each feature
        for block in feature_blocks:
            if not block.strip():
                continue
                
            try:
                # Parse feature components using regex
                name_match = re.search(r'^(.+?)(?:\nPROBLEM:|$)', block, re.MULTILINE)
                problem_match = re.search(r'PROBLEM:(.+?)(?:\nSOLUTION:|$)', block, re.MULTILINE | re.DOTALL)
                solution_match = re.search(r'SOLUTION:(.+?)$', block, re.MULTILINE | re.DOTALL)
                
                if name_match:
                    feature = {
                        'name': name_match.group(1).strip(),
                        'problem': problem_match.group(1).strip() if problem_match else '',
                        'solution': solution_match.group(1).strip() if solution_match else ''
                    }
                    features.append(feature)
                    
            except Exception as e:
                self.logger.warning(f"Error parsing feature block: {str(e)}")
                continue
        
        return features

    def generate_features(self, reviews_df: pd.DataFrame) -> List[Dict[str, str]]:
        """
        Analyze reviews and generate a list of product features.
        
        Args:
            reviews_df (pd.DataFrame): DataFrame containing product reviews
            
        Returns:
            List[Dict[str, str]]: List of dictionaries containing extracted features
                Each dict has 'name', 'problem', and 'solution' keys
            
        Raises:
            ValueError: If reviews_df is empty
        """
        self.logger.info("Starting feature generation...")
        
        if reviews_df.empty:
            self.logger.warning("Empty DataFrame provided to generate_features")
            return []
        
        # Prepare input for the LLM
        formatted_input = self._prepare_llm_input(reviews_df)
        if not formatted_input:
            self.logger.warning("No valid input text generated from reviews")
            return []
            
        self.logger.info(
            f"Prepared input for LLM processing (length: {len(formatted_input)} chars)"
        )
        
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
            
        self.logger.debug(f"Raw LLM response: {llm_response[:500]}...")
        
        # Parse features from the response
        features = self._parse_features(llm_response)
        self.logger.info(f"Successfully extracted {len(features)} features")
        
        return features