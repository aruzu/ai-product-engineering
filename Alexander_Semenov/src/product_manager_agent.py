"""
Module for Product Manager agent that analyzes reviews and generates feature suggestions.
"""

import asyncio
from agents import Agent, Runner
from typing import List
from src.logger_config import setup_logger
from src.llm_client import call_openai_api

class ProductManagerAgent:
    """
    Agent that acts as a senior product manager to analyze reviews and suggest features.
    """
    
    def __init__(self):
        self.agent = Agent(
            name="SeniorProductManagerAgent",
            instructions="""
            You are a senior product manager with extensive experience working on digital products.
            Your primary tasks are to analyze user reviews and feedback, identify key problems and user pain points, formulate specific product tasks and features to address these issues, and build a target user profile based on the reviews.
            You are also responsible for conducting user research when necessary.
            Respond clearly and professionally.
            """,
            model="gpt-4",
        )
    
    def _parse_features_from_response(self, response: str) -> List[str]:
        """
        Parse features from LLM response into a list.
        
        Args:
            response (str): Raw LLM response text
            
        Returns:
            List[str]: List of feature descriptions, each containing feature name and description
        """
        if not response:
            return []
            
        features = []
        # Split response by '---' delimiter and remove empty blocks
        feature_blocks = [block.strip() for block in response.split('---') if block.strip()]
        
        for block in feature_blocks:
            try:
                # Find feature name and description using the prefixes
                name_start = block.find("Feature Name:")
                desc_start = block.find("Description:")
                
                if name_start != -1 and desc_start != -1:
                    # Extract name (text between "Feature Name:" and "Description:")
                    name = block[name_start + len("Feature Name:"):desc_start].strip()
                    # Extract description (text after "Description:")
                    description = block[desc_start + len("Description:"):].strip()
                    # Format the feature string
                    feature = f"Feature Name: {name}\nDescription: {description}"
                    features.append(feature)
            except Exception as e:
                logger = setup_logger(__name__)
                logger.warning(f"Error parsing feature block: {str(e)}")
                continue
        
        return features
        
    async def identify_key_user_pain_points(self, reviews: List[str]) -> List[str]:
        """
        Analyze user reviews to identify key pain points and problems.
        
        Args:
            reviews (List[str]): List of user reviews to analyze
            
        Returns:
            List[str]: List of identified features, where each feature includes its name and description
        """
        logger = setup_logger(__name__)
        
        prompt = """
You are a Senior Product Manager with extensive experience in digital products.
I'm going to share a set of user reviews with you.
Your task is to deeply analyze the feedback and identify the key user pain points.

Here's how you should approach it:
1. Carefully read all the reviews. Pay attention not only to explicit complaints but also to subtle signs of frustration, unmet needs, or unspoken expectations.
2. Group similar feedback together. If multiple reviews touch on the same issue (e.g., "slow loading" and "freezes on startup"), treat them as one common problem.
3. Based on these insights, define 3 new product features. Focus on the issues that are the most critical or most frequently mentioned.

Feature writing guidelines:
One feature = one paragraph.
* Each paragraph should be clear, cohesive, and no longer than 7 sentences.
* The description must explain which user problem it addresses, how the feature works, and what benefit it brings to users.
* Avoid vague statements or generalities — be specific, professional, and actionable.
* Focus on features that would directly solve the identified problems and could realistically be implemented.
* Write in a clear, logical, and structured way. Avoid phrases like "users are unhappy" — instead, clearly define problems in a way that can immediately guide solution design.
* The output must contain only the list of features in the specified format — no additional summaries, explanations, or conclusions.

Response format:

Feature Name: [Feature Name]
Description: [Feature description in one paragraph]
---
Feature Name: [Feature Name]
Description: [Feature description in one paragraph]
---
Feature Name: [Feature Name]
Description: [Feature description in one paragraph]

---
Reviews to analyze:
{reviews_text}
        """.format(reviews_text="\n\n".join(reviews))

        result = await Runner.run(self.agent, prompt)
        return self._parse_features_from_response(result.final_output)