import argparse
import os
from dotenv import load_dotenv
import logging
import asyncio
from src.data_loader import load_reviews
from src.logger_config import setup_logger
from src.feature_generator import FeatureGenerator
from src.agent_persona_creator import PersonaCreatorAgent
from src.reviews_preparer import prepare_reviews
from agents import Agent, Runner
from src.llm_client import call_openai_api

async def main():
    """
    Main function that handles the review analysis and focus group simulation.
    
    This function:
    1. Processes command line arguments
    2. Validates environment variables
    3. Loads and processes the CSV file with reviews
    4. Performs review analysis
    5. Simulates focus group discussions
    """
    # Set up logging
    logger = setup_logger(__name__)
    logger.info("Starting pipeline...")
    
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Analyze product reviews and simulate focus group discussions'
    )
    parser.add_argument(
        '--csv-path', '-c',
        required=True,
        help='Path to the CSV file containing product reviews'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get CSV path from arguments
    csv_path = args.csv_path
    
    # Load environment variables
    load_dotenv()
    
    # Get OpenAI API key from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # Check if API key exists
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        exit(1)
    
    # Load and validate reviews
    try:
        reviews_df = load_reviews(csv_path)
        logger.info(f"Successfully loaded {len(reviews_df)} reviews")
    except FileNotFoundError:
        logger.error(f"CSV file not found at path: {csv_path}")
        exit(1)
    except ValueError as e:
        logger.error(str(e))
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}")
        exit(1)

    # Prepare formatted reviews
    reviews = ""
    try:
        reviews = prepare_reviews(reviews_df, max_reviews=10)
        logger.info("\nFormatted reviews:")
        for review in reviews:
            logger.info(f"\n{review}")
    except ValueError as e:
        logger.error(str(e))
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}")
        exit(1)
    
    #Init senior product manager agent
    agent = Agent(
        name="SeniorProductManagerAgent",
        instructions="""
        You are a senior product manager with extensive experience working on digital products.
        Your primary tasks are to analyze user reviews and feedback, identify key problems and user pain points, formulate specific product tasks and features to address these issues, and build a target user profile based on the reviews.
        You are also responsible for conducting user research when necessary.
        Respond clearly and professionally.
        """,
        model="gpt-4",
    )

    prompt = """
        You are a Senior Product Manager with extensive experience in digital products.
        I’m going to share a set of user reviews with you.
        Your task is to deeply analyze the feedback and identify the key user pain points.

        Here’s how you should approach it:
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

        Feature Name
        Description (up to 7 sentences in one paragraph)

        Feature Name
        Description (up to 7 sentences in one paragraph)

        Feature Name
        Description (up to 7 sentences in one paragraph)

        ---
        Reviews to analyze:
        {reviews_text}
    """.format(reviews_text=reviews)

    result = await Runner.run(agent, prompt)
    print(result.final_output)
    
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    asyncio.run(main())