import argparse
import os
from dotenv import load_dotenv
import logging
from src.data_loader import load_reviews
from src.logger_config import setup_logger
from src.feature_generator import FeatureGenerator
from src.agent_persona_creator import PersonaCreatorAgent
from src.reviews_preparer import prepare_reviews

def main():
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
    try:
        formatted_reviews = prepare_reviews(reviews_df, max_reviews=50)
        logger.info("\nFormatted reviews:")
        for review in formatted_reviews:
            logger.info(f"\n{review}")
    except ValueError as e:
        logger.error(str(e))
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}")
        exit(1)
    
    # Initialize feature generator
    feature_generator = FeatureGenerator(api_key=openai_api_key)
    
    # Generate features
    features = feature_generator.generate_features(reviews_df)
    logger.info(f"Generated {len(features)} features")
            
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    main()