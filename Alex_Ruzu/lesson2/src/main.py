"""
Main entry point for the cluster-driven product development pipeline
"""

import asyncio
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from pipeline import run_pipeline
from virtual_board import VirtualUserBoard
from data_types import PipelineConfig
from utils import setup_logging, ValidationError
from pydantic import ValidationError as PydanticValidationError

# Load environment variables
load_dotenv()

# Set up logging with both files
logger = setup_logging(log_file="logs/main.log")
# Add pipeline logging handler
setup_logging(log_file="logs/pipeline.log")

async def main():
    """
    Main entry point for the cluster-driven pipeline with OpenAI Agents SDK virtual board
    """
    print("ALEX_RUZU LESSON2 SOLUTION")
    print("Cluster-Driven Product Development Pipeline")
    print("Using Complete Dataset Analysis + OpenAI Agents SDK")
    print()

    try:
        # Load and validate configuration using Pydantic
        try:
            config = PipelineConfig.from_env()
            logger.info("Configuration validated successfully")
            logger.info(f"Config summary: {config.get_summary()}")
        except (ValueError, PydanticValidationError) as e:
            logger.error(f"Configuration validation failed: {e}")
            return

        # Verify environment setup
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY environment variable is required")
            logger.error("Please set your API key in the .env file")
            return

        # Check if CSV file exists
        if not os.path.exists(config.csv_path):
            logger.error(f"CSV file not found: {config.csv_path}")
            logger.error("Please ensure the viber.csv file exists in the data/ directory")
            return

        logger.info(f"Using CSV file: {config.csv_path}")
        logger.info(f"Configuration: {config}")

        # Execute the complete pipeline
        logger.info("Starting Complete Pipeline Execution...")
        logger.info("Phase 1: Clustering Analysis (Full Dataset)")
        logger.info("Phase 2: Persona Extraction from Clusters")
        logger.info("Phase 3: Feature Generation from Pain Points")
        logger.info("Phase 4: Results Persistence")
        logger.info("Phase 5: OpenAI Agents SDK Virtual Board")

        start_time = datetime.now()

        # Add progress indicators for user experience
        # Read CSV to get actual count
        try:
            df = pd.read_csv(config.csv_path)
            review_count = len(df)
            print(f"\n🔄 Analyzing {review_count:,} Viber reviews...")
        except Exception as e:
            logger.warning(f"Could not read CSV for count: {e}")
            print(f"\n🔄 Analyzing Viber reviews from {config.csv_path}...")

        print("📊 Running clustering algorithm to identify user segments...")

        success = await run_pipeline(config.csv_path)

        if success:
            print("✅ Analysis complete! Personas and features generated.")

        # Run virtual board if pipeline succeeded
        virtual_board_success = False
        if success:
            print("\n🎭 Starting virtual user board simulation...")
            logger.info("Starting Virtual Board Simulation...")
            try:
                board = VirtualUserBoard()
                board_result = await board.run_board_simulation_from_files()
                virtual_board_success = board_result.get("success", False)
                if virtual_board_success:
                    print("✅ Virtual board simulation completed successfully!")
                    logger.info("Virtual board simulation completed successfully!")
                else:
                    print("❌ Virtual board simulation failed")
                    logger.error(f"Virtual board failed: {board_result.get('error', 'Unknown error')}")
            except Exception as e:
                print("❌ Virtual board simulation failed")
                logger.error(f"Virtual board execution failed: {e}")

        end_time = datetime.now()

        total_duration = end_time - start_time

        # Display results summary
        print(f"\n🎉 Pipeline completed in {total_duration}")
        logger.info("=" * 70)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 70)

        if success:
            logger.info("SUCCESS")
            logger.info(f"Total Execution Time: {total_duration}")
            logger.info("Architecture: Cluster-Based Pipeline + OpenAI Agents SDK")

            if virtual_board_success:
                logger.info("Virtual Board: COMPLETED")
            else:
                logger.info("Virtual Board: FAILED")

            logger.info("Generated Files:")
            logger.info(f"   - {config.output_dir}/generated_personas.json")
            logger.info(f"   - {config.output_dir}/generated_features.json")
            logger.info(f"   - {config.output_dir}/personas_and_features.md")
            if virtual_board_success:
                logger.info(f"   - {config.output_dir}/virtual_user_board_summary.md")

            logger.info("Architecture Benefits:")
            logger.info("   - Cluster-based personas from real user segments")
            logger.info("   - Professional error handling with retry logic")
            logger.info("   - Comprehensive logging and monitoring")
            logger.info("   - Type-safe data structures")
            logger.info("   - Environment-based configuration")
            logger.info("   - OpenAI Agents SDK for multi-agent virtual board")
            logger.info("   - Complete dataset processing for comprehensive insights")

        else:
            logger.error("FAILED")
            logger.error(f"Execution Time: {total_duration}")

            logger.info("Troubleshooting Tips:")
            logger.info("   1. Check OpenAI API key is valid and has credits")
            logger.info("   2. Ensure CSV file exists and is properly formatted")
            logger.info("   3. Verify all dependencies are installed")
            logger.info("   4. Check network connectivity")
            logger.info("   5. Check logs in logs/main.log and logs/pipeline.log")

    except KeyboardInterrupt:
        logger.warning("Pipeline execution interrupted by user")
        sys.exit(1)

    except ValidationError as e:
        logger.error(f"Configuration or validation error: {e}")
        sys.exit(2)

    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")
        import traceback
        logger.error("Stack trace:", exc_info=True)
        sys.exit(3)

if __name__ == "__main__":
    asyncio.run(main())