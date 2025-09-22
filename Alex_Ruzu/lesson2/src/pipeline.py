"""
Cluster-Driven Pipeline Execution
Orchestrates the complete review-to-feature pipeline using clustering analysis
"""

import os
import json
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from datetime import datetime

from data_types import Persona, FeatureProposal, PipelineConfig, ClusterSource, FeatureExtractionOutput
from utils import (
    LLMClient, PipelineError, LLMError, ValidationError, ClusteringError,
    safe_json_parse, validate_required_fields, safe_file_write,
    NumpyEncoder, setup_logging, ArtifactManager
)
from pydantic import ValidationError as PydanticValidationError

# Load environment variables
load_dotenv()

# Get logger (logging setup is handled in main.py)
logger = logging.getLogger(__name__)


def formulate_features(analysis_data: Dict[str, Any], personas: List[Persona], config: PipelineConfig, llm_client: LLMClient) -> List[FeatureProposal]:
    """Generate features based on cluster pain points and personas"""
    logger.info(f"Starting feature formulation for {config.min_features}-{config.max_features} features")

    try:
        pain_analysis = analysis_data.get("pain_analysis", {})
        pain_points = pain_analysis.get("pain_points", [])

        if not pain_points:
            raise ValidationError("No pain points found in analysis for feature generation")

        logger.debug(f"Found {len(pain_points)} pain points for feature generation")

        # Build persona summaries for the prompt
        personas_summary = []
        for persona in personas:
            personas_summary.append({
                "name": persona.name,
                "profile": persona.profile,
                "pain_points": persona.pain_points,
                "cluster_id": persona.cluster_source.cluster_id
            })

        prompt = f"""Based on the following review analysis and user personas for Viber app, generate {config.min_features}-{config.max_features} product features that address the most critical user pain points.

PAIN POINTS IDENTIFIED:
{json.dumps(pain_points, indent=2, cls=NumpyEncoder)}

USER PERSONAS:
{json.dumps(personas_summary, indent=2)}

REQUIREMENTS:
1. Generate {config.min_features}-{config.max_features} features maximum
2. Each feature must address specific pain points from the analysis
3. Features should be relevant to the generated personas
4. Include evidence (review IDs) supporting each feature need
5. Each feature should have: name, description, problem_addressed, target_personas, value_proposition, review_evidence

Prioritize features that:
- Address the most severe pain points (highest severity scores)
- Affect multiple personas
- Have clear evidence from multiple reviews

Return response as a JSON array of features."""

        system_prompt = "You are a product manager expert at translating user feedback into actionable product features. Create evidence-based features that address user pain points."

        features_text = llm_client.ask(prompt, system_prompt, max_tokens=1500, temperature=0.2)
        logger.debug(f"LLM response received (length: {len(features_text)})")

        # Parse and validate the JSON response using Pydantic
        try:
            features_data = safe_json_parse(features_text, "feature generation")

            if not isinstance(features_data, list):
                raise ValidationError("LLM response is not a JSON array")

            # Validate using Pydantic model
            feature_extraction = FeatureExtractionOutput(features=[
                FeatureProposal(**feature_dict) for feature_dict in features_data[:config.max_features]
            ])

            features = feature_extraction.features

        except PydanticValidationError as e:
            logger.error(f"LLM response validation failed: {e}")
            # Try to salvage what we can from the response
            features = []
            if isinstance(features_data, list):
                for i, feature_dict in enumerate(features_data[:config.max_features]):
                    try:
                        feature = FeatureProposal(**feature_dict)
                        features.append(feature)
                        logger.debug(f"Salvaged feature: {feature.name}")
                    except PydanticValidationError as fe:
                        logger.warning(f"Skipping invalid feature {i+1}: {fe}")
                        continue

            if not features:
                raise LLMError(f"No valid features could be extracted from LLM response. Validation error: {e}")

        except (json.JSONDecodeError, KeyError) as e:
            raise LLMError(f"Failed to parse LLM response as valid JSON: {e}")

        if len(features) < config.min_features:
            raise ValidationError(f"Generated only {len(features)} valid features, minimum required: {config.min_features}")

        logger.info(f"Successfully generated and validated {len(features)} features")
        return features

    except (LLMError, ValidationError) as e:
        logger.error(f"Feature formulation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in feature formulation: {e}")
        raise PipelineError(f"Feature formulation failed: {e}")


def save_results(personas: List[Persona], features: List[FeatureProposal], config: PipelineConfig, artifacts: ArtifactManager) -> None:
    """Save generated personas and features to files using artifact manager"""
    logger.info(f"Saving {len(personas)} personas and {len(features)} features using artifact manager")

    try:
        # Convert to dictionaries for JSON serialization
        personas_data = [persona.to_dict() for persona in personas]
        features_data = [feature.to_dict() for feature in features]

        # Save to timestamped artifacts
        artifacts.save_artifact("personas", personas_data, format="json")
        artifacts.save_artifact("features", features_data, format="json")

        # Also save to main output directory for backward compatibility
        os.makedirs(config.output_dir, exist_ok=True)

        # Save personas to JSON file
        personas_file = os.path.join(config.output_dir, "generated_personas.json")
        personas_json = json.dumps(personas_data, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        safe_file_write(personas_file, personas_json)

        # Save features to JSON file
        features_file = os.path.join(config.output_dir, "generated_features.json")
        features_json = json.dumps(features_data, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        safe_file_write(features_file, features_json)

        # Update personas_and_features.md file
        markdown_file = os.path.join(config.output_dir, "personas_and_features.md")

        markdown_content = []
        markdown_content.append("# Cluster-Based Personas and Features")
        markdown_content.append("")
        markdown_content.append(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        markdown_content.append(f"**Source:** Viber app review clustering analysis (full dataset)")
        markdown_content.append(f"**Method:** Complete cluster-driven pipeline - personas and features from real user segments")
        markdown_content.append("")

        # Add personas
        markdown_content.append("## Generated User Personas")
        markdown_content.append("")
        markdown_content.append("*Generated from clustering analysis of user reviews - representing real user segments*")
        markdown_content.append("")

        for i, persona in enumerate(personas, 1):
            markdown_content.append(f"### {i}. {persona.name}")
            markdown_content.append(f"**Profile:** {persona.profile}")
            markdown_content.append(f"**Background:** {persona.background}")
            markdown_content.append("")

            if persona.pain_points:
                markdown_content.append("**Pain Points:**")
                for pain in persona.pain_points:
                    markdown_content.append(f"- {pain}")
                markdown_content.append("")

            if persona.needs:
                markdown_content.append("**Needs:**")
                for need in persona.needs:
                    markdown_content.append(f"- {need}")
                markdown_content.append("")

            markdown_content.append(f"**Usage Pattern:** {persona.usage_pattern}")

            if persona.review_evidence:
                markdown_content.append(f"**Evidence (Review IDs):** {', '.join(map(str, persona.review_evidence))}")

            # Add cluster source information
            markdown_content.append(f"**Data Source:** Cluster {persona.cluster_source.cluster_id} ({persona.cluster_source.cluster_size} reviews, {persona.cluster_source.average_rating:.1f}/5 rating)")
            markdown_content.append(f"**Urgency Score:** {persona.cluster_source.urgency_score:.3f}")
            markdown_content.append("")

        # Add features
        markdown_content.append("## Generated Product Features")
        markdown_content.append("")

        for i, feature in enumerate(features, 1):
            markdown_content.append(f"### {i}. {feature.name}")
            markdown_content.append(f"**Description:** {feature.description}")
            markdown_content.append(f"**Problem Addressed:** {feature.problem_addressed}")

            if feature.target_personas:
                markdown_content.append(f"**Target Personas:** {', '.join(feature.target_personas)}")

            markdown_content.append(f"**Value Proposition:** {feature.value_proposition}")

            if feature.review_evidence:
                markdown_content.append(f"**Supporting Evidence:** {', '.join(map(str, feature.review_evidence))}")

            markdown_content.append("")

        # Save markdown file to both locations
        markdown_content_str = '\n'.join(markdown_content)
        safe_file_write(markdown_file, markdown_content_str)

        # Also save to artifacts
        artifacts.save_artifact("personas_and_features", markdown_content_str, format="md")

        # Save session info
        session_info = artifacts.get_session_info()
        session_info.update({
            "personas_count": len(personas),
            "features_count": len(features),
            "config_summary": config.get_summary()
        })
        artifacts.save_artifact("session_info", session_info, format="json")

        logger.info(f"Successfully saved to artifacts (session: {artifacts.timestamp}) and legacy locations")
        logger.info(f"Artifacts saved to: {artifacts.session_dir}")

    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise PipelineError(f"Failed to save results: {e}")


async def run_pipeline(csv_path: str = None) -> bool:
    """Execute the complete cluster-driven pipeline with enhanced features"""
    start_time = datetime.now()

    # Load and validate configuration using Pydantic
    try:
        config = PipelineConfig.from_env()
        logger.info("Configuration loaded and validated successfully")
    except (ValueError, PydanticValidationError) as e:
        logger.error(f"Configuration validation failed: {e}")
        raise ValidationError(f"Invalid configuration: {e}")

    # Use config csv_path if not provided
    if csv_path is None:
        csv_path = config.csv_path

    # Initialize artifact manager
    artifacts = ArtifactManager(config.output_dir)
    logger.info(f"Artifact manager initialized - session: {artifacts.timestamp}")

    # Initialize LLM client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValidationError("OPENAI_API_KEY environment variable is required")

    llm_client = LLMClient(api_key, model=config.openai_model, temperature=config.openai_temperature)

    logger.info("=" * 60)
    logger.info("STARTING ENHANCED CLUSTER-DRIVEN PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Session ID: {artifacts.timestamp}")
    logger.info(f"CSV path: {csv_path}")
    logger.info(f"Configuration: {config.get_summary()}")

    # Show previous runs for context
    previous_runs = artifacts.list_previous_runs(limit=3)
    if previous_runs:
        logger.info("Recent previous runs:")
        for run in previous_runs:
            logger.info(f"  - {run['session_id']}: {run['artifacts_count']} artifacts")
    else:
        logger.info("No previous runs found")

    try:
        # Step 1: Run Clustering Analysis (Full Dataset)
        logger.info("STEP 1: Running clustering analysis on full dataset...")
        from clustering import UserClusteringAnalyzer

        extractor = UserClusteringAnalyzer()
        cluster_analysis = extractor.run_analysis(limit=None)

        if not cluster_analysis:
            raise ClusteringError("Clustering analysis failed - no meaningful clusters found")

        # Extract pain points from clustering results
        all_pain_points = []
        for cluster_id, analysis in cluster_analysis.items():
            for feature_req in analysis['feature_requests']:
                all_pain_points.append({
                    'category': feature_req['category'],
                    'severity': feature_req['priority_score'],
                    'count': feature_req['count'],
                    'cluster_id': cluster_id
                })

        logger.info(f"Pain points identified from clustering: {len(set(p['category'] for p in all_pain_points))}")
        for category in set(p['category'] for p in all_pain_points):
            category_points = [p for p in all_pain_points if p['category'] == category]
            max_severity = max(p['severity'] for p in category_points)
            logger.info(f"  - {category}: {max_severity:.3f} max priority")

        # Step 2: Generate Personas from Clustering
        logger.info("STEP 2: Generating personas from clusters...")
        personas_dict_list = extractor.generate_personas_from_clusters(cluster_analysis)

        if not personas_dict_list:
            raise ClusteringError("No personas could be generated from clusters")

        # Convert dictionary personas to Persona objects
        personas = []
        for persona_dict in personas_dict_list:
            try:
                persona = Persona.from_dict(persona_dict)
                personas.append(persona)
            except Exception as e:
                logger.warning(f"Skipping invalid persona: {e}")

        if not personas:
            raise ValidationError("No valid personas could be created from clustering results")

        logger.info(f"Generated {len(personas)} cluster-based personas")
        logger.info(f"Personas: {[p.name for p in personas]}")

        # Validate personas
        if len(personas) > config.max_cluster_personas:
            logger.warning(f"Generated {len(personas)} personas, limiting to {config.max_cluster_personas}")
            personas = personas[:config.max_cluster_personas]

        # Step 3: Formulate Features from Clustering Data
        logger.info("STEP 3: Formulating features from clustering analysis...")

        # Create analysis-compatible format for feature formulation
        clustering_analysis_format = {
            "reviews": [],  # Not needed for AI feature generation
            "pain_analysis": {
                "pain_points": [
                    {
                        "category": category,
                        "description": category.replace("_", " ").title(),
                        "severity": max(p['severity'] for p in all_pain_points if p['category'] == category),
                        "affected_clusters": list(set(p['cluster_id'] for p in all_pain_points if p['category'] == category))
                    }
                    for category in set(p['category'] for p in all_pain_points)
                ]
            }
        }

        features = formulate_features(clustering_analysis_format, personas, config, llm_client)
        logger.info(f"Generated {len(features)} features")

        # Step 4: Save Results
        logger.info("STEP 4: Saving results...")
        save_results(personas, features, config, artifacts)

        end_time = datetime.now()
        duration = end_time - start_time

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total duration: {duration}")

        return True

    except (PipelineError, LLMError, ClusteringError, ValidationError) as e:
        end_time = datetime.now()
        logger.error(f"Pipeline execution failed: {e}")
        return False
    except Exception as e:
        end_time = datetime.now()
        logger.error(f"Unexpected error in pipeline execution: {e}")
        return False


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_pipeline())