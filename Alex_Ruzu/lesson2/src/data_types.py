"""
Shared Data Types and Configurations

This module contains shared data structures used across the cluster-driven pipeline.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import os

# Add Pydantic imports
from pydantic import BaseModel, Field, validator


@dataclass
class ClusterSource:
    """Represents the cluster source information for personas."""
    cluster_id: int
    cluster_size: int
    average_rating: float
    urgency_score: float
    keywords: List[str]
    sentiment_distribution: Dict[str, int]


class FeatureProposal(BaseModel):
    """Represents a proposed feature with comprehensive metadata and validation."""
    name: str = Field(min_length=1, max_length=100, description="Feature name")
    description: str = Field(min_length=10, max_length=500, description="Feature description")
    problem_addressed: str = Field(min_length=5, max_length=300, description="Problem this feature solves")
    target_personas: List[str] = Field(min_items=1, description="Target user personas")
    value_proposition: str = Field(min_length=10, max_length=300, description="Value proposition")
    review_evidence: List[str] = Field(default_factory=list, description="Supporting review evidence")

    class Config:
        """Pydantic configuration"""
        validate_assignment = True

    @validator('name')
    def name_must_be_meaningful(cls, v):
        """Ensure feature name is meaningful"""
        if not v.strip():
            raise ValueError('Feature name cannot be empty or whitespace')
        # Check for meaningless names
        meaningless_patterns = ['feature', 'improvement', 'enhancement', 'fix']
        if v.strip().lower() in meaningless_patterns:
            raise ValueError(f'Feature name "{v}" is too generic. Be more specific.')
        return v.strip()

    @validator('description')
    def description_must_be_detailed(cls, v):
        """Ensure description is detailed enough"""
        if len(v.strip().split()) < 3:
            raise ValueError('Feature description must be at least 3 words')
        return v.strip()

    @validator('target_personas')
    def personas_must_be_valid(cls, v):
        """Ensure target personas are not empty"""
        if not v:
            raise ValueError('At least one target persona must be specified')
        cleaned_personas = [p.strip() for p in v if p.strip()]
        if not cleaned_personas:
            raise ValueError('Target personas cannot be empty strings')
        return cleaned_personas

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "problem_addressed": self.problem_addressed,
            "target_personas": self.target_personas,
            "value_proposition": self.value_proposition,
            "review_evidence": self.review_evidence
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureProposal':
        """Create from dictionary with validation."""
        return cls(**data)

    def md(self) -> str:
        """Returns markdown representation of the feature."""
        return f"**{self.name}**\n{self.description}"


class FeatureExtractionOutput(BaseModel):
    """Structured output for LLM feature extraction with validation."""
    features: List[FeatureProposal] = Field(min_items=1, description="List of extracted features")

    @validator('features')
    def features_must_have_unique_names(cls, v):
        """Ensure feature names are unique"""
        names = [f.name.lower() for f in v]
        if len(names) != len(set(names)):
            raise ValueError('Feature names must be unique')
        return v


@dataclass
class Persona:
    """Represents a user persona with cluster-based evidence."""
    name: str
    profile: str
    background: str
    pain_points: List[str]
    needs: List[str]
    usage_pattern: str
    review_evidence: List[str]
    cluster_source: ClusterSource

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "profile": self.profile,
            "background": self.background,
            "pain_points": self.pain_points,
            "needs": self.needs,
            "usage_pattern": self.usage_pattern,
            "review_evidence": self.review_evidence,
            "cluster_source": {
                "cluster_id": self.cluster_source.cluster_id,
                "cluster_size": self.cluster_source.cluster_size,
                "average_rating": self.cluster_source.average_rating,
                "urgency_score": self.cluster_source.urgency_score,
                "keywords": self.cluster_source.keywords,
                "sentiment_distribution": self.cluster_source.sentiment_distribution
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Persona':
        """Create from dictionary."""
        cluster_data = data.get("cluster_source", {})
        cluster_source = ClusterSource(
            cluster_id=cluster_data.get("cluster_id", 0),
            cluster_size=cluster_data.get("cluster_size", 0),
            average_rating=cluster_data.get("average_rating", 0.0),
            urgency_score=cluster_data.get("urgency_score", 0.0),
            keywords=cluster_data.get("keywords", []),
            sentiment_distribution=cluster_data.get("sentiment_distribution", {})
        )

        return cls(
            name=data.get("name", ""),
            profile=data.get("profile", ""),
            background=data.get("background", ""),
            pain_points=data.get("pain_points", []),
            needs=data.get("needs", []),
            usage_pattern=data.get("usage_pattern", ""),
            review_evidence=data.get("review_evidence", []),
            cluster_source=cluster_source
        )

    @property
    def system_prompt(self) -> str:
        """Generates system prompt for the LLM agent representing this persona."""
        pain_str = "; ".join(self.pain_points)
        return (
            f"You are {self.name}. Act and respond authentically based on this profile:\n"
            f"- Background: {self.background}\n"
            f"- Profile: {self.profile}\n"
            f"- Usage Pattern: {self.usage_pattern}\n"
            f"- Key Pain Points: {pain_str}\n"
            f"Always speak in the first person ('I', 'me', 'my'). Keep your responses concise (1-3 sentences unless asked otherwise) and focused on the discussion topic. "
            f"Ground your opinions in your background and pain points. Be honest and natural."
        )

    def md(self) -> str:
        """Returns markdown representation of the persona."""
        pain_str = "\n  - ".join(self.pain_points)
        needs_str = "\n  - ".join(self.needs)
        return (
            f"### {self.name}\n\n"
            f"**Profile:** {self.profile}\n\n"
            f"**Background:** {self.background}\n\n"
            f"**Pain Points:**\n  - {pain_str}\n\n"
            f"**Needs:**\n  - {needs_str}\n\n"
            f"**Usage Pattern:** {self.usage_pattern}\n\n"
            f"**Evidence (Review IDs):** {', '.join(map(str, self.review_evidence))}\n\n"
            f"**Data Source:** Cluster {self.cluster_source.cluster_id} "
            f"({self.cluster_source.cluster_size} reviews, "
            f"{self.cluster_source.average_rating:.1f}/5 rating)\n"
            f"**Urgency Score:** {self.cluster_source.urgency_score:.3f}\n"
        )


class PipelineConfig(BaseModel):
    """Enhanced configuration for the cluster-driven pipeline with Pydantic validation."""

    # File paths
    csv_path: str = Field(default="data/viber.csv", description="Path to CSV file with reviews")
    output_dir: str = Field(default="docs", description="Directory for output artifacts")

    # Pipeline parameters
    min_features: int = Field(ge=1, le=10, default=2, description="Minimum features to generate")
    max_features: int = Field(ge=2, le=20, default=3, description="Maximum features to generate")
    max_cluster_personas: int = Field(ge=1, le=15, default=5, description="Maximum personas from clustering")

    # OpenAI settings
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model to use")
    openai_temperature: float = Field(ge=0.0, le=2.0, default=0.7, description="Temperature for OpenAI requests")

    # Virtual board settings
    virtual_board_rounds: int = Field(ge=1, le=10, default=3, description="Number of virtual board rounds")

    class Config:
        """Pydantic configuration"""
        validate_assignment = True
        use_enum_values = True

    @validator('max_features')
    def max_features_must_be_greater_than_min(cls, v, values):
        """Ensure max_features >= min_features"""
        if 'min_features' in values and v < values['min_features']:
            raise ValueError('max_features must be >= min_features')
        return v

    @validator('csv_path')
    def csv_file_must_exist(cls, v):
        """Validate that CSV file exists"""
        csv_path = Path(v)
        if not csv_path.exists():
            raise ValueError(f"CSV file not found: {v}")
        if not csv_path.suffix.lower() == '.csv':
            raise ValueError(f"File must be a CSV file, got: {csv_path.suffix}")
        return str(csv_path)

    @validator('output_dir')
    def output_dir_must_be_valid(cls, v):
        """Validate output directory"""
        output_path = Path(v)
        try:
            # Try to create directory if it doesn't exist
            output_path.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise ValueError(f"Cannot create output directory {v}: {e}")
        return str(output_path)

    @validator('openai_model')
    def openai_model_must_be_valid(cls, v):
        """Validate OpenAI model name"""
        valid_models = [
            "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini",
            "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"
        ]
        if v not in valid_models:
            raise ValueError(f"OpenAI model must be one of: {', '.join(valid_models)}")
        return v

    @classmethod
    def from_env(cls) -> 'PipelineConfig':
        """Create configuration from environment variables with validation."""
        try:
            return cls(
                min_features=int(os.getenv('MIN_FEATURES', '2')),
                max_features=int(os.getenv('MAX_FEATURES', '3')),
                max_cluster_personas=int(os.getenv('MAX_CLUSTER_PERSONAS', '5')),
                openai_model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
                openai_temperature=float(os.getenv('OPENAI_TEMPERATURE', '0.7')),
                csv_path=os.getenv('CSV_PATH', 'data/viber.csv'),
                output_dir=os.getenv('OUTPUT_DIR', 'docs'),
                virtual_board_rounds=int(os.getenv('VIRTUAL_BOARD_ROUNDS', '3'))
            )
        except ValueError as e:
            # Re-raise with more context about environment variables
            raise ValueError(f"Invalid environment variable configuration: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary for logging"""
        return {
            "csv_path": self.csv_path,
            "output_dir": self.output_dir,
            "features_range": f"{self.min_features}-{self.max_features}",
            "max_personas": self.max_cluster_personas,
            "model": self.openai_model,
            "temperature": self.openai_temperature,
            "board_rounds": self.virtual_board_rounds
        }