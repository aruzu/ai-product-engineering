# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🤖 CLUSTER-DRIVEN PRODUCT DEVELOPMENT PIPELINE

**IMPORTANT**: This repository contains a data-driven pipeline that uses clustering analysis to extract real user personas and generate features from actual user feedback.

### Architecture Overview
- **Cluster-Based Personas** - Personas generated from actual user segments, not AI hallucination
- **Full Dataset Processing** - Analyzes all available reviews for comprehensive insights
- **Multi-Agent Virtual Board** - Enhanced validation using persona agents from real clusters
- **Environment Configuration** - Configurable via .env file for different scenarios

## Conda Environment Management

**CRITICAL**: All Python commands must be executed within the conda environment. Common issues occur when attempting to run Python without proper environment activation.

### Correct Environment Activation
```bash
# Always use this pattern for conda activation in bash commands
conda activate edu_ai_prod_l2 && cd lesson2

# Then run the cluster-driven pipeline
python src/main.py
```

### Common Mistakes to Avoid
- ❌ Running `python` directly without conda activation
- ❌ Using `conda activate` without sourcing conda first
- ❌ Installing packages with pip outside the conda environment
- ❌ Assuming conda is available in all bash contexts

### Verification Commands
```bash
# Check if environment is active
echo $CONDA_DEFAULT_ENV  # Should show: edu_ai_prod_l2

# Verify Python path
which python  # Should point to conda environment path
```

### Package Installation
```bash
# Always install within the activated environment
conda activate edu_ai_prod_l2 && pip install package_name
```

## Essential Resources

- **OpenAI Agents Guide**: https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf
- **OpenAI Agents Python SDK**: https://openai.github.io/openai-agents-python/ (Use Context7 MCP for latest docs)

## Running the Pipeline

The main entry point is the cluster-driven pipeline that processes Viber reviews:
```bash
# Run complete pipeline
python src/main.py
```

The pipeline automatically:
1. Runs clustering analysis on the complete dataset
2. Generates personas from real user clusters
3. Creates features based on cluster pain points
4. Validates through virtual board simulation

### Log Files
- **logs/main.log** - Application flow and business-level progress
- **logs/pipeline.log** - Technical implementation details and processing steps

## Core Architecture

This is an **automated, data-driven pipeline** with strict requirements:

### Data Flow
```
viber.csv → Clustering Analysis → Cluster-Based Personas → AI Feature Generation → Virtual Board Validation
```

### Key Constraints
- **Cluster-based personas** - personas extracted from real user segments via k-means clustering
- **Full dataset processing** - uses all available reviews, not just a subset
- **Traceability required** - every persona references specific cluster data and review evidence
- **Data grounding** - features address pain points identified through clustering analysis
- Input: Complete `data/viber.csv` dataset (format: reviewId, content, score)
- Output: Cluster-based personas and features saved to `docs/` directory

### Component Responsibilities
- `main.py`: Single entry point that orchestrates the complete pipeline
- `pipeline.py`: Core pipeline implementation with clustering and feature generation
- `clustering.py`: K-means clustering and persona extraction from user segments
- `virtual_board.py`: Multi-agent simulation using cluster-derived personas

### Configuration
The pipeline behavior can be customized via `.env` file:
```env
# Pipeline Configuration
MIN_FEATURES=2              # Minimum features to generate
MAX_FEATURES=3              # Maximum features to generate
MAX_CLUSTER_PERSONAS=5      # Maximum personas from clustering
```

## Quality Gates

When implementing or reviewing code:
1. Verify all personas trace back to specific cluster data and review evidence
2. Ensure features address actual pain points identified through clustering analysis
3. Check that virtual board agents use cluster-derived personas consistently
4. Validate that personas represent real user segments, not AI-generated stereotypes
5. Confirm the pipeline processes the complete dataset, not just samples