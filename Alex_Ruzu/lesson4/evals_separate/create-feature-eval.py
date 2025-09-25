from dotenv import load_dotenv

load_dotenv()

import openai, os, json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from schemas import ReviewRow
from feature_graders import feature_quality_grader

openai.api_key = os.environ["OPENAI_API_KEY"]

data_source_config = {
    "type": "custom",
    "item_schema": ReviewRow.model_json_schema(),
    "include_sample_schema": True,
}

eval_create_response = openai.evals.create(
    name="feature-quality-eval",
    metadata={
        "description": "Checks feature proposal quality for the AI Product-Manager agent"
    },
    data_source_config=data_source_config,
    testing_criteria=[feature_quality_grader],
)

EVAL_ID = eval_create_response.id
print("Feature Eval created:", EVAL_ID)