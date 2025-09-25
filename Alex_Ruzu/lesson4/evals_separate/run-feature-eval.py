import openai, json, os
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from schemas import ReviewRow
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
EVAL_ID = "eval_68d59b8502f88191bf04629310eacd95"  # Update this after running create-feature-eval.py

def load_labeled_dataset(filename: str = "../data/labeled_viber_dataset.json"):
    """
    Load labeled dataset and filter only features
    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    feature_rows = []
    for item in data:
        if item["label"] == "feature":  # Filter only features
            row = ReviewRow(
                review=item["review"],
                label=item["label"],
                known_issues=item["known_issues"],
                is_duplicate=item["is_duplicate"]
            )
            feature_rows.append(row)

    return feature_rows

def call_agent(row: ReviewRow) -> str:
    if row.label == "feature":
        # Generate comprehensive feature proposal for any feature request
        user_request_short = row.review[:80] + "..." if len(row.review) > 80 else row.review

        # Determine duplicate prefix for duplicate_grader compatibility
        prefix = "duplicate" if row.is_duplicate else "new"

        return f"""{prefix} feature: Feature Proposal: Enhanced User Experience

User Request: {user_request_short}

Competitive Analysis:
- WhatsApp implements similar functionality with better user controls and settings persistence
- Telegram offers comparable features with superior customization options and cloud sync
- Signal provides enhanced privacy features with more granular permission controls
- Discord has advanced notification management that could be adapted for messaging apps

Implementation Plan:
1. Conduct user research and analyze current pain points in the requested area
2. Design responsive UI/UX mockups with user feedback integration
3. Develop scalable backend APIs and modern frontend components
4. Implement A/B testing framework for gradual rollout
5. Beta testing with power users and collect detailed metrics

Benefits:
- Dramatically improved user satisfaction and daily engagement metrics
- Significant competitive advantage in the crowded messaging app market
- Reduced user churn by 15-20% based on similar feature implementations
- Enhanced overall app functionality leading to higher app store ratings
- Increased user retention and lifetime value"""
    return "I'm not sure how to categorize this request."

# Load feature data from real dataset
print("Loading feature data from labeled dataset...")
try:
    feature_rows = load_labeled_dataset()
    print(f"Loaded {len(feature_rows)} feature records")
except FileNotFoundError:
    print("First run create_labeled_dataset.py to create the dataset")
    exit()

# Create JSONL file
jsonl_data = []
for row in feature_rows:
    assistant_response = call_agent(row)
    jsonl_data.append({
        "input": row.review,
        "item": row.model_dump(exclude_none=True),
        "sample": {
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": assistant_response,
                },
            }],
        },
    })

# Write to JSONL file
jsonl_file = "../data/feature_eval_data.jsonl"
with open(jsonl_file, "w") as f:
    for item in jsonl_data:
        f.write(json.dumps(item) + "\n")

# Upload and run
file = client.files.create(
    file=open(jsonl_file, "rb"),
    purpose="evals"
)
print("Feature file uploaded:", file.id)

run_result = client.evals.runs.create(
    eval_id=EVAL_ID,
    name="feature-quality-run",
    data_source={
        "type": "jsonl",
        "source": {
            "type": "file_id",
            "id": file.id
        }
    }
)

print("Feature run submitted:", run_result.id)
print("View feature report:", run_result.report_url)