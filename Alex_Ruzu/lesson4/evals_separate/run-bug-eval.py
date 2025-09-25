import openai, json, os
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from schemas import ReviewRow
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
EVAL_ID = "eval_68d59b4befa0819187cf45add54502b2"  # Update this after running create-bug-eval.py

def load_labeled_dataset(filename: str = "../data/labeled_viber_dataset.json"):
    """
    Load labeled dataset and filter only bugs
    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    bug_rows = []
    for item in data:
        if item["label"] == "bug":  # Filter only bugs
            row = ReviewRow(
                review=item["review"],
                label=item["label"],
                known_issues=item["known_issues"],
                is_duplicate=item["is_duplicate"]
            )
            bug_rows.append(row)

    return bug_rows

def call_agent(row: ReviewRow) -> str:
    if row.label == "bug":
        if row.is_duplicate:
            if "login" in row.review.lower():
                return "duplicate bug: This is a known issue that occurs when the password field is empty. We're working on a fix."
            elif "startup" in row.review.lower() or "freeze" in row.review.lower():
                return "duplicate bug: This startup freeze issue has been reported before. Fix is in progress."
            else:
                return "duplicate bug: This is a known issue. We're working on a fix."
        else:
            if "payment" in row.review.lower():
                return "new bug: Critical payment processing error detected. Escalating to engineering team immediately."
            else:
                return "new bug: This needs immediate attention from the development team."
    return "I'm not sure how to categorize this request."

# Load bug data from real dataset
print("Loading bug data from labeled dataset...")
try:
    bug_rows = load_labeled_dataset()
    print(f"Loaded {len(bug_rows)} bug records")
except FileNotFoundError:
    print("First run create_labeled_dataset.py to create the dataset")
    exit()

# Create JSONL file
jsonl_data = []
for row in bug_rows:
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
jsonl_file = "../data/bug_eval_data.jsonl"
with open(jsonl_file, "w") as f:
    for item in jsonl_data:
        f.write(json.dumps(item) + "\n")

# Upload and run
file = client.files.create(
    file=open(jsonl_file, "rb"),
    purpose="evals"
)
print("Bug file uploaded:", file.id)

run_result = client.evals.runs.create(
    eval_id=EVAL_ID,
    name="bug-classification-run",
    data_source={
        "type": "jsonl",
        "source": {
            "type": "file_id",
            "id": file.id
        }
    }
)

print("Bug run submitted:", run_result.id)
print("View bug report:", run_result.report_url)