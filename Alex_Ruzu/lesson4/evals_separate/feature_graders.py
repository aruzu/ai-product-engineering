feature_quality_grader = {
    "name": "Feature proposal rubric",
    "type": "label_model",
    "model": "gpt-4o-mini",
    "input": [
        {
            "role": "developer",
            "content": (
                "You are a senior PM. Evaluate the draft below.\n"
                "Pass if ALL are true:\n"
                "1. Addresses the user's request accurately.\n"
                "2. Mentions at least one relevant competitor.\n"
                "3. Contains a clear benefit statement and implementation outline.\n"
                "Answer with: PASS or FAIL."
            ),
        },
        {
            "role": "user",
            "content": (
                "User review: {{item.review}}\n\n"
                "Draft proposal:\n{{sample.choices[0].message.content}}"
            ),
        },
    ],
    "passing_labels": ["PASS"],
    "labels": ["PASS", "FAIL"],
}