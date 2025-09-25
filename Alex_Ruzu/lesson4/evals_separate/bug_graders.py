bug_feature_grader = {
    "name": "Bug vs Feature Match",
    "type": "string_check",
    "input": "{{sample.choices[0].message.content}}",
    "operation": "like",
    "reference": "{{item.label}}",
}

duplicate_grader = {
    "name": "Duplicate Check",
    "type": "python",
    "source": """
def grade(sample, item):
    expected = "duplicate" if item['is_duplicate'] else "new"

    # Get agent response
    if 'output_text' in sample:
        response = sample['output_text']
    else:
        response = sample['choices'][0]['message']['content']

    # Extract first word (should be "duplicate" or "bug")
    got = response.strip().lower().split()[0]

    # Return float score (1.0 for correct, 0.0 for incorrect)
    return 1.0 if got == expected else 0.0
"""
}