bug_feature_grader = {
    "name": "Bug vs Feature Match",
    "type": "string_check",                             # built-in template
    "input": "{{sample.choices[0].message.content}}",   # agent's output to check
    "operation": "like",                                      # pattern matching (substring)
    "reference": "{{item.label}}",                      # expected label to compare against
}

duplicate_grader = {
    "name": "Duplicate Check",
    "type": "python",
    "source": """
def grade(sample, item):
    # Fix: correct key access for is_duplicate
    expected = "duplicate" if item['is_duplicate'] else "new"

    # Handle both output_text and choices format
    if 'output_text' in sample:
        response = sample['output_text']
    else:
        response = sample['choices'][0]['message']['content']

    # Extract first word from agent response
    got = response.strip().lower().split()[0]

    # Return float score (required by OpenAI Evals)
    return 1.0 if got == expected else 0.0
"""
}

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

# ORIGINAL overall_workflow_grader (for reference):
# overall_workflow_grader = {
#     "name": "Overall Workflow Correctness",
#     "type": "python",
#     "script": """
# def grade(sample, item, **_):
#     # item.expected_trace: list of expected actions (e.g., ['classify:bug', 'deduplicate:new', 'log_github'])
#     # item.expected_side_effects: dict of expected side effects (e.g., {'github_issue': True, 'slack_message': False})
#     # sample.actual_trace: list of actions taken by the agent
#     # sample.actual_side_effects: dict of side effects produced
#
#     trace_correct = item['expected_trace'] == sample['actual_trace']
#     side_effects_correct = item['expected_side_effects'] == sample['actual_side_effects']
#     return {"score": trace_correct and side_effects_correct}
# """
# }

# WORKING overall_workflow_grader (adapted for AI Product Manager):
overall_workflow_grader = {
    "name": "Overall Workflow Correctness",
    "type": "python",
    "source": """
def grade(sample, item):
    # Extract agent response
    if 'output_text' in sample:
        response = sample['output_text'].lower()
    else:
        response = sample['choices'][0]['message']['content'].lower()

    # Expected workflow actions for AI Product Manager
    expected_actions = []

    # 1. Classification step
    if item['label'] == 'bug':
        expected_actions.append('classify:bug')
    else:
        expected_actions.append('classify:feature')

    # 2. Duplicate detection step
    if item.get('is_duplicate', False):
        expected_actions.append('deduplicate:duplicate')
    else:
        expected_actions.append('deduplicate:new')

    # 3. Response generation step
    if item['label'] == 'bug':
        expected_actions.append('respond:bug_handling')
    else:
        expected_actions.append('respond:feature_proposal')

    # Score each workflow step
    workflow_score = 0.0
    total_steps = len(expected_actions)

    for action in expected_actions:
        step_type, expected_value = action.split(':')

        if step_type == 'classify':
            # Check correct classification in response
            if expected_value in response:
                workflow_score += 1.0

        elif step_type == 'deduplicate':
            # Check duplicate detection in response
            if expected_value in response:
                workflow_score += 1.0

        elif step_type == 'respond':
            # Check appropriate response type
            if expected_value == 'bug_handling' and any(word in response for word in ['bug', 'issue', 'problem', 'fix']):
                workflow_score += 1.0
            elif expected_value == 'feature_proposal' and any(word in response for word in ['feature', 'proposal', 'implementation', 'competitive']):
                workflow_score += 1.0

    # Return normalized score (0.0 to 1.0)
    return workflow_score / total_steps if total_steps > 0 else 0.0
"""
}

