from agents import Agent

def create_comparison_judge_agent() -> Agent:
    return Agent(
        name="Comparison Judge",

        instructions = (
            "You are an expert review analysis agent who compares different types of summaries. "
            "You were designed to help analyze customer feedback by generating different summary types "
            "and comparing their effectiveness."
        )
    )