from agents import Agent, ModelSettings, function_tool, handoff
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

@function_tool
def feature_research_plan(context: str) -> str:
    """Generates a plan that will be used to conduct feature research.

    Args:
        context: The detailed description of the feature.
    """
    return "mock research plan"

feature_handler_agent = Agent(
    name="Feature Handler Agent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You're the feature planner Agent that works for Spotify. Your task is to create a plan 
    on how you will research the proposed feature and integrate it into the product.""",
    model="gpt-4.1-mini",
    handoff_description="An agent that specializes in handling features proposals",
    tools=[feature_research_plan]
)