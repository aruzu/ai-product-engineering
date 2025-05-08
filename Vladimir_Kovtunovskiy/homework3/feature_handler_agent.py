from agents import Agent, ModelSettings, function_tool, handoff
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

@function_tool
def do_feature_research(ctx: str) -> str:
    """This is the mock for now"""
    return ""

feature_handler_agent = Agent(
    name="Feature Handler Agent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You're the feature planner Agent that works for Spotify. Your task is to create a plan 
    on how you will implement 
    the proposed feature into the product.""",
    model="gpt-4.1-mini",
    handoff_description="An agent that specializes in handling features proposals",
    tools=[do_feature_research]
)