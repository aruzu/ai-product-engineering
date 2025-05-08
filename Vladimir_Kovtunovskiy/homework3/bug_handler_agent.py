from agents import Agent, Runner, function_tool

@function_tool
def create_bug_report(ctx: str) -> str:
    """This is the mock for now"""
    return ""

bug_handler_agent = Agent(
    name="Bug Handler Agent",
    instructions="""{RECOMMENDED_PROMPT_PREFIX}
    Create a plan on how you will handle the bug.""",
    model="gpt-4.1-mini",
    handoff_description="An agent that specializes in handling bugs.",
    tools=[create_bug_report]
)