from agents import Agent, Runner, function_tool
from pydantic import BaseModel
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

class BugHandlerOutput(BaseModel):
    bug_report_id: str
    """Your reasoning for why this task is important"""

    bug_report_description: str
    """Detailed description of the task"""

@function_tool
def create_bug_report(ctx: str) -> BugHandlerOutput:
    """This is the mock for now"""
    return BugHandlerOutput(bug_report_id="id", bug_report_description="test description.")

bug_handler_agent = Agent(
    name="Bug Handler Agent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    Create a bag report and provide its id back""",
    model="gpt-4.1",
    handoff_description="An agent that specializes in handling bugs.",
    tools=[create_bug_report]
)