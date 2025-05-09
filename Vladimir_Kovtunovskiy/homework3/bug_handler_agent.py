from agents import Agent, Runner, function_tool
from pydantic import BaseModel
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

class BugHandlerOutput(BaseModel):
    bug_id: str
    """Bug id in the ticket system"""

    bug_description: str
    """Detailed description of the bug"""

@function_tool
def create_bug_ticket(bug_description: str) -> BugHandlerOutput:
    """Creates a bug ticket and returns BugHandler Output Object with bug bug_id and bug_description

    Args:
        bug_description: The detailed description of the bug
    """
    return BugHandlerOutput(bug_id="id_1", bug_description="test description of the bug ticket")

bug_handler_agent = Agent(
    name="Bug Handler Agent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    Create a bug ticket and provide its id back.""",
    model="gpt-4.1-mini",
    handoff_description="An agent that specializes in creating bug reports using tools",
    tools=[create_bug_ticket]
)