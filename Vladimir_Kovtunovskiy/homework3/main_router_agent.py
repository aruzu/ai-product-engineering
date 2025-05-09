import asyncio
from pydantic import BaseModel
from dotenv import load_dotenv 
from agents import Agent, ModelSettings, function_tool, handoff, Runner, HandoffInputData
from openai.types.responses import ResponseTextDeltaEvent
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from bug_handler_agent import bug_handler_agent
from feature_handler_agent import feature_handler_agent

load_dotenv()

class RouterOutput(BaseModel):
    reason: str
    """Your reasoning for why this task is important"""

    task_description: str
    """Detailed description of the task"""

    task_classification: str
    """Bug or feature request task"""

@function_tool
def read_report() -> str:
    report_path = "Vladimir_Kovtunovskiy/homework3/board_session_report.md"

    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
            return report_content
    
    except FileNotFoundError:
        error_msg = f"Error: Report file not found at {report_path}"
        return error_msg
    except Exception as e:
        error_msg = f"An error occurred during reading the report: {str(e)}"
        return error_msg

router_agent = Agent(
    name="Router Agent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
                    You are the router agent. Your goal is to read a report about product research 
                    and decide which task is the most important to do next. This task should be 
                    classified either as a bug or as a feature request.
                    After classification handoff corresponding feature to a specialized agent
                    for handling bugs or feature requests.
                    For feature request - ask Agent to do a research on how to implement the feature into the product.
                    For Bugs - ask Agent to create a bug ticket.""",
    model="gpt-4.1",
    tools=[read_report, bug_handler_agent.as_tool(tool_name="handle_bugs", 
                                                  tool_description="Creates a bug ticket"), 
           feature_handler_agent.as_tool(tool_name="handle_feature_request", 
           tool_description="Creates a research plan and executes it")]
)

async def main():
    """Main function to simulate an agent interaction."""
    print("--- Router Agent Simulation Start ---")
    result = Runner.run_streamed(router_agent, input="Analyze the latest report and decide which task to do next.")
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)
    print("--- Router Agent Simulation End ---")

if __name__ == "__main__":
    asyncio.run(main())
