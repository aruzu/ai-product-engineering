import asyncio
from pydantic import BaseModel
from dotenv import load_dotenv 
from agents.mcp import MCPServer, MCPServerStdio
from agents import Agent, ModelSettings, function_tool, handoff, Runner, HandoffInputData, trace
from openai.types.responses import ResponseTextDeltaEvent
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
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

async def run(mcp_server: MCPServer, directory_path: str):
    bug_handler_agent = Agent(
    name="Bug Handler Agent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are a specialized agent for handling bug reports.
    Your primary task is to create a github issue using the available tools or mcp.
    The user will provide a description of the bug. First search if github has simillar issue, if not create a new issue as a critical bug.
    """,
    model="gpt-4.1",
    handoff_description="An agent that specializes in creating and managing bug reports.",
    mcp_servers=[mcp_server]
    )

    router_agent = Agent(
        name="Router Agent",
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
                    You are the router agent. Your goal is to read a report about product research 
                    and decide which task is the most important to do next. This task should be 
                    classified either as a bug or as a feature request.
                    After classification handoff corresponding feature to a specialized agent
                    for handling bugs or feature requests.
                    For feature request - ask Agent to do a research on how to implement the feature into the product.
                    For Bugs - ask Agent to create an issue.""",
        model="gpt-4.1",
        tools=[read_report, bug_handler_agent.as_tool(tool_name="handle_bugs", 
                                                  tool_description="Creates a bug ticket"), 
           feature_handler_agent.as_tool(tool_name="handle_feature_request", 
           tool_description="Creates a research plan and executes it")])

    message = "Analyze the latest report and decide which task to do next."
    print("\n" + "-" * 40)
    print(f"Running: {message}")
    result = await Runner.run(starting_agent=router_agent, input=message)
    print(result.final_output)

async def main():
    # Initialize the bug handler agent with MCP
    async with MCPServerStdio(
        cache_tools_list=True,  # Cache the tools list, for demonstration
        params={"command": "uvx", "args": ["mcp-server-git"]},
    ) as server:
        with trace(workflow_name="MCP Git Example"):
            await run(server, "${GITHUB_REPO}")

if __name__ == "__main__":
    asyncio.run(main())
