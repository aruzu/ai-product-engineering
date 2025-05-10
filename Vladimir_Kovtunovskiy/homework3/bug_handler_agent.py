from agents import Agent, Runner, function_tool
from pydantic import BaseModel
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from agents.mcp import MCPServer, MCPServerStdio
import asyncio
import shutil
import os
from dotenv import load_dotenv 

load_dotenv()

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
    print(f"[Bug Handler Tool] Creating bug ticket for: {bug_description}")
    # In a real scenario, this would interact with a ticketing system or GitHub Issues via MCP
    return BugHandlerOutput(bug_id="id_1_simulated", bug_description=bug_description)

async def setup_mcp_server(repo_path: str): # Added repo_path argument
    """Setup and return the MCP server for GitHub integration"""
    if not os.path.isdir(os.path.join(repo_path, ".git")):
        raise ValueError(f"The provided path '{repo_path}' is not a valid git repository.")
        
    print(f"[Bug Handler] Setting up MCP Server with repo_path: {repo_path}")
    return MCPServerStdio(
        cache_tools_list=True, # If True, tools are fetched once. If False, on every agent run that needs them.
        params={
            "command": "uvx",
            "args": ["mcp-server-git"],
            "github_token": "${GITHUB_TOKEN}",
            "repo": "${GITHUB_REPO}",  # Format: "owner/repository_name"
            "cwd": repo_path  # Crucial for mcp-server-git to know the local repo context
        }
    )

# Create the agent instance that can be imported by other modules
bug_handler_agent = Agent(
    name="Bug Handler Agent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are a specialized agent for handling bug reports.
    Your primary task is to create a bug ticket using the available tools.
    The user will provide a description of the bug.
    If the instructions mention checking GitHub for existing issues,
    you should state that you will proceed to create a new ticket as per your current capability
    (using the 'create_bug_ticket' tool).
    If you need to create a bug ticket, use the 'create_bug_ticket' tool.
    Respond with the BugHandlerOutput containing the bug_id and bug_description.
    """,
    model="gpt-4.1-mini",
    handoff_description="An agent that specializes in creating and managing bug reports.",
    tools=[create_bug_ticket] # Explicitly uses only this tool
)

# This global variable will store the MCP server instance for this module
# It's a list because an agent can have multiple MCP servers.
# We are modifying this global state, which is generally okay for this kind of setup.
mcp_servers_for_bug_handler = [] 

async def initAgentWithMCP(repo_path: str): # Added repo_path argument
    """Initializes the bug_handler_agent with its MCP server."""
    print(f"[Bug Handler] Initializing Agent with MCP for repo: {repo_path}")
    # Setup MCP server
    mcp_server = await setup_mcp_server(repo_path)
    
    # Clear any previous servers and add the new one
    mcp_servers_for_bug_handler.clear()
    mcp_servers_for_bug_handler.append(mcp_server)
    
    # Assign the server list to the agent instance
    bug_handler_agent.mcp_servers = mcp_servers_for_bug_handler
    print(f"[Bug Handler] MCP Server configured for bug_handler_agent: {bug_handler_agent.mcp_servers}")


async def main():
    # This main is for testing bug_handler.py standalone
    if not shutil.which("uvx"):
        raise RuntimeError("uvx is not installed. Please install it with `pip install uvx`.")
    if not os.getenv("GITHUB_TOKEN"):
        raise RuntimeError("GITHUB_TOKEN environment variable is not set")
    if not os.getenv("GITHUB_REPO"):
        raise RuntimeError("GITHUB_REPO environment variable is not set (e.g., 'owner/repo').")

    # Get the current directory path (where the script is located)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to get to the root of the repository
    repo_path = os.path.dirname(os.path.dirname(current_dir))

    # Initialize agent with MCP server
    await initAgentWithMCP(repo_path="https://github.com/VVK93/edu-ai-product-engineer-1")  # Pass the local repository path
    
    # Create a runner for the agent
    runner = Runner(bug_handler_agent)
    
    # Run the agent with the provided directory path (which configures MCP's CWD)
    # The actual input to the agent for its task
    user_bug_description = "The login button is not working on the main page."
    
    # The mcp_servers list is now on bug_handler_agent directly
    if bug_handler_agent.mcp_servers:
        print(f"[Bug Handler Main] Starting MCP server context for: {bug_handler_agent.mcp_servers[0]}")
        async with bug_handler_agent.mcp_servers[0]: # MCP server context
            print(f"[Bug Handler Main] MCP server started. Running agent for: {user_bug_description}")
            result = await runner.run(user_bug_description)
            print(f"[Bug Handler Main] Agent execution result: {result}")
    else:
        print("[Bug Handler Main] MCP server not initialized for bug_handler_agent.")

if __name__ == "__main__":
    asyncio.run(main())