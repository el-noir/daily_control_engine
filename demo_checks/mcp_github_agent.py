import os
import asyncio
from typing import Annotated, TypedDict, Union

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from datetime import datetime
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

# Load environment variables from .env
load_dotenv()

# 1. Define the State
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    discovery_done: bool
    github_owner: str
    github_repo: str

# 2. Get Tools from MCP Server
async def get_tools():
    """Connects to the GitHub MCP server and retrieves available tools."""
    print("Connecting to GitHub MCP Server...")
    github_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    
    if not github_token:
        print("Error: GITHUB_PERSONAL_ACCESS_TOKEN not found in .env file.")
        return [], None

    client = MultiServerMCPClient(
        {
            "github": {
                "command": "npx.cmd",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "transport": "stdio",
                "env": {
                    "GITHUB_PERSONAL_ACCESS_TOKEN": github_token
                }
            }
        }
    )
    try:
        tools = await client.get_tools()
        print(f"Successfully loaded {len(tools)} tools from GitHub MCP.")
        return tools, client
    except Exception as e:
        print(f"Error connecting to GitHub MCP: {e}")
        return [], None

# 3. Setup the Graph
async def create_agent():
    # Initialize LLM
    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

    # Get MCP tools and bind to LLM
    tools, client = await get_tools()
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    # Default settings from .env
    env_owner = os.getenv("GITHUB_OWNER", "el-noir")
    env_repo = os.getenv("GITHUB_REPO", "daily_control_engine")

    # Define the nodes
    def chatbot(state: State):
        """Invoke the LLM with a professional agent persona."""
        now = datetime.now()
        current_time = now.strftime("%A, %B %d, %Y %I:%M %p")
        
        # Pull owner and repo from state or defaults
        owner = state.get("github_owner") or env_owner
        repo = state.get("github_repo") or env_repo
        
        system_instr = [
            f"You are the 'GitHub Control Engine'. Today is {current_time}.",
            f"PRIMARY TARGET: {owner}/{repo}",
            "\n### CRITICAL OPERATING PROCEDURES ###",
            f"1. **Owner**: Always use '{owner}' for the owner parameter.",
            f"2. **Repo**: Always use '{repo}' (Note the underscores!).",
            "3. **Commits**: For `list_commits`, use the `since` parameter for time filters. NEVER use `sha` for dates.",
            "4. **Format**: Trigger tools directly. Do not output text-based tool calls.",
            "5. **Feedback**: If a tool returns 'Not Found', verify you didn't swap underscores for hyphens."
        ]
        
        system_message = SystemMessage(content="\n".join(system_instr))
        # Limit history to stay under token limits
        messages = [system_message] + state["messages"][-5:]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: State):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    def discovery_handler(state: State):
        """Ensures the discovery state is initialized."""
        return {"github_owner": env_owner, "github_repo": env_repo, "discovery_done": True}

    # Build the graph
    workflow = StateGraph(State)
    workflow.add_node("chatbot", chatbot)
    
    if tools:
        workflow.add_node("tools", ToolNode(tools))
        workflow.add_node("discovery", discovery_handler)
        
        workflow.add_edge(START, "discovery")
        workflow.add_edge("discovery", "chatbot")
        workflow.add_conditional_edges("chatbot", should_continue, ["tools", END])
        workflow.add_edge("tools", "chatbot")
    else:
        workflow.add_edge(START, "chatbot")
        workflow.add_edge("chatbot", END)

    return workflow.compile(), client

# 4. Interactive Chat Loop
async def run_chat():
    agent, client = await create_agent()
    print("\n--- GitHub MCP Chatbot Ready ---")
    print("Type 'exit' to quit.")
    
    # Initialize session state
    session_state = {
        "messages": [], 
        "discovery_done": True, 
        "github_owner": os.getenv("GITHUB_OWNER", "el-noir"), 
        "github_repo": os.getenv("GITHUB_REPO", "daily_control_engine")
    }
    
    while True:
        try:
            user_input = input("\nYou: ")
        except EOFError: break
        if user_input.lower() in ["exit", "q", "quit"]: break

        session_state["messages"].append(HumanMessage(content=user_input))
        
        try:
            async for chunk in agent.astream(session_state, stream_mode="updates"):
                for node, values in chunk.items():
                    if node == "chatbot":
                        last_msg = values['messages'][-1]
                        if last_msg.content: print(f"\nAI: {last_msg.content}")
                        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                            for tc in last_msg.tool_calls:
                                print(f"\n[DEBUG Tool Call]: {tc['name']} with args: {tc['args']}")
                            print(f"\n[AI is calling GitHub tools: {', '.join([tc['name'] for tc in last_msg.tool_calls])}]")
                    elif node == "tools":
                        for msg in values['messages']:
                            print(f"\n[DEBUG Tool Result]: {msg.content[:300]}...")
                
                # Update shared state messages with truncation to save tokens
                if "messages" in values:
                    for msg in values["messages"]:
                        # If it's a ToolMessage, truncate its content
                        if hasattr(msg, "content") and isinstance(msg.content, str) and len(msg.content) > 1500:
                            msg.content = msg.content[:1500] + "... [TRUNCATED TO SAVE TOKENS]"
                        session_state["messages"].append(msg)
                    
        except Exception as e:
            print(f"\nError during execution: {e}")

    if client:
        # Proper cleanup to prevent asyncio errors
        await client.close()

if __name__ == "__main__":
    try:
        asyncio.run(run_chat())
    except KeyboardInterrupt:
        pass
