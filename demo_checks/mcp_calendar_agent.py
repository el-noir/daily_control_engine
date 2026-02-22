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
    # add_messages is a reducer that appends new messages to the list
    messages: Annotated[list[BaseMessage], add_messages]

# 2. Get Tools from MCP Server
async def get_tools():
    """
    Connects to the Google Calendar MCP server and retrieves available tools.
    """
    print("Connecting to MCP Server...")
    # Use path relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    creds_path = os.path.join(script_dir, "credentials.json")
    
    client = MultiServerMCPClient(
        {
            "google-calendar": {
                "command": "npx.cmd",
                # The package from the user-provided repo is @cocal/google-calendar-mcp
                "args": ["-y", "@cocal/google-calendar-mcp"],
                "transport": "stdio",
                "env": {
                    "GOOGLE_OAUTH_CREDENTIALS": creds_path
                }
            }
        }
    )
    try:
        tools = await client.get_tools()
        print(f"Successfully loaded {len(tools)} tools from MCP.")
        return tools
    except Exception as e:
        import traceback
        print(f"Error connecting to MCP: {e}")
        traceback.print_exc()
        return []

# 3. Setup the Graph
async def create_agent():
    # Initialize LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

    # Get MCP tools and bind to LLM
    tools = await get_tools()
    if not tools:
        print("Warning: No tools found. The agent will operate without calendar access.")
        llm_with_tools = llm
    else:
        llm_with_tools = llm.bind_tools(tools)

    # Define the nodes
    def chatbot(state: State):
        """Invoke the LLM with the current message history."""
        now = datetime.now()
        current_time = now.strftime("%A, %B %d, %Y %I:%M %p")
        
        system_message = SystemMessage(content=(
            f"You are a highly accurate Calendar Assistant. Today is {current_time}. "
            f"The current year is {now.year}.\n\n"
            "STRICT RULES:\n"
            "1. NEVER output 'function=...' or XML tags. If you need to use a tool, use the tool-calling feature.\n"
            "2. DATE VALIDATION: February 29th ONLY exists in leap years (2024, 2028). 2026 IS NOT a leap year. "
            "If the user asks for Feb 29, 2026, respond: 'I cannot schedule that because February 29th doesn't exist in 2026. Did you mean February 28th or March 1st?'\n"
            "3. FOR LISTING EVENTS: If a user asks 'what am I doing', use 'list-events' with timeMin at the start of the day and timeMax at the end.\n"
            "4. TIMEZONES: Use the user's local timezone (Asia/Karachi) for all events unless specified otherwise."
        ))
        
        messages = [system_message] + state["messages"]
        return {"messages": [llm_with_tools.invoke(messages)]}

    def should_continue(state: State):
        """Determine whether to call a tool or end the conversation."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

    # Build the graph
    workflow = StateGraph(State)
    
    workflow.add_node("chatbot", chatbot)
    
    # If tools exist, add them to the graph
    if tools:
        workflow.add_node("tools", ToolNode(tools, handle_tool_errors=True))
        workflow.add_edge(START, "chatbot")
        workflow.add_conditional_edges("chatbot", should_continue, ["tools", END])
        workflow.add_edge("tools", "chatbot")
    else:
        workflow.add_edge(START, "chatbot")
        workflow.add_edge("chatbot", END)

    return workflow.compile()

# 4. Interactive Chat Loop
async def run_chat():
    agent = await create_agent()
    print("\n--- MCP Calendar Chatbot Ready ---")
    print("Type 'exit' to quit.")
    
    while True:
        try:
            user_input = input("\nYou: ")
        except EOFError:
            break
            
        if user_input.lower() in ["exit", "q", "quit"]:
            break

        # Run the agent
        async for chunk in agent.astream(
            {"messages": [HumanMessage(content=user_input)]},
            stream_mode="updates"
        ):
            for node, values in chunk.items():
                if node == "chatbot":
                    last_msg = values['messages'][-1]
                    if last_msg.content:
                        print(f"\nAI: {last_msg.content}")
                    elif last_msg.tool_calls:
                        print(f"\n[AI is calling tools: {', '.join([tc['name'] for tc in last_msg.tool_calls])}]")

if __name__ == "__main__":
    try:
        asyncio.run(run_chat())
    except KeyboardInterrupt:
        pass
