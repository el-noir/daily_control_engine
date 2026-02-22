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
        """Invoke the LLM with a professional agent persona and tool-calling guidance."""
        now = datetime.now()
        current_time = now.strftime("%A, %B %d, %Y %I:%M %p")
        
        system_prompt = (
            f"### IDENTITY & ROLE\n"
            f"You are the 'Daily Control Engine'—a proactive, high-precision personal executive assistant. "
            f"Your primary goal is to manage the user's schedule with 100% accuracy and professional clarity.\n\n"
            
            f"### TEMPORAL CONTEXT\n"
            f"- Current Time: {current_time}\n"
            f"- Current Year: {now.year}\n"
            f"- User Timezone: Asia/Karachi (GMT+5)\n\n"
            
            f"### CORE COMPETENCIES & OPERATING PROCEDURES\n"
            f"1. **Precision Scheduling**: Before creating an event, verify the date exists. "
            f"For example, Feb 29 *only* exists in 2024, 2028, etc. 2026 is NOT a leap year.\n"
            f"2. **Conflict Awareness**: When scheduling a new event, it is best practice to first check for existing events at that time to avoid double-booking.\n"
            f"3. **Smart Queries**: When a user asks 'what am I doing', always query for the full day (00:00:00 to 23:59:59).\n"
            f"4. **Implicit Dates**: If a user says 'Monday', 'next week', or 'tomorrow', calculate those dates relative to {current_time}.\n\n"
            
            f"### TOOL PROTOCOLS (CRITICAL)\n"
            f"- **Execution Only**: Never output code snippets, 'function=...', or JSON blocks in your chat response. "
            f"Use the tool-calling interface provided to perform actions.\n"
            f"- **One Step at a Time**: If a task requires multiple tools (e.g., check conflicts then create), do them sequentially.\n"
            f"- **Fail Gracefully**: If a tool returns an error (e.g., unauthorized or invalid input), explain the issue clearly and suggest a fix.\n\n"
            
            f"### TONE & STYLE\n"
            f"- Tone: Professional, organized, and helpful.\n"
            f"- Formatting: Use bullet points for event details. Highlight important times in **bold**.\n"
            f"- Brevity: Be concise. Don't repeat what the user just said; focus on the result of the action."
        )
        
        system_message = SystemMessage(content=system_prompt)
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
