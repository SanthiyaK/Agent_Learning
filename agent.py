import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

# Load environment variables
load_dotenv()

# Check if API key is present
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("Error: GROQ_API_KEY not found in .env file")
    exit(1)

# Initialize the Groq chat model
chat = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile",
    api_key=api_key
)

# Define a simple tool
@tool
def calculate(expression: str) -> str:
    """Evaluates a mathematical expression."""
    try:
        # Use eval responsibly - strictly for educational demo purposes
        # In production, use a safer math parser
        return str(eval(expression))
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_current_weather(location: str) -> str:
    """Get the weather for a location."""
    if "london" in location.lower():
        return "It is rainy and 15 degrees Celsius." 
    elif "new york" in location.lower():
        return "It is sunny and 20 degrees Celsius."
    else:
        return "I don't have weather data for that location."

# List of tools
tools = [calculate, get_current_weather]

# Bind tools to the model
chat_with_tools = chat.bind_tools(tools)

def simple_agent():
    print("Agent: Hello! I am an AI agent with TOOLS (Calculator & Weather).")
    print("Try asking: 'What is 25 * 4?' or 'What is the weather in London?'")
    print("(Type 'exit' to quit)")

    messages = [
        SystemMessage(content="You are a helpful assistant. You have access to tools. Use them when needed.")
    ]

    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Agent: Goodbye!")
            break

        messages.append(HumanMessage(content=user_input))
        
        try:
            # First call to the model
            response = chat_with_tools.invoke(messages)
            messages.append(response)

            # Check if the model decided to call a tool
            if response.tool_calls:
                print(f"Agent (Thinking): I need to use a tool... {response.tool_calls}")
                
                # Process each tool call
                for tool_call in response.tool_calls:
                    selected_tool = {"calculate": calculate, "get_current_weather": get_current_weather}[tool_call["name"]]
                    tool_output = selected_tool.invoke(tool_call["args"])
                    
                    print(f"Tool Output: {tool_output}")
                    
                    # Add the tool output back to conversation history
                    messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

                # Get final response from model after usage of tool
                final_response = chat_with_tools.invoke(messages)
                print(f"Agent: {final_response.content}")
                messages.append(final_response)
            else:
                # No tool needed, just print response
                print(f"Agent: {response.content}")

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    simple_agent()
