import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

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
    model_name="llama-3.3-70b-versatile",  # You can change this to other available models
    api_key=api_key
)

def simple_agent():
    print("Agent: Hello! I am a simple AI agent powered by Groq. How can I help you today?")
    print("(Type 'exit' to quit)")

    messages = [
        SystemMessage(content="You are a helpful and concise AI assistant.")
    ]

    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Agent: Goodbye!")
            break

        messages.append(HumanMessage(content=user_input))
        
        try:
            response = chat.invoke(messages)
            print(f"Agent: {response.content}")
            messages.append(response)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    simple_agent()
