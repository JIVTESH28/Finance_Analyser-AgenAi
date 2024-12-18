from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define agents
web_agent = Agent(
    name="Web Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    instructions=["Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

# Define the team of agents
agent_team = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    team=[web_agent, finance_agent],
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
)


def process_user_input(agent_team):
    """
    Continuously gets user input, processes it using the agent team, 
    and prints the result until the user says 'bye', 'goodbye', or 'end'.
    
    Args:
        agent_team: The agent team responsible for handling the task.
    """
    print("Chat with the agent! (Type 'bye', 'goodbye', or 'end' to exit)\n")
    
    while True:
        # Get input prompt from the user
        user_input = input("Enter your prompt: ")
        
        # Check for exit conditions
        if user_input.lower() in ['bye', 'goodbye', 'end']:
            print("Goodbye! Have a great day!")
            break
        
        print("\nProcessing your request...\n")
        agent_team.print_response(user_input, stream=True)
        print("\n--- Ready for your next query! ---\n")

if __name__ == "__main__":
    process_user_input(agent_team)
