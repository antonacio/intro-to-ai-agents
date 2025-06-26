"""Example usage of the Legal Conversation Agent for client onboarding."""

from langchain_openai import ChatOpenAI

# Uncomment the following line and comment the above if using Anthropic
# from langchain_anthropic import ChatAnthropic

from hackathon.agents.conversation.legal_conversation_agent import (
    ConversationAgent,
)


def main():
    # Initialize the LLM
    # Make sure to set your API key in the environment variable
    # Note: o3 requires Responses API which isn't supported by ChatOpenAI
    # Using gpt-4o instead which is very capable for legal conversations
    llm = ChatOpenAI(model="o3-2025-04-16")
    # For Anthropic:
    # llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0.7)

    # Create the conversation agent
    agent = ConversationAgent(
        llm=llm, use_memory=True  # Enable memory for conversation continuity
    )

    # Example conversation flow
    print("Legal Client Onboarding System")
    print("=" * 50)

    # You can use a unique thread_id for each client conversation
    thread_id = "client_12345"

    # Simulate a conversation
    example_inputs = [
        "Hi, I'm the CEO of TechStartup Inc. We're looking for legal help with some intellectual property issues.",
        "We're a software company with about 50 employees. We've developed a new AI algorithm and want to protect it.",
        "We're also concerned about some open-source libraries we've used in our product.",
        "We need this sorted out in the next 2 months before our Series B funding round.",
    ]

    for user_input in example_inputs:
        print(f"\nClient: {user_input}")

        # Run the conversation turn
        result = agent.run_conversation(user_input, thread_id)

        # Extract the assistant's response
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            print(f"Agent: {last_message.content}")

        print("-" * 50)

    # Extract conversation summary for the drafting agent
    summary = agent.extract_conversation_summary(thread_id)
    print("\nConversation Summary for Drafting Agent:")
    print(summary)


def interactive_demo():
    """Run an interactive conversation with the agent."""
    # Initialize the LLM
    llm = ChatOpenAI(model="o3-2025-04-16")

    # Create the conversation agent
    agent = ConversationAgent(llm=llm, use_memory=True)

    print("Legal Client Onboarding System - Interactive Mode")
    print("=" * 50)
    print("Type 'quit' to exit the conversation")
    print("Type 'summary' to see the extracted information")
    print("-" * 50)

    thread_id = f"interactive_{id(agent)}"

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == "quit":
            break
        elif user_input.lower() == "summary":
            summary = agent.extract_conversation_summary(thread_id)
            print("\nExtracted Information:")
            print(summary)
            continue

        # Run the conversation
        result = agent.run_conversation(user_input, thread_id)

        # Extract and display the response
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            print(f"\nAgent: {last_message.content}")


if __name__ == "__main__":
    # Run the example conversation
    # main()

    # Uncomment the following line to run the interactive demo
    interactive_demo()
