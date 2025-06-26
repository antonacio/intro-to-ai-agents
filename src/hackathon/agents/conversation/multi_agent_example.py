"""Example usage of the multi-agent legal workflow."""

from typing import Dict, Any
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from hackathon.agents.conversation import MultiAgentLegalGraph

from dotenv import load_dotenv

load_dotenv()


def run_legal_workflow_example():
    """
    Example demonstrating the multi-agent legal conversation → drafting workflow.

    This shows how:
    1. A client interacts with the conversation agent
    2. The conversation agent gathers information and uses tools
    3. The conversation ends and transitions to drafting
    4. The drafting agent receives the conversation history
    """

    print("🏛️  LEGAL MULTI-AGENT WORKFLOW EXAMPLE")
    print("=" * 50)

    # Initialize the LLM (you'll need to set your API key)
    llm = ChatOpenAI(model="o3-2025-04-16")

    # Create the multi-agent graph
    legal_workflow = MultiAgentLegalGraph(llm=llm, use_memory=True)

    # Example client messages simulating a consultation
    client_messages = [
        "Hi, I'm Sarah Chen, CEO of TechFlow Analytics. We're a 50-person B2B SaaS company that helps retailers analyze customer data. We're looking for legal help with a few urgent issues.",
        "We have three main concerns: First, we're about to close a Series B funding round and need help with the legal documentation. Second, we're launching in the EU next month and need GDPR compliance review. Third, we recently received a patent infringement claim from a competitor.",
        "The funding round is worth $15M and we need to close by end of next month. The GDPR work is critical since we handle sensitive customer data. The patent claim seems frivolous but we need proper defense. We've worked with Cooley LLP before but they're swamped. We need a firm that can handle all three areas efficiently.",
        "Our revenue is about $8M ARR and growing 150% year over year. We're based in Austin but have remote employees across the US. The EU launch will be our first international expansion. Can you help us with these legal challenges?",
    ]

    thread_id = "demo_client_001"

    print(f"🗣️  Starting conversation with thread ID: {thread_id}")
    print("-" * 50)

    # Process each client message
    for i, message in enumerate(client_messages, 1):
        print(f"\n👤 CLIENT MESSAGE {i}:")
        print(f"{message}")
        print("\n" + "⚖️" * 20)

        # Prepare input
        input_data = {"messages": [HumanMessage(content=message)]}

        # Run the workflow
        print(f"🤖 Processing with multi-agent workflow...")
        result = legal_workflow.run(input_data, thread_id=thread_id)

        # Show the result
        print(f"\n✅ Workflow completed for message {i}")
        print(f"Current Agent: {result.get('current_agent', 'N/A')}")
        print(f"Conversation Complete: {result.get('conversation_complete', False)}")
        print(f"Drafting Complete: {result.get('drafting_complete', False)}")

        # If conversation is complete, break (drafting will have run)
        if result.get("conversation_complete", False):
            print(f"\n🎯 CONVERSATION COMPLETED - DRAFTING AGENT ACTIVATED")
            break

        print(f"\n" + "=" * 50)

    print(f"\n📋 FINAL WORKFLOW STATE:")
    print(f"- Messages processed: {len(result.get('messages', []))}")
    print(f"- Conversation complete: {result.get('conversation_complete', False)}")
    print(f"- Drafting complete: {result.get('drafting_complete', False)}")
    print(f"- Final agent: {result.get('current_agent', 'N/A')}")

    if result.get("draft_output"):
        print(f"\n📄 DRAFT OUTPUT:")
        print(f"{result['draft_output']}")

    print(f"\n✨ Multi-agent workflow demonstration complete!")


def run_streaming_example():
    """
    Example demonstrating streaming execution of the multi-agent workflow.
    """

    print("\n🌊 STREAMING WORKFLOW EXAMPLE")
    print("=" * 50)

    # Initialize the LLM
    llm = ChatOpenAI(model="o3-2025-04-16")

    # Create the multi-agent graph
    legal_workflow = MultiAgentLegalGraph(llm=llm, use_memory=True)

    # Simple client message that will trigger end_conversation tool
    input_data = {
        "messages": [
            HumanMessage(
                content="I need help with a contract review. That's all for now, thanks."
            )
        ]
    }

    print("🌊 Streaming workflow execution...")
    print("-" * 30)

    # Stream the workflow
    for i, chunk in enumerate(
        legal_workflow.stream(input_data, thread_id="streaming_demo")
    ):
        print(f"📦 Chunk {i+1}: {list(chunk.keys())}")

        # Show key state changes
        for node_name, state in chunk.items():
            current_agent = state.get("current_agent")
            if current_agent:
                print(f"   {node_name}: Current agent = {current_agent}")

    print("\n✅ Streaming demonstration complete!")


if __name__ == "__main__":
    """
    Run the multi-agent legal workflow examples.

    Make sure you have your OpenAI API key set in your environment:
    export OPENAI_API_KEY="your-api-key-here"
    """

    try:
        # Run the main example
        run_legal_workflow_example()

        # Run the streaming example
        run_streaming_example()

    except Exception as e:
        print(f"❌ Error running example: {e}")
        print("💡 Make sure you have set your OPENAI_API_KEY environment variable")
        print(
            "💡 Also ensure all dependencies are installed: pip install langchain-openai"
        )
