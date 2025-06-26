"""Interactive demo script showing the full multi-agent workflow."""

import os
from typing import Dict, Any
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from hackathon.agents.conversation import MultiAgentLegalGraph, ConversationAgent


def print_header(title: str, char: str = "="):
    """Print a formatted header."""
    print(f"\n{char * 60}")
    print(f"{title:^60}")
    print(f"{char * 60}")


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'─' * 40}")
    print(f"📋 {title}")
    print(f"{'─' * 40}")


def setup_llm():
    """Set up the language model."""
    print_section("Setting up Language Model")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("💡 Please set your OpenAI API key:")
        print("   export OPENAI_API_KEY='your-key-here'")
        return None
    
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        print("✅ Language model initialized successfully")
        return llm
    except Exception as e:
        print(f"❌ Error initializing LLM: {e}")
        return None


def demo_conversation_agent_only(llm):
    """Demo the conversation agent in isolation."""
    print_header("🗣️  CONVERSATION AGENT DEMO", "=")
    
    print("This demonstrates the conversation agent working independently")
    print("before integrating with the multi-agent graph.")
    
    # Create conversation agent
    agent = ConversationAgent(llm, use_memory=True)
    thread_id = f"demo_standalone_{id(agent)}"
    
    print_section("Starting Conversation Agent")
    print("💬 Type your messages below. The agent will gather legal information.")
    print("💡 Try mentioning: legal needs, company info, M&A, contracts, IP, etc.")
    print("⚠️  Type 'next' to move to the multi-agent demo")
    print("⚠️  Type 'quit' to exit")
    
    while True:
        user_input = input("\n👤 Client: ").strip()
        
        if user_input.lower() == 'quit':
            return False
        elif user_input.lower() == 'next':
            return True
            
        if not user_input:
            continue
        
        try:
            print("🤖 Processing...")
            result = agent.run_conversation(user_input, thread_id)
            
            # Extract the last AI message
            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1]
                print(f"\n🤖 Agent: {last_message.content}")
                
                # Check if conversation ended
                if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    for tool_call in last_message.tool_calls:
                        if tool_call.get('name') == 'end_conversation':
                            print("\n🎯 Conversation ended! Moving to multi-agent demo...")
                            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            continue


def demo_multi_agent_graph(llm):
    """Demo the full multi-agent workflow."""
    print_header("🔄 MULTI-AGENT WORKFLOW DEMO", "=")
    
    print("This demonstrates the full conversation → drafting workflow")
    print("where conversation data is passed to the drafting agent.")
    
    # Create multi-agent graph
    workflow = MultiAgentLegalGraph(llm, use_memory=True)
    thread_id = f"demo_multiagent_{id(workflow)}"
    
    print_section("Starting Multi-Agent Workflow")
    print("💬 Type your messages. When conversation ends, you'll see the drafting agent receive the data.")
    print("⚠️  Type 'stream' to see streaming output")
    print("⚠️  Type 'quit' to exit")
    
    conversation_ended = False
    
    while True:
        if conversation_ended:
            user_input = input("\n🔄 Continue conversation or type 'quit': ").strip()
            if user_input.lower() == 'quit':
                break
            conversation_ended = False
        else:
            user_input = input("\n👤 Client: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'stream':
            demo_streaming_workflow(workflow, thread_id)
            continue
            
        if not user_input:
            continue
            
        try:
            print("🤖 Processing through multi-agent workflow...")
            
            # Prepare input
            input_data = {"messages": [HumanMessage(content=user_input)]}
            
            # Run the workflow
            result = workflow.run(input_data, thread_id=thread_id)
            
            # Show results
            print_workflow_result(result)
            
            # Check if we've completed the full workflow
            if result.get('drafting_complete', False):
                print("\n🎉 FULL WORKFLOW COMPLETED!")
                print("✅ Conversation → Drafting handoff successful")
                conversation_ended = True
                
        except Exception as e:
            print(f"❌ Workflow error: {e}")
            continue


def demo_streaming_workflow(workflow: MultiAgentLegalGraph, thread_id: str):
    """Demo streaming execution of the workflow."""
    print_section("Streaming Workflow Demo")
    
    user_input = input("👤 Enter message for streaming demo: ").strip()
    if not user_input:
        return
        
    input_data = {"messages": [HumanMessage(content=user_input)]}
    
    print("🌊 Streaming workflow execution...")
    print("=" * 40)
    
    try:
        for i, chunk in enumerate(workflow.stream(input_data, thread_id=thread_id)):
            print(f"\n📦 Chunk {i+1}:")
            for node_name, state in chunk.items():
                current_agent = state.get('current_agent', 'unknown')
                conversation_complete = state.get('conversation_complete', False)
                drafting_complete = state.get('drafting_complete', False)
                
                print(f"   🔧 Node: {node_name}")
                print(f"   👤 Current Agent: {current_agent}")
                print(f"   💬 Conversation Complete: {conversation_complete}")
                print(f"   📄 Drafting Complete: {drafting_complete}")
                
                if state.get('draft_output'):
                    print(f"   📋 Draft Output: {state['draft_output'][:100]}...")
                    
    except Exception as e:
        print(f"❌ Streaming error: {e}")


def print_workflow_result(result: Dict[str, Any]):
    """Print the workflow result in a formatted way."""
    print("\n" + "🔍 WORKFLOW RESULT " + "="*20)
    
    # Basic state info
    current_agent = result.get('current_agent', 'unknown')
    conversation_complete = result.get('conversation_complete', False)
    drafting_complete = result.get('drafting_complete', False)
    legal_area = result.get('legal_area', 'Not classified')
    
    print(f"Current Agent: {current_agent}")
    print(f"Legal Area: {legal_area}")
    print(f"Conversation Complete: {'✅' if conversation_complete else '❌'}")
    print(f"Drafting Complete: {'✅' if drafting_complete else '❌'}")
    
    # Show latest messages
    messages = result.get('messages', [])
    if messages:
        print(f"\nTotal Messages: {len(messages)}")
        last_message = messages[-1]
        msg_type = getattr(last_message, 'type', 'unknown')
        content = getattr(last_message, 'content', '')
        
        # Truncate long content
        display_content = content[:200] + "..." if len(content) > 200 else content
        print(f"Latest Message [{msg_type}]: {display_content}")
        
        # Show tool calls if any
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            print(f"Tool Calls: {[tc.get('name', 'unknown') for tc in last_message.tool_calls]}")
    
    # Show draft output if available
    if result.get('draft_output'):
        print(f"\n📄 Draft Output:")
        print(f"{result['draft_output']}")


def main():
    """Main interactive demo function."""
    print_header("🏛️  LEGAL WORKFLOW INTERACTIVE DEMO")
    
    print("This demo shows:")
    print("1. 🗣️  Conversation Agent - Gathering client information")
    print("2. 🔄 Multi-Agent Graph - Full conversation → drafting workflow")
    print("3. 📄 Drafting Handoff - How conversation data flows to drafting")
    
    # Setup
    llm = setup_llm()
    if not llm:
        return
    
    print("\n🎯 Demo Options:")
    print("A) Start with conversation agent only")
    print("B) Jump to multi-agent workflow")
    print("C) Exit")
    
    choice = input("\nSelect option (A/B/C): ").strip().upper()
    
    if choice == 'C':
        print("👋 Goodbye!")
        return
    elif choice == 'A':
        # Demo conversation agent first
        continue_to_multiagent = demo_conversation_agent_only(llm)
        if continue_to_multiagent:
            demo_multi_agent_graph(llm)
    elif choice == 'B':
        # Jump to multi-agent demo
        demo_multi_agent_graph(llm)
    else:
        print("❌ Invalid choice. Exiting.")
        return
    
    print_header("🎉 DEMO COMPLETE")
    print("Thanks for trying the Legal Workflow Demo!")
    print("You've seen how conversation data flows from the conversation agent")
    print("to the drafting agent in a seamless multi-agent workflow.")


if __name__ == "__main__":
    main()