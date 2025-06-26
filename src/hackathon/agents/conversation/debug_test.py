"""Debug script to test multi-agent workflow step by step."""

import os
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

print("🔍 DEBUGGING MULTI-AGENT WORKFLOW")
print("=" * 40)

# Step 1: Test basic imports
print("1. Testing imports...")
try:
    from hackathon.agents.conversation import ConversationAgent, MultiAgentLegalGraph

    print("   ✅ Imports successful")
except Exception as e:
    print(f"   ❌ Import error: {e}")
    exit(1)

# Step 2: Test LLM initialization
print("\n2. Testing LLM initialization...")
try:
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("   ⚠️  OPENAI_API_KEY not set - using mock")
        # Use a mock LLM for testing
        from langchain_core.language_models.fake import FakeListLLM

        llm = FakeListLLM(responses=["Hello, I'm a legal assistant."])
    else:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    print("   ✅ LLM initialized")
except Exception as e:
    print(f"   ❌ LLM error: {e}")
    exit(1)

# Step 3: Test LegalConversationAgent alone
print("\n3. Testing LegalConversationAgent in isolation...")
try:
    conversation_agent = ConversationAgent(llm, use_memory=False)
    print("   ✅ LegalConversationAgent created")

    # Test a simple run
    print("   Testing simple conversation...")
    result = conversation_agent.run_conversation("Hello", "test_thread")
    print(f"   ✅ Conversation result: {type(result)}")
    print(f"   Messages in result: {len(result.get('messages', []))}")

except Exception as e:
    print(f"   ❌ LegalConversationAgent error: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

# Step 4: Test MultiAgentLegalGraph initialization
print("\n4. Testing MultiAgentLegalGraph initialization...")
try:
    legal_workflow = MultiAgentLegalGraph(llm=llm, use_memory=False)
    print("   ✅ MultiAgentLegalGraph created")
    print(f"   Graph nodes: {legal_workflow.graph.nodes}")
except Exception as e:
    print(f"   ❌ MultiAgentLegalGraph error: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

# Step 5: Test simple input processing with recursion limit
print("\n5. Testing simple input processing...")
try:
    input_data = {"messages": [HumanMessage(content="Hello, I need legal help.")]}

    print("   Running workflow with recursion limit...")

    # Add recursion limit to config
    run_config = {"configurable": {"thread_id": "debug_test"}, "recursion_limit": 5}

    # Prepare initial state
    initial_state = {
        "messages": input_data.get("messages", []),
        "thread_id": "debug_test",
        "conversation_complete": False,
        "client_info": {},
        "legal_areas": [],
        "legal_needs": [],
        "conversation_summary": {},
        "drafting_complete": False,
        "draft_output": None,
        "current_agent": "conversation",
        "next_agent": None,
    }

    result = legal_workflow.compiled_graph.invoke(initial_state, config=run_config)
    print(f"   ✅ Workflow completed")
    print(f"   Result keys: {list(result.keys())}")
    print(f"   Current agent: {result.get('current_agent')}")
    print(f"   Conversation complete: {result.get('conversation_complete')}")

except Exception as e:
    print(f"   ❌ Workflow error: {e}")
    print("   This is expected with fake LLM - conversation won't naturally end")

    # Test with a manually completed conversation
    print("\n   Testing with manually completed conversation...")
    try:
        # Simulate a completed conversation state
        completed_state = {
            "messages": [
                HumanMessage(content="Hello, I need legal help."),
            ],
            "thread_id": "debug_test",
            "conversation_complete": True,  # Force completion
            "client_info": {"name": "Test Client"},
            "legal_areas": ["contract"],
            "legal_needs": ["review"],
            "conversation_summary": {},
            "drafting_complete": False,
            "draft_output": None,
            "current_agent": "conversation",
            "next_agent": "drafting",
        }

        # This should go directly to drafting
        result = legal_workflow.compiled_graph.invoke(completed_state)
        print(f"   ✅ Manual completion test passed")
        print(f"   Final agent: {result.get('current_agent')}")
        print(f"   Drafting complete: {result.get('drafting_complete')}")

    except Exception as e2:
        print(f"   ❌ Manual test error: {e2}")
        import traceback

        traceback.print_exc()

print("\n🎯 Debug test complete!")
