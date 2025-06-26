"""Drafting node for multi-agent legal workflow."""

from typing import Dict, Any
from .state import MultiAgentState


def drafting_node(state: MultiAgentState) -> Dict[str, Any]:
    """
    TODO: Drafting node that will generate pitch decks based on conversation history.

    Currently shows that conversation history has been successfully passed from
    the conversation agent to the drafting agent.

    Args:
        state: The shared multi-agent state containing conversation data

    Returns:
        dict: Updated state with drafting status and placeholder output
    """
    print("=" * 60)
    print("DRAFTING AGENT RECEIVED CONVERSATION DATA")
    print("=" * 60)

    # Show that we received the conversation data
    print(f"Thread ID: {state.get('thread_id', 'N/A')}")
    print(f"Conversation Complete: {state.get('conversation_complete', False)}")
    print(f"Current Agent: {state.get('current_agent', 'N/A')}")

    # Display conversation history
    messages = state.get("messages", [])
    print(f"\nReceived {len(messages)} messages in conversation history:")

    for i, msg in enumerate(messages):
        msg_type = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", "")
        # Truncate long content for display
        display_content = content[:100] + "..." if len(content) > 100 else content
        print(f"  {i+1}. [{msg_type}] {display_content}")

    # Display extracted client info if available
    client_info = state.get("client_info", {})
    if client_info:
        print(f"\nExtracted Client Info:")
        for key, value in client_info.items():
            print(f"  {key}: {value}")

    # Display legal areas if available
    legal_areas = state.get("legal_areas", [])
    if legal_areas:
        print(f"\nIdentified Legal Areas: {', '.join(legal_areas)}")

    # Display conversation summary if available
    conversation_summary = state.get("conversation_summary", {})
    if conversation_summary:
        print(f"\nConversation Summary Keys: {list(conversation_summary.keys())}")

    print("\n" + "=" * 60)
    print("TODO: Implement actual pitch deck generation logic here")
    print("TODO: - Analyze conversation data")
    print("TODO: - Generate tailored pitch deck")
    print("TODO: - Create presentation materials")
    print("TODO: - Format output for client")
    print("=" * 60)

    # Return updated state
    return {
        "drafting_complete": True,
        "draft_output": "TODO: Placeholder - pitch deck would be generated here based on conversation data",
        "current_agent": "drafting",
        "next_agent": None,
    }
