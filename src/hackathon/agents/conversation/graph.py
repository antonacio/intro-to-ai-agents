"""Multi-agent graph for legal conversation and drafting workflow."""

from typing import Dict, Any, Literal
from langchain_core.language_models import BaseLanguageModel
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from .multi_agent_state import MultiAgentState
from .legal_conversation_agent import ConversationAgent
from .drafting_node import drafting_node


def create_conversation_node(conversation_agent: ConversationAgent):
    """Create a conversation node that wraps the LegalConversationAgent."""

    def conversation_node(state: MultiAgentState) -> Dict[str, Any]:
        """
        Conversation node that handles client interaction and information gathering.

        Args:
            state: The shared multi-agent state

        Returns:
            dict: Updated state with conversation results
        """
        # Get the current messages from state
        messages = state.get("messages", [])
        if not messages:
            return {
                "conversation_complete": True,
                "current_agent": "conversation",
                "next_agent": "drafting",
            }

        # Get the thread ID
        thread_id = state.get("thread_id", "default")

        # Run the conversation agent's graph directly with current messages
        # This avoids double-processing and message duplication
        input_data = {"messages": messages}
        run_config = (
            {"configurable": {"thread_id": thread_id}}
            if conversation_agent.memory
            else {}
        )

        # Run the conversation agent
        result = conversation_agent.run(
            input=input_data,
            run_config=run_config,
            checkpointer=conversation_agent.memory,
        )

        # Check if conversation should end (look for end_conversation tool calls)
        conversation_complete = False
        result_messages = result.get("messages", [])
        for msg in result_messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if tool_call.get("name") == "end_conversation":
                        conversation_complete = True
                        break

        # Extract conversation summary when conversation is complete
        conversation_summary = {}
        if conversation_complete:
            summary_data = conversation_agent.extract_conversation_summary(thread_id)
            conversation_summary = summary_data

        # Return state updates
        updates = {
            "messages": result_messages,
            "conversation_complete": conversation_complete,
            "current_agent": "conversation",
        }

        if conversation_complete:
            updates["next_agent"] = "drafting"
            updates["conversation_summary"] = conversation_summary

        return updates

    return conversation_node


def should_continue_conversation(
    state: MultiAgentState,
) -> Literal["drafting", "conversation", "__end__"]:
    """
    Determine the next step in the workflow based on current state.

    Args:
        state: The current multi-agent state

    Returns:
        str: Next node to execute or "__end__" to finish
    """
    # If conversation is complete, move to drafting
    if state.get("conversation_complete", False):
        if not state.get("drafting_complete", False):
            return "drafting"
        else:
            return "__end__"

    # Continue conversation if not complete
    return "conversation"


class MultiAgentLegalGraph:
    """Multi-agent graph for legal client onboarding and pitch deck generation."""

    def __init__(self, llm: BaseLanguageModel, use_memory: bool = True):
        """
        Initialize the multi-agent legal workflow graph.

        Args:
            llm: The language model to use for the conversation agent
            use_memory: Whether to use memory for conversation continuity
        """
        self.llm = llm
        self.use_memory = use_memory

        # Create the conversation agent
        self.conversation_agent = ConversationAgent(llm, use_memory=use_memory)

        # Build the graph
        self.graph = self._build_graph()

        # Set up memory if requested
        self.memory = MemorySaver() if use_memory else None

        # Compile the graph
        self.compiled_graph = self.graph.compile(checkpointer=self.memory)

    def _build_graph(self) -> StateGraph:
        """Build the multi-agent workflow graph."""

        # Create the graph
        graph = StateGraph(MultiAgentState)

        # Add nodes
        conversation_node = create_conversation_node(self.conversation_agent)
        graph.add_node("conversation", conversation_node)
        graph.add_node("drafting", drafting_node)

        # Define edges
        graph.add_edge(START, "conversation")
        graph.add_conditional_edges(
            "conversation",
            should_continue_conversation,
            {"conversation": "conversation", "drafting": "drafting", "__end__": END},
        )
        graph.add_edge("drafting", END)

        return graph

    def run(
        self, input_data: Dict[str, Any], thread_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Run the multi-agent workflow.

        Args:
            input_data: Input data including messages
            thread_id: Thread ID for conversation continuity

        Returns:
            dict: Final state after workflow completion
        """
        # Prepare initial state
        initial_state = {
            "messages": input_data.get("messages", []),
            "thread_id": thread_id,
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

        # Run configuration with thread ID for memory
        run_config = {"configurable": {"thread_id": thread_id}} if self.memory else {}

        # Execute the workflow
        result = self.compiled_graph.invoke(initial_state, config=run_config)

        return result

    def stream(self, input_data: Dict[str, Any], thread_id: str = "default"):
        """
        Stream the multi-agent workflow execution.

        Args:
            input_data: Input data including messages
            thread_id: Thread ID for conversation continuity

        Yields:
            dict: Intermediate states during workflow execution
        """
        # Prepare initial state
        initial_state = {
            "messages": input_data.get("messages", []),
            "thread_id": thread_id,
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

        # Run configuration with thread ID for memory
        run_config = {"configurable": {"thread_id": thread_id}} if self.memory else {}

        # Stream the workflow
        for chunk in self.compiled_graph.stream(initial_state, config=run_config):
            yield chunk
