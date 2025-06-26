"""API-specific multi-agent graph for stateless request/response handling."""

from typing import Dict, Any, Literal, Optional
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from .state import MultiAgentState
from .conversation_agent import ConversationAgent
from .drafting_node import drafting_node


def create_api_conversation_node(conversation_agent: ConversationAgent):
    """Create a conversation node for API-based interactions."""

    def api_conversation_node(state: MultiAgentState) -> Dict[str, Any]:
        """
        Conversation node that processes messages without waiting for human input.
        
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
        
        # Run the conversation agent with current messages
        input_data = {"messages": messages}
        run_config = (
            {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
            if conversation_agent.memory
            else {"recursion_limit": 25}
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

        # Return state updates
        updates = {
            "messages": result_messages,
            "conversation_complete": conversation_complete,
            "current_agent": "conversation",
        }

        if conversation_complete:
            updates["next_agent"] = "drafting"

        return updates

    return api_conversation_node


def should_api_continue(
    state: MultiAgentState,
) -> Literal["drafting", "__end__"]:
    """
    Determine the next step for API workflow.

    Args:
        state: The current multi-agent state

    Returns:
        str: Next node to execute or "__end__" to finish
    """
    # If conversation is complete, move to drafting
    if state.get("conversation_complete", False):
        if not state.get("drafting_complete", False):
            return "drafting"
    
    # End the graph execution to wait for next API call
    return "__end__"


class APIMultiAgentGraph:
    """API-specific multi-agent graph for legal client onboarding."""

    def __init__(self, llm: BaseLanguageModel, use_memory: bool = True):
        """
        Initialize the API-specific multi-agent workflow graph.

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
        """Build the API-specific workflow graph without human nodes."""

        # Create the graph
        graph = StateGraph(MultiAgentState)

        # Add nodes
        conversation_node = create_api_conversation_node(self.conversation_agent)
        graph.add_node("conversation", conversation_node)
        graph.add_node("drafting", drafting_node)

        # Define edges - no human node needed
        graph.add_edge(START, "conversation")
        graph.add_conditional_edges(
            "conversation",
            should_api_continue,
            {"drafting": "drafting", "__end__": END},
        )
        graph.add_edge("drafting", END)

        return graph

    def process_message(
        self, message: str, thread_id: str = "default", existing_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a single message in the conversation.

        Args:
            message: The user's message
            thread_id: Thread ID for conversation continuity
            existing_state: Existing conversation state if available

        Returns:
            dict: Updated state with AI response
        """
        # Get existing messages from state or memory
        if existing_state:
            messages = existing_state.get("messages", [])
        else:
            # Try to get from memory
            if self.memory:
                config = {"configurable": {"thread_id": thread_id}}
                saved_state = self.memory.get(config)
                if saved_state:
                    messages = saved_state.get("messages", [])
                else:
                    messages = []
            else:
                messages = []

        # Add the new human message
        messages.append(HumanMessage(content=message))

        # Prepare state
        state = {
            "messages": messages,
            "thread_id": thread_id,
            "conversation_complete": existing_state.get("conversation_complete", False) if existing_state else False,
            "legal_area": existing_state.get("legal_area") if existing_state else None,
            "drafting_complete": existing_state.get("drafting_complete", False) if existing_state else False,
            "draft_output": existing_state.get("draft_output") if existing_state else None,
            "current_agent": "conversation",
            "next_agent": None,
        }

        # Run configuration
        run_config = {"configurable": {"thread_id": thread_id}} if self.memory else {}

        # Execute the workflow
        result = self.compiled_graph.invoke(state, config=run_config)

        return result
    
    def start_conversation(self, thread_id: str = "default") -> Dict[str, Any]:
        """
        Start a new conversation with Iris's introduction.
        
        Args:
            thread_id: Thread ID for the conversation
            
        Returns:
            dict: Initial state with Iris's introduction
        """
        # Create initial state with Iris's introduction
        iris_intro = AIMessage(content="Hello! I'm Iris, the AI front-of-house for our law firm. My role is to understand your legal needs and get you in front of the right person as quickly as possible. Could you please tell me about the legal challenges you're facing?")
        
        initial_state = {
            "messages": [iris_intro],
            "thread_id": thread_id,
            "conversation_complete": False,
            "legal_area": None,
            "drafting_complete": False,
            "draft_output": None,
            "current_agent": "conversation",
            "next_agent": None,
        }
        
        # Don't save initial state to memory - let the graph handle it
        # The memory will be populated when the graph runs
        
        return initial_state