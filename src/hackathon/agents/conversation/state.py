from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class MultiAgentState(TypedDict):
    """Shared state for multi-agent conversation and drafting workflow."""

    # Core conversation data
    messages: Annotated[List[AnyMessage], add_messages]

    # Conversation metadata
    thread_id: str
    conversation_complete: bool

    # Client information extracted during conversation
    client_info: Dict[str, Any]

    # Legal classification data
    legal_areas: List[str]
    legal_needs: List[str]

    # Conversation summary for drafting
    conversation_summary: Dict[str, Any]

    # Drafting status and outputs
    drafting_complete: bool
    draft_output: Optional[str]

    # Agent control flow
    current_agent: str
    next_agent: Optional[str]
