from typing_extensions import Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# define ReAct Agent state with a messages field
# this is the same as using langgraph.graph.MessagesState directly
class ReActState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
