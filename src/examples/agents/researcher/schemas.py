from typing import Literal
from typing_extensions import Annotated
from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from operator import add


class UserQueryClassification(BaseModel):
    """Schema used to classify the user's query."""

    reasoning: str = Field(
        description="Short sentence explaining the reasoning behind the classification of the user's query."
    )
    classification: Literal[
        "ask_for_more_info", "conduct_research", "respond_to_user"
    ] = Field(description="The classification of the user's query.")


class ResearchPlan(BaseModel):
    """Schema used to generate a research plan."""

    steps: list[str] = Field(description="A list of steps in the research plan.")


class InputState(BaseModel):
    """Input state for the Research Agent graph."""

    messages: Annotated[list[AnyMessage], add_messages] = Field(
        description="The chat history between the user and the agent."
    )


class ResearchState(BaseModel):
    """State of the Research Agent graph"""

    messages: Annotated[list[AnyMessage], add_messages] = Field(
        description="The chat history between the user and the agent."
    )
    query_classification: UserQueryClassification = Field(
        default=None, description="The classification of the user's query."
    )
    research_steps: list[str] = Field(
        default=[], description="A list of steps in the research plan."
    )
    current_step: int = Field(
        default=0, description="Index of the current step in the research plan."
    )
    documents: Annotated[list[Document], add] = Field(
        default=[],
        description="This is a list of documents that the agent can reference to answer the user's query.",
    )
    research_results: str = Field(
        default="",
        description="The research results that will be used to answer the user's query.",
    )
