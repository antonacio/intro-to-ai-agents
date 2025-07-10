from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from examples.agents.base_agent import BaseAgent
from examples.agents.react.prompts import REACT_SYSTEM_PROMPT
from examples.agents.react.schemas import ReActState


class ReActAgent(BaseAgent):
    """ReAct agent class for handling ReAct-based tasks."""

    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: list[type[BaseTool]],
        system_prompt: str = REACT_SYSTEM_PROMPT,
        bind_tools_kwargs: dict[str, Any] = None,
        **kwargs,
    ):
        """Initialize the ReAct agent.

        Args:
            llm: The language model to use.
            tools: List of tools to be used by the agent.
            system_prompt: System prompt for the agent.
            bind_tools_kwargs: Additional arguments for binding tools.
            **kwargs: Additional keyword arguments for the base agent initialization.
        """
        super().__init__(llm=llm, **kwargs)
        self._tools = tools
        self._system_prompt = system_prompt
        # Bind tools to the LLM
        bind_tools_kwargs = bind_tools_kwargs or {}
        self.llm_with_tools = self.llm.bind_tools(self._tools, **bind_tools_kwargs)

    def build_graph(self) -> StateGraph:
        """Build the ReAct agent graph."""

        # System message
        sys_msg = SystemMessage(content=self._system_prompt)

        # Nodes
        def llm_node(state: ReActState) -> ReActState:
            return {
                "messages": [self.llm_with_tools.invoke([sys_msg] + state["messages"])]
            }

        tool_node = ToolNode(self._tools)

        # Graph
        react_graph = StateGraph(ReActState)

        # Define nodes
        react_graph.add_node("llm", llm_node)
        react_graph.add_node("tools", tool_node)

        # Define edges
        react_graph.add_edge(START, "llm")
        react_graph.add_conditional_edges(
            "llm", tools_condition
        )  # routes to "tools" or "__end__"
        react_graph.add_edge("tools", "llm")

        return react_graph
