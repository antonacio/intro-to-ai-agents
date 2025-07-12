from langchain_core.language_models import BaseLanguageModel
from langgraph.graph import START, END, StateGraph

from examples.agents.base_agent import BaseAgent
from examples.agents.dummy.schemas import DummyState


class DummyAgent(BaseAgent):
    """Dummy agent that just makes a call to the LLM"""

    def __init__(
        self,
        llm: BaseLanguageModel,
        **kwargs,
    ):
        """Initialize the dummy agent.

        Args:
            llm: The language model to use.
            **kwargs: Additional keyword arguments for the base agent initialization.
        """
        super().__init__(llm=llm, **kwargs)

    def build_graph(self) -> StateGraph:
        """Build the dummy agent graph."""

        # dummy node that just makes a call to the LLM
        def llm_node(state: DummyState):
            response = self.llm.invoke(state["messages"])
            return {"messages": [response]}

        # define the graph
        dummy_graph = StateGraph(DummyState)

        # add nodes to the graph
        dummy_graph.add_node("llm", llm_node)

        # define graph edges
        dummy_graph.add_edge(START, "llm")
        dummy_graph.add_edge("llm", END)

        return dummy_graph
