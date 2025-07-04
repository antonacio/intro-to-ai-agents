from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator

from langchain_core.language_models import BaseLanguageModel
from langgraph.graph import StateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver


class BaseAgent(ABC):
    """Abstract base agent class"""

    def __init__(
        self,
        llm: BaseLanguageModel,
        recursion_limit: int = 50,
    ):
        """Initialize the base agent class.

        Args:
            llm: The language model to use.
            recursion_limit: Maximum recursion limit for the graph execution.
        """
        self.llm = llm
        self._recursion_limit = recursion_limit

    @abstractmethod
    def build_graph(self) -> StateGraph:
        """Builds the LangGraph graph for the agent.

        This method must be implemented in any subclass to define the specific
        behavior of the LangGraph agent.

        Returns:
            The LangGraph agent's graph object.
        """

    def run(
        self,
        input: dict[str, Any],
        run_config: dict[str, Any] = None,
        stream_mode: str = "values",
        checkpointer: BaseCheckpointSaver = None,
    ) -> dict[str, Any] | Any:
        """Run the agent graph synchronously.

        Args:
            input: The input kwargs to the agent graph execution.
            run_config: Configuration for running the agent graph.
            stream_mode: The mode for streaming the output.
            checkpointer: The checkpointer to use.

        Returns:
            The output state of the agent graph execution.
        """
        # set recursion limit to default if not provided
        run_config = run_config or {}
        if "recursion_limit" not in run_config:
            run_config["recursion_limit"] = self._recursion_limit

        # compile the agent graph with the checkpointer
        compiled_graph = self.build_graph().compile(checkpointer=checkpointer)
        # run the agent graph synchronously
        results = compiled_graph.invoke(
            input=input,
            config=run_config,
            stream_mode=stream_mode,
        )
        return results

    async def arun(
        self,
        input: dict[str, Any],
        run_config: dict[str, Any] = None,
        stream_mode: str = "values",
        checkpointer: BaseCheckpointSaver = None,
    ) -> AsyncGenerator:
        """Run the agent graph asynchronously.

        Args:
            input: The input kwargs to the agent graph execution.
            run_config: Configuration for running the agent graph.
            stream_mode: The mode for streaming the output.
            checkpointer: The checkpointer to use.

        Yields:
            Stream of states from the agent graph execution.
        """
        # set recursion limit to default if not provided
        run_config = run_config or {}
        if "recursion_limit" not in run_config:
            run_config["recursion_limit"] = self._recursion_limit

        # compile the agent graph with the checkpointer
        compiled_graph = self.build_graph().compile(checkpointer=checkpointer)
        # run the agent graph asynchronously
        async for event in compiled_graph.astream(
            input=input,
            config=run_config,
            stream_mode=stream_mode,
        ):
            yield event
