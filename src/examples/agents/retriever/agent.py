from langchain_core.language_models import BaseLanguageModel
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from langchain_chroma import Chroma
from langgraph.utils.runnable import RunnableCallable

from examples.agents.base_agent import BaseAgent
from examples.agents.retriever.prompts import GENERATE_QUERIES_SYSTEM_PROMPT
from examples.agents.retriever.schemas import (
    GeneratedQueries,
    SingleQueryState,
    InputState,
    RetrieverState,
)
from examples.config import (
    vector_store_collection_name,
    embedding_model,
    vector_store_client,
)


class RetrieverAgent(BaseAgent):
    """Retriever agent that generates search queries and retrieves documents from the vector store."""

    def __init__(
        self,
        llm: BaseLanguageModel,
        generate_queries_prompt: str = GENERATE_QUERIES_SYSTEM_PROMPT,
        **kwargs,
    ):
        """Initialize the Retriever agent.

        Args:
            llm: The language model to use.
            generate_queries_prompt: System prompt for the generate_queries node.
            **kwargs: Additional keyword arguments for the base agent initialization.
        """
        super().__init__(llm=llm, **kwargs)
        self._generate_queries_prompt = generate_queries_prompt
        # instantiate the vector store to retrieve documents from
        self._vector_store = Chroma(
            client=vector_store_client,
            collection_name=vector_store_collection_name,
            collection_metadata=vector_store_client.get_or_create_collection(
                vector_store_collection_name
            ).metadata,
            embedding_function=embedding_model,
        )
        # retrieve documents using max marginal relevance
        self._retrieval_kwargs = {
            "search_type": "mmr",
            "search_kwargs": {"k": 2, "fetch_k": 5},
        }

    def build_graph(self) -> StateGraph:
        """Build the Retriever agent graph."""

        def generate_queries(state: InputState) -> RetrieverState:
            """Generate search queries based on the research task (a step in the research plan)."""
            generate_queries_prompt = self._generate_queries_prompt.format(
                research_task=state.research_task
            )
            generated_queries = self.llm.with_structured_output(
                GeneratedQueries
            ).invoke(generate_queries_prompt)

            return {"search_queries": generated_queries.queries}

        async def query_documents_async(state: SingleQueryState) -> RetrieverState:
            """Retrieve documents asynchronously based on a given search query."""
            retriever = self._vector_store.as_retriever(**self._retrieval_kwargs)
            retrieved_chunks = await retriever.ainvoke(state.query)
            return {"retrieved_chunks": retrieved_chunks}

        def query_documents(state: SingleQueryState) -> RetrieverState:
            """Retrieve documents synchronously based on a given search query."""
            retriever = self._vector_store.as_retriever(**self._retrieval_kwargs)
            retrieved_chunks = retriever.invoke(state.query)
            return {"retrieved_chunks": retrieved_chunks}

        def initialize_retrieval(state: RetrieverState) -> list[Send]:
            """Creates a retrieval task for each generated query."""
            return [
                Send("query_documents", SingleQueryState(query=query))
                for query in state.search_queries
            ]

        # define the graph
        retriever_graph = StateGraph(
            input_schema=InputState, state_schema=RetrieverState
        )
        retriever_graph.add_node("generate_queries", generate_queries)
        # use RunnableCallable to handle both sync and async graph executions
        retriever_graph.add_node(
            "query_documents",
            RunnableCallable(func=query_documents, afunc=query_documents_async),
        )

        retriever_graph.add_edge(START, "generate_queries")
        retriever_graph.add_conditional_edges(
            "generate_queries", initialize_retrieval, ["query_documents"]
        )
        retriever_graph.add_edge("query_documents", END)

        return retriever_graph
