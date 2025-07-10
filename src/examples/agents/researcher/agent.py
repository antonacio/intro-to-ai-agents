from typing import Literal
from langchain_core.language_models import BaseLanguageModel
from langgraph.graph import END, START, StateGraph
from langgraph.utils.runnable import RunnableCallable

from examples.agents.base_agent import BaseAgent
from examples.agents.researcher.prompts import (
    CLASSIFY_QUERY_SYSTEM_PROMPT,
    CREATE_RESEARCH_PLAN_SYSTEM_PROMPT,
    ASK_FOR_MORE_INFO_SYSTEM_PROMPT,
    RESPOND_TO_USER_SYSTEM_PROMPT,
    RESPOND_WITH_RESEARCH_SYSTEM_PROMPT,
)
from examples.agents.researcher.schemas import (
    InputState,
    ResearchState,
    ResearchPlan,
    UserQueryClassification,
)
from examples.agents.retriever.agent import RetrieverAgent


class ResearchAgent(BaseAgent):
    """Research Agent that conducts research on documents to answer the user's query."""

    def __init__(
        self,
        llm: BaseLanguageModel,
        classify_query_prompt: str = CLASSIFY_QUERY_SYSTEM_PROMPT,
        ask_for_more_info_prompt: str = ASK_FOR_MORE_INFO_SYSTEM_PROMPT,
        respond_to_user_prompt: str = RESPOND_TO_USER_SYSTEM_PROMPT,
        create_research_plan_prompt: str = CREATE_RESEARCH_PLAN_SYSTEM_PROMPT,
        respond_with_research_prompt: str = RESPOND_WITH_RESEARCH_SYSTEM_PROMPT,
        **kwargs,
    ):
        super().__init__(llm=llm, **kwargs)
        self._classify_query_prompt = classify_query_prompt
        self._ask_for_more_info_prompt = ask_for_more_info_prompt
        self._respond_to_user_prompt = respond_to_user_prompt
        self._create_research_plan_prompt = create_research_plan_prompt
        self._respond_with_research_prompt = respond_with_research_prompt
        self.retriever_agent_graph = RetrieverAgent(llm=llm).build_graph()

    def build_graph(self) -> StateGraph:
        """Build the Research Agent graph."""

        def classify_user_query(state: InputState) -> ResearchState:
            """Classify the user's query to determine the next steps."""
            # format the chat history
            chat_history = "\n".join(
                [f"{msg.type.upper()}: {msg.content}" for msg in state.messages]
            )
            # format the prompt
            query_classifier_prompt = self._classify_query_prompt.format(
                chat_history=chat_history
            )
            # invoke the model
            query_classification = self.llm.with_structured_output(
                UserQueryClassification
            ).invoke(query_classifier_prompt)

            return {
                "messages": state.messages,
                "query_classification": query_classification,
            }

        def route_query(
            state: ResearchState,
        ) -> Literal["ask_for_more_info", "create_research_plan", "respond_to_user"]:
            """Route the user's query to the appropriate next step."""
            classification = state.query_classification.classification.strip().lower()
            if classification == "ask_for_more_info":
                return "ask_for_more_info"
            elif classification == "respond_to_user":
                return "respond_to_user"
            elif classification == "conduct_research":
                return "create_research_plan"
            else:
                raise ValueError(f"Invalid query classification: {classification}")

        def ask_for_more_info(state: ResearchState) -> ResearchState:
            """Ask the user for more information."""
            # format the chat history
            chat_history = "\n".join(
                [f"{msg.type.upper()}: {msg.content}" for msg in state.messages]
            )
            # format the prompt
            more_info_prompt = self._ask_for_more_info_prompt.format(
                reasoning=state.query_classification.reasoning,
                chat_history=chat_history,
            )
            # invoke the model
            more_info_response = self.llm.invoke(more_info_prompt)

            return {"messages": [more_info_response]}

        def respond_to_user(state: ResearchState) -> ResearchState:
            """Respond to the user without conducting any research."""
            # format the chat history
            chat_history = "\n".join(
                [f"{msg.type.upper()}: {msg.content}" for msg in state.messages]
            )
            # format the prompt
            respond_to_user_prompt = self._respond_to_user_prompt.format(
                chat_history=chat_history,
            )
            # invoke the model
            response_to_user = self.llm.invoke(respond_to_user_prompt)

            return {"messages": [response_to_user]}

        def create_research_plan(state: ResearchState) -> ResearchState:
            """Create a step-by-step research plan to answer the user's query."""
            # format the chat history
            chat_history = "\n".join(
                [f"{msg.type.upper()}: {msg.content}" for msg in state.messages]
            )
            # format the prompt
            research_plan_prompt = self._create_research_plan_prompt.format(
                chat_history=chat_history,
            )
            # invoke the model
            research_plan = self.llm.with_structured_output(ResearchPlan).invoke(
                research_plan_prompt
            )

            return {
                "research_steps": research_plan.steps,
                "current_step": 0,
            }

        # compile the Retriever Agent graph outside the conduct_research node
        # so that it is rendered as a sub graph in the Research Agent graph
        retriever_agent_compiled_graph = self.retriever_agent_graph.compile()

        async def conduct_research_async(state: ResearchState) -> ResearchState:
            """Conduct research on the documents asynchronously."""
            # format the chat history
            current_research_task = state.research_steps[state.current_step]
            # invoke the retriever agent asynchronously
            retriever_agent_result = await retriever_agent_compiled_graph.ainvoke(
                {"research_task": current_research_task}
            )
            # update the state
            return {
                "documents": retriever_agent_result["retrieved_chunks"],
                "current_step": state.current_step + 1,
            }

        def conduct_research(state: ResearchState) -> ResearchState:
            """Conduct research on the documents synchronously."""
            # format the chat history
            current_research_task = state.research_steps[state.current_step]
            # invoke the retriever agent synchronously
            retriever_agent_result = retriever_agent_compiled_graph.invoke(
                {"research_task": current_research_task}
            )
            # update the state
            return {
                "documents": retriever_agent_result["retrieved_chunks"],
                "current_step": state.current_step + 1,
            }

        def check_research_plan_completion(
            state: ResearchState,
        ) -> Literal["conduct_research", "collect_documents"]:
            """Check if the research plan is complete."""
            if state.current_step < len(state.research_steps):
                return "conduct_research"
            else:
                # research plan is complete, collect the retrieved documents
                return "collect_documents"

        def collect_documents(state: ResearchState) -> ResearchState:
            """Collect the documents retrieved by the Retriever Agent."""
            formatted_documents = []
            collected_document_ids = set()
            for doc in state.documents:
                if doc.id not in collected_document_ids:
                    collected_document_ids.add(doc.id)
                    # format the metadata
                    metadata = doc.metadata or {}
                    formatted_metadata = "".join(
                        f" {k}={v!r}" for k, v in metadata.items()
                    )
                    # format the document content with the metadata
                    formatted_documents.append(
                        f"<document{formatted_metadata}>\n"
                        f"{doc.page_content}\n"
                        f"</document>"
                    )

            return {"research_results": "\n\n".join(formatted_documents)}

        def respond_with_research_results(state: ResearchState) -> ResearchState:
            """Respond with the research results."""
            # format the chat history
            chat_history = "\n".join(
                [f"{msg.type.upper()}: {msg.content}" for msg in state.messages]
            )
            # format the prompt
            respond_with_research_prompt = self._respond_with_research_prompt.format(
                chat_history=chat_history,
                research_results=state.research_results,
            )
            # invoke the model
            response_with_research = self.llm.invoke(respond_with_research_prompt)

            return {"messages": [response_with_research]}

        research_graph = StateGraph(state_schema=ResearchState, input_schema=InputState)
        research_graph.add_node("classify_user_query", classify_user_query)
        research_graph.add_node("ask_for_more_info", ask_for_more_info)
        research_graph.add_node("respond_to_user", respond_to_user)
        research_graph.add_node("create_research_plan", create_research_plan)
        # use RunnableCallable to handle both sync and async graph executions
        research_graph.add_node(
            "conduct_research",
            RunnableCallable(func=conduct_research, afunc=conduct_research_async),
        )
        research_graph.add_node("collect_documents", collect_documents)
        research_graph.add_node(
            "respond_with_research_results", respond_with_research_results
        )

        research_graph.add_edge(START, "classify_user_query")
        research_graph.add_conditional_edges("classify_user_query", route_query)
        research_graph.add_edge("create_research_plan", "conduct_research")
        research_graph.add_conditional_edges(
            "conduct_research", check_research_plan_completion
        )
        research_graph.add_edge("collect_documents", "respond_with_research_results")
        research_graph.add_edge("respond_with_research_results", END)
        research_graph.add_edge("ask_for_more_info", END)
        research_graph.add_edge("respond_to_user", END)

        return research_graph
