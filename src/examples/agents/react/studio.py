"""LangGraph Studio script for the ReAct Agent."""

import os
from examples.agents.react.agent import ReActAgent
from examples.tools.math import add, multiply, divide
from examples.tools.rag import retrieve
from examples.tools.web_search import web_search
from examples.config import llm, ModelProviders

if os.getenv("MODEL_PROVIDER") == ModelProviders.OPENAI:
    # we set parallel tool calling to false as math generally is done sequentially
    bind_math_tools_kwargs = {"parallel_tool_calls": False}
else:
    # Ollama model in LangChain does not support the parallel tool calling parameter
    bind_math_tools_kwargs = {}

# Math agent
math_react_agent = ReActAgent(
    llm=llm,
    tools=[add, multiply, divide],
    system_prompt="You are a helpful assistant tasked with performing arithmetic on a set of inputs.",
    bind_tools_kwargs=bind_math_tools_kwargs,
)
math_react_agent_compiled_graph = math_react_agent.build_graph().compile()

# RAG agent
rag_react_agent = ReActAgent(
    llm=llm,
    tools=[retrieve],
    system_prompt="You are a helpful assistant for question-answering tasks.",
)
rag_react_agent_compiled_graph = rag_react_agent.build_graph().compile()

# Web search agent
web_search_react_agent = ReActAgent(
    llm=llm,
    tools=[web_search],
    system_prompt="You are a helpful assistant. You can use the web_search tool to search the web for information.",
)
web_search_react_agent_compiled_graph = web_search_react_agent.build_graph().compile()
