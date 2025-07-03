import os
from examples.agents.react.agent import ReActAgent
from examples.tools.math import add, multiply, divide
from examples.tools.rag import retrieve
from examples.config import llm, ModelProviders

if os.getenv("MODEL_PROVIDER") == ModelProviders.OPENAI:
    # we set parallel tool calling to false as math generally is done sequentially
    bind_math_tools_kwargs = {"parallel_tool_calls": False}
else:
    # Ollama model in LangChain does not support the parallel tool calling parameter
    bind_math_tools_kwargs = {}

# Math agent
react_math_agent = ReActAgent(
    llm=llm,
    tools=[add, multiply, divide],
    system_prompt="You are a helpful assistant tasked with performing arithmetic on a set of inputs.",
    bind_tools_kwargs=bind_math_tools_kwargs,
)
react_math_agent_compiled_graph = react_math_agent.build_graph().compile()

# RAG agent
react_rag_agent = ReActAgent(
    llm=llm,
    tools=[retrieve],
    system_prompt="You are a helpful assistant for question-answering tasks.",
)
react_rag_agent_compiled_graph = react_rag_agent.build_graph().compile()
