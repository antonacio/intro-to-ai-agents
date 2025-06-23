from hackathon.agents.react.agent import ReActAgent
from hackathon.tools.math import add, multiply, divide
from hackathon.config import OLLAMA_MODEL
from langchain_ollama import ChatOllama

llm = ChatOllama(model=OLLAMA_MODEL)

react_math_agent = ReActAgent(
    llm=llm,
    tools=[add, multiply, divide],
    system_prompt="You are a helpful assistant tasked with performing arithmetic on a set of inputs.",
)
react_math_agent_compiled_graph = react_math_agent.build_graph().compile()
