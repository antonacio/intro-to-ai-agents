from hackathon.agents.dummy.agent import DummyAgent
from hackathon.config import OLLAMA_MODEL
from langchain_ollama import ChatOllama

llm = ChatOllama(model=OLLAMA_MODEL)

dummy_agent = DummyAgent(llm=llm)
dummy_agent_compiled_graph = dummy_agent.build_graph().compile()
