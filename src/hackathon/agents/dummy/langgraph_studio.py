from hackathon.agents.dummy.agent import DummyAgent
from hackathon.config import llm

dummy_agent = DummyAgent(llm=llm)
dummy_agent_compiled_graph = dummy_agent.build_graph().compile()
