"""LangGraph Studio script for the Research Agent."""

from examples.agents.researcher.agent import ResearchAgent
from examples.config import llm

# Research agent
research_agent = ResearchAgent(llm=llm)
research_agent_compiled_graph = research_agent.build_graph().compile()
