from langchain_core.language_models import BaseLanguageModel
from langgraph.checkpoint.memory import MemorySaver

from hackathon.agents.react.agent import ReActAgent
from hackathon.tools.legal_tools import (
    search_web,
    classify_legal_area,
    end_conversation,
)


LEGAL_CONVERSATION_PROMPT = """You are a professional legal intake specialist working for a prestigious law firm. 
Your role is to conduct an initial consultation with potential clients to understand their legal needs.

Your objectives are:
1. Build rapport and make the client feel comfortable
2. Understand the type of legal assistance they need
3. Gather relevant information about their situation
4. Classify their legal needs into appropriate practice areas
5. Collect key information that will help the law firm prepare a tailored pitch

Guidelines for the conversation:
- Be professional yet warm and approachable
- Ask clarifying questions to better understand their needs
- Don't provide legal advice - you're gathering information only
- Be thorough but respectful of the client's time
- Use the tools available to classify legal areas and search for relevant information
- Keep track of important details mentioned by the client

Key information to gather:
- Company/Individual name and background
- Industry or sector
- Specific legal challenges or needs
- Timeline and urgency
- Company size (if applicable)
- Any specific requirements or preferences

Remember: Your goal is to gather enough information for the drafting team to create a compelling pitch deck 
that showcases the firm's relevant expertise and proposed team."""


class ConversationAgent(ReActAgent):
    """Specialized ReAct agent for legal client onboarding conversations."""

    def __init__(self, llm: BaseLanguageModel, use_memory: bool = True, **kwargs):
        """Initialize the legal conversation agent.

        Args:
            llm: The language model to use
            use_memory: Whether to use memory for conversation continuity
            **kwargs: Additional arguments for the base ReActAgent
        """
        # Define the tools for this agent
        tools = [
            search_web,
            classify_legal_area,
            end_conversation,
        ]

        # Initialize with legal-specific system prompt
        super().__init__(
            llm=llm, tools=tools, system_prompt=LEGAL_CONVERSATION_PROMPT, **kwargs
        )

        # Set up memory if requested
        self.memory = MemorySaver() if use_memory else None

    def run_conversation(self, user_input: str, thread_id: str = "default") -> dict:
        """Run a conversation turn with the user.

        Args:
            user_input: The user's message
            thread_id: Thread ID for conversation continuity

        Returns:
            dict: The agent's response and extracted information
        """
        # Prepare the input
        input_data = {"messages": [("user", user_input)]}

        # Run configuration with thread ID for memory
        run_config = {"configurable": {"thread_id": thread_id}} if self.memory else {}

        # Run the agent
        result = self.run(
            input=input_data, run_config=run_config, checkpointer=self.memory
        )

        return result
