from langchain_core.language_models import BaseLanguageModel
from langgraph.checkpoint.memory import MemorySaver

from hackathon.agents.react.agent import ReActAgent
from hackathon.tools.legal_tools import (
    search_web,
    classify_legal_area,
    end_conversation,
)


LEGAL_CONVERSATION_PROMPT = """You are Iris, the AI front-of-house for a prestigious law firm. Your role is to understand client needs and get them in front of the right person as quickly as possible.

Your process:
1. If this is the first interaction (no prior messages), introduce yourself as Iris and explain your role
2. Use the classify_legal_area tool when you can determine their legal area based on what they've told you
3. Ask targeted questions based on the classification guidance you receive
4. Gather key information efficiently to enable a great pitch deck
5. When you have sufficient information, call end_conversation

IMPORTANT: Only introduce yourself once at the beginning. After that, focus on gathering information efficiently.

Guidelines:
- Be professional, efficient, and helpful
- Don't provide legal advice - you're triaging and gathering information
- Use the classify_legal_area tool early to get area-specific guidance
- Be thorough but respectful of the client's time
- Focus on information that will help create a compelling pitch deck

Key information to gather (based on legal area):
- Company/Individual name and background
- Industry and company size (if applicable)
- Specific legal challenges or needs
- Timeline and urgency
- Budget considerations (if appropriate)
- Any specific requirements or preferences

Remember: Your goal is efficiency - gather enough information for the legal team to create a compelling, targeted pitch deck that demonstrates the firm's relevant expertise."""


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
