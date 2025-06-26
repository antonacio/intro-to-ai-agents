# Legal Conversation Agent

A specialized ReAct agent for conducting legal client onboarding conversations. This agent is designed to interact with potential clients, understand their legal needs, and gather relevant information for creating pitch decks.

## Features

- **Interactive Conversations**: Conducts natural conversations with clients to understand their legal needs
- **Memory Support**: Maintains conversation history for context continuity
- **Tool Integration**: Uses specialized tools for:
  - Web search for legal information
  - Legal area classification
  - Client information extraction
  - Lawyer database search

## Usage

### Basic Example

```python
from langchain_openai import ChatOpenAI
from hackathon.agents.conversation import LegalConversationAgent

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Create agent
agent = LegalConversationAgent(llm=llm, use_memory=True)

# Run conversation
result = agent.run_conversation(
    "I need help with intellectual property protection",
    thread_id="client_123"
)
```

### Interactive Mode

```python
# Run the interactive demo
python src/hackathon/agents/conversation/example_usage.py
```

## Tools (TODO Implementation)

1. **search_web**: Search for legal information online
2. **classify_legal_area**: Classify the type of legal assistance needed
3. **extract_client_info**: Extract structured information from conversations
4. **search_lawyers_db**: Find appropriate lawyers based on expertise

## Next Steps

1. Implement the actual tool functionality
2. Connect to a real lawyer database
3. Add more sophisticated information extraction
4. Integrate with the drafting agent for pitch deck creation 