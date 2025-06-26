# Legal Multi-Agent Conversation System

This system implements a multi-agent workflow for legal client onboarding, featuring Iris (AI front-of-house) who gathers information and passes it to a drafting agent for pitch deck creation.

## 🏗️ Architecture

The system consists of:
- **Iris (Conversation Agent)**: AI front-of-house that classifies legal needs and gathers information
- **Drafting Agent**: Receives conversation data and prepares pitch decks (currently TODO implementation)
- **Human-in-the-Loop**: Interactive nodes for natural conversation flow

## 🚀 Quick Start

### Prerequisites

1. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

2. Install dependencies:
```bash
poetry install
poetry add fastapi "uvicorn[standard]"  # if not already installed
```

### Running the System

You have three ways to interact with the system:

#### 1. Interactive Demo (CLI)

Run the interactive command-line demo:

```bash
poetry run python src/hackathon/agents/conversation/interactive_demo.py
```

- Choose option **B** for the multi-agent workflow
- Iris will introduce herself and start the conversation
- Type your legal needs and have a natural conversation
- The system will automatically transition to drafting when complete

#### 2. FastAPI Backend

Start the API server:

```bash
poetry run uvicorn src.hackathon.agents.conversation.api:app --reload --port 8000
```

Then access:
- **Interactive Docs**: http://localhost:8000/docs
- **API Endpoints**: See below for details

#### 3. Python Script

```python
from hackathon.agents.conversation import MultiAgentLegalGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Initialize
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
workflow = MultiAgentLegalGraph(llm=llm)

# Start conversation (Iris introduces herself)
result = workflow.run({"messages": []}, thread_id="client_123")

# Continue conversation
result = workflow.run(
    {"messages": [HumanMessage(content="I need help with an acquisition")]}, 
    thread_id="client_123"
)
```

## 📡 API Endpoints

### Start Conversation
```http
POST /conversation/start
Content-Type: application/json

{
  "thread_id": "optional_custom_id"
}
```

Response:
```json
{
  "thread_id": "conv_123",
  "message": "Hello! I'm Iris, the AI front-of-house...",
  "status": "active"
}
```

### Send Message
```http
POST /conversation/message
Content-Type: application/json

{
  "thread_id": "conv_123",
  "message": "I need help with a merger"
}
```

Response:
```json
{
  "thread_id": "conv_123",
  "message": "I understand you need help with a merger...",
  "conversation_complete": false,
  "legal_area": "mergers_and_acquisitions",
  "status": "active"
}
```

### Get Status
```http
POST /conversation/status
Content-Type: application/json

{
  "thread_id": "conv_123"
}
```

### Get History
```http
GET /conversation/{thread_id}/history
```

## 🛠️ How It Works

### Conversation Flow

1. **Iris Introduction**: Iris introduces herself as the AI front-of-house
2. **Legal Area Classification**: Based on your input, Iris uses the `classify_legal_area` tool
3. **Targeted Questions**: Iris asks specific questions based on the legal area
4. **Information Gathering**: Continues until sufficient information is collected
5. **Handoff**: Calls `end_conversation` and passes data to drafting agent

### Available Legal Areas

- Mergers and Acquisitions
- Commercial Contracts
- Intellectual Property
- Data Protection
- Employment Disputes
- Regulatory Compliance
- And many more (see `LegalAreas` enum)

### Key Components

- **`conversation_agent.py`**: Iris implementation with ReAct pattern
- **`graph.py`**: Multi-agent workflow orchestration
- **`state.py`**: Shared state management
- **`drafting_node.py`**: Drafting agent (TODO implementation)
- **`api.py`**: FastAPI backend
- **`interactive_demo.py`**: CLI demo

## 💡 Usage Tips

### For Best Results

1. **Be specific** about your legal needs
2. **Provide context** about your company/situation
3. **Answer Iris's questions** to help classification
4. **Mention urgency** if time-sensitive

### Example Conversation

```
Iris: Hello! I'm Iris, the AI front-of-house for our law firm...

You: Hi, I'm the CEO of TechCorp. We're planning to acquire a competitor.

Iris: [Uses classify_legal_area tool → "mergers_and_acquisitions"]
      I see you're interested in an acquisition. Let me gather some key information...
      What's the approximate deal size?

You: Around $50 million

Iris: What stage is the transaction in? Have you signed an LOI?

[Conversation continues until Iris has enough information]

Iris: Perfect! I have all the information I need. I'm now passing this on to the relevant parties...
```

## 🔧 Development

### Adding New Legal Areas

1. Add to `LegalAreas` enum in `legal_tools.py`
2. Add area-specific guidance in `classify_legal_area` function
3. Update prompts as needed

### Implementing Drafting Agent

The drafting agent is currently a TODO implementation. To complete it:

1. Update `drafting_node.py` with actual pitch deck generation
2. Integrate with document generation tools
3. Add templates for different legal areas

### Testing

```bash
# Test conversation agent alone
poetry run python src/hackathon/agents/conversation/example_usage.py

# Test API endpoints
poetry run python src/hackathon/agents/conversation/api_example.py

# Run interactive demo
poetry run python src/hackathon/agents/conversation/interactive_demo.py
```

## 🐛 Troubleshooting

### Common Issues

1. **Recursion limit error**: The conversation agent is stuck in a loop
   - Solution: Already handled with 25-step limit

2. **No API key**: Set `OPENAI_API_KEY` environment variable

3. **Connection refused**: Make sure the API server is running

4. **Conversation not ending**: Iris needs more information
   - Provide specific details about your legal needs
   - Or type "quit" to force end

### Debug Mode

For detailed logging, set:
```bash
export LANGCHAIN_VERBOSE=true
```

## 📚 Further Reading

- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangChain ReAct Pattern](https://python.langchain.com/docs/modules/agents/agent_types/react) 