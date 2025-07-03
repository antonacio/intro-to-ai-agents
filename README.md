# Introduction to AI Agents

This is a tutorial on how to build and orchestrate agentic systems using LangGraph. It was created for participants of the [LegalTechTalk Hackathon](https://www.legaltech-talk.com/legaltechtalk-hackathon/) taking place on June 26-27, 2025.

## 1. Setup

### Environment

First, rename the [`.env.example`](./.env.example) file to `.env` and add your team's OpenAI API key in the `OPENAI_API_KEY` field

Then, follow these steps to set up your Python environment:
1. Create a virtual environment with Python 3.12 using [UV](https://docs.astral.sh/uv/getting-started/installation/) or [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html):
    - **UV**: Run `uv venv --python 3.12` to create the environment, then `source .venv/bin/activate` to activate it
    - **conda**: Run `conda create -n legal-hack python=3.12` to create the environment, then `conda activate legal-hack` to activate it
2. Run `pip install -r requirements.txt` to install the project dependencies
3. Run `pip install -e .` to install this repository's source code in editable mode
4. Run `pre-commit install` to install pre-commit hooks

### LangGraph Studio

First, let's clarify some terms:
- [LangChain](https://python.langchain.com/docs/introduction/) is an open-source framework for developing applications powered by large language models (LLMs)
- [LangGraph](https://langchain-ai.github.io/langgraph/) is an open-source framework built on top of LangChain for creating AI agents and multi-agent applications
- [LangSmith](https://docs.smith.langchain.com/) is a unified observability and evaluation platform to debug, test, and monitor LLM apps
- [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/) is a web UI for visualizing, interacting with, and debugging LangGraph agents. It integrates seamlessly with LangSmith for LLM tracing

In this tutorial, we will use LangGraph Studio to interact with our AI agents. Follow the instructions below to set up LangGraph Studio:
1. LangSmith integration:
    - **With tracing (recommended):** [Sign up for LangSmith](https://smith.langchain.com/settings), create an API key (Personal Access Token), and add it to your `.env` file in the `LANGSMITH_API_KEY` field. The traces will be available at the [LangSmith Platform](https://smith.langchain.com/)
    - **Without tracing:** Delete the `LANGSMITH_API_KEY` field and set `LANGSMITH_TRACING` to `false` in your `.env` file. LangGraph Studio will display a warning about the missing LangSmith API key, which can be safely ignored.
2. Run `langgraph dev` in your terminal to start the LangGraph API server locally
3. Access the LangGraph Studio Web UI at: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

## 2. Let's Build AI Agents!

To start building AI Agents, work through the content of these notebooks:
- [`1_graph-basics.ipynb`](notebooks/1_graph-basics.ipynb) introduces basic LangGraph concepts like State, Nodes, Edges, and Graphs
- [`2_math-agent.ipynb`](notebooks/2_math-agent.ipynb) builds a simple ReAct agent that calls math tools to perform arithmetic operations
- [`3_rag-agent.ipynb`](notebooks/3_rag-agent.ipynb) builds a pipeline to ingest and index data from source documents and creates a ReAct agent that can answer questions about those documents through Retrieval Augmented Generation (RAG)

If you'd like to ingest and index additional documents for the RAG agent, move the desired files into the `data/` folder and then run `python src/hackathon/index_data.py`. This command executes the indexing pipeline, which loads every document in the `data/` directory, splits each one into chunks, embeds those chunks as vectors, and stores the resulting embeddings in a vector store under the `vector_store/` directory.

Once you have completed the notebooks, feel free to explore the source code in `src/hackathon/`, interact with the agents in LangGraph Studio, and start building your own agentic systems. I'm excited to see what you build!

## 3. Additional Resources (Optional)

### Run LLMs Locally with Ollama

[Ollama](https://ollama.com/) is an open-source tool that allows you to run large language models locally on your machine. Follow these steps to set up Ollama:
1. Download and install the Ollama desktop app from https://ollama.com/download
2. Run the command `ollama` in your terminal to verify the installation was successful
3. Select a model from the [Ollama library](https://ollama.com/library). Keep in mind that:
    - The model should support tool/function calling to enable agentic use cases
    - As a general rule, you should have at least 8 GB of RAM available to run 8B models, 16 GB to run 16B models, and so on
    - We recommend using the [Qwen3 family of models](https://ollama.com/library/qwen3): `qwen3:8b` if you have 8 GB of RAM, `qwen3:14b` if you have 16 GB, or `qwen3:32b` if you have 32 GB
4. Run the command `ollama run <your_model>` in your terminal to download the model. When the download is complete, a chat interface will start with the selected model. Type `/bye` to exit the chat interface
5. In the `.env` file, set `MODEL_PROVIDER` to `"ollama"` and `OLLAMA_MODEL` to your model of choice
6. Finally, start the Ollama app in the background OR run `ollama serve` in your terminal to expose an HTTP API on localhost so you can send requests to the model via Python code

### Explore Further

See the following resources to learn more about:
- AI agents: https://www.anthropic.com/engineering/building-effective-agents
- LangGraph (free) course: https://academy.langchain.com/courses/intro-to-langgraph
- LangGraph advanced examples: https://langchain-ai.github.io/langgraph/tutorials/overview
