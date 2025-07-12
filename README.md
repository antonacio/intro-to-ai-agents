# Let's build AI Agents!

This is a hands-on tutorial for building and orchestrating agentic systems, which I presented to the participants of the [LegalTechTalk Hackathon](https://www.legaltech-talk.com/legaltechtalk-hackathon/) (London, 26-27 June 2025).

In this tutorial you will build and run to the following AI Agents:
- Research Agent with Agentic RAG
- RAG Agent
- Web Search Agent
- Math Agent

## 1. Setup

### 🐍 Environment

First, create a `.env` file by running `cp .env.example .env` in your terminal and add your OpenAI API key in the `.env` file's `OPENAI_API_KEY` field.

Then, create a virtual environment with Python 3.12 using [UV](https://docs.astral.sh/uv/getting-started/installation/) or [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html).
- If you are using **conda**, run:
```bash
conda create -n ai-agents python=3.12  # create the environment
conda activate ai-agents               # activate the environment
pip install -r requirements.txt        # install dependencies
pip install -e .                       # install this repo's source code in editable mode
```
- If you are using **UV**, run:
```bash
uv venv --python 3.12                  # create the environment
source .venv/bin/activate              # activate the environment
uv pip install -r requirements.txt     # install dependencies
uv pip install -e .                    # install this repo's source code in editable mode
```

Finally, run `pre-commit install` to install pre-commit hooks.

### 🗂️ Data

Some agents in this tutorial retrieve information from source documents using a technique known as [retrieval-augmented generation (RAG)](https://python.langchain.com/docs/concepts/rag/). RAG enables LLMs to incorporate domain-specific, up-to-date information instead of relying solely on their static training data, improving the accuracy and reliability of their answers.

For demonstration purposes, this repo includes [Meta’s Terms of Service](https://mbasic.facebook.com/legal/terms/plain_text_terms/) in the `data/` folder. To use your own documents, simply drop them into that same folder.

Next, ingest and index the documents by running:

```bash
python src/examples/ingest_data.py
```

This script loads every file in `data/`, splits each document into chunks, embeds those chunks as vectors, and stores the resulting embeddings in a vector store located at `vector_store/`. For a detailed walkthrough of that process, see the [`3_rag-agent.ipynb`](notebooks/3_rag-agent.ipynb) notebook.

>Note: The ingestion pipeline currently supports PDF files only

## 2. Usage

First, let's clarify some terms from the [LangChain](https://python.langchain.com/docs/introduction/) ecosystem:
- [LangGraph](https://langchain-ai.github.io/langgraph/) is an open-source framework for building AI agents and multi-agent systems using LLMs
- [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/) is a web UI for visualizing, interacting with, and debugging LangGraph agents. It integrates seamlessly with LangSmith for LLM tracing
- [LangSmith](https://docs.smith.langchain.com/) is an observability and evaluation platform to debug, test, and monitor LLM apps

In this tutorial, we use LangGraph to build and LangGraph Studio (+ LangSmith) to interact with our AI agents.

### 🦜 LangGraph Studio

Follow the steps below to configure and run LangGraph Studio:
1. LangSmith integration:
    - **With tracing (recommended):** [Sign up for LangSmith](https://smith.langchain.com/settings), create an API key (Personal Access Token), and add it to your `.env` file in the `LANGSMITH_API_KEY` field. The traces will be available at the [LangSmith Platform](https://smith.langchain.com/)
    - **Without tracing:** Delete the `LANGSMITH_API_KEY` field and set `LANGSMITH_TRACING` to `false` in your `.env` file. LangGraph Studio will display a warning about the missing LangSmith API key, which can be safely ignored.
2. Run `langgraph dev` in your terminal to start the LangGraph API server locally
3. Access the LangGraph Studio Web UI and interact with the AI Agents at: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

### 📔 Notebook Tutorials

To learn the fundamentals of **building agents with LangGraph**, walk through these notebooks:
- [`1_graph-basics.ipynb`](notebooks/1_graph-basics.ipynb) — introduces core LangGraph concepts such as State, Nodes, Edges, and Graphs
- [`2_math-agent.ipynb`](notebooks/2_math-agent.ipynb) — builds a simple ReAct agent that invokes math tools to perform arithmetic operations
- [`3_rag-agent.ipynb`](notebooks/3_rag-agent.ipynb) — shows how to ingest data from source documents and create a ReAct agent that answers questions about those documents via RAG

### 🤖 Source Code

After you complete the notebook tutorial, dive into `src/examples/` and start building your own agentic systems. A great first stop is the [**ResearchAgent**](src/examples/agents/researcher/agent.py)—study its inner workings and experiment from there.

Contributions of any kind are welcome. Have an idea, question, or bug fix? Open an issue or submit a pull request to improve the code, docs, or tests. Every suggestion and patch makes the project better for everyone.

I’m excited to see what you build!

## 3. Additional Resources (Optional)

### 🦙 Run LLMs locally with Ollama

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

### 📚 Explore Further

See the following resources to learn more about:
- AI agents: https://www.anthropic.com/engineering/building-effective-agents
- LangGraph (free) course: https://academy.langchain.com/courses/intro-to-langgraph
- LangGraph advanced examples: https://langchain-ai.github.io/langgraph/tutorials/overview
