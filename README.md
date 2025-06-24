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

[LangGraph](https://langchain-ai.github.io/langgraph/) is an open-source framework for building agents and multi-agent applications. [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/) is a web UI that enables visualization, interaction, and debugging of agentic systems built with LangGraph. Follow these steps to set up LangGraph Studio:
1. [Sign up for LangSmith](https://smith.langchain.com/settings?__hstc=5909356.74979cac47c29358b9e8426e0283c1f3.1750601004239.1750619531728.1750635426214.3&__hssc=5909356.4.1750635426214&__hsfp=268443588&_gl=1*1l57sy0*_gcl_au*Nzc0NTQ0NTcyLjE3NTA2MDEwMDI.*_ga*OTQ4MzIzOTcyLjE3NDA2MTgxMTQ.*_ga_47WX3HKKY2*czE3NTA2MzU0MjQkbzYkZzEkdDE3NTA2MzY1NjYkajYwJGwwJGgw), create an API key (Personal Access Token), and add it to your `.env` file in the `LANGSMITH_API_KEY` field
2. Run `langgraph dev` in your terminal to start the LangGraph API server locally
3. Access the LangGraph Studio Web UI at https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

## 2. Let's Build AI Agents!

To start building AI Agents, work through the content of these notebooks:
- [`1_simple-graph.ipynb`](notebooks/1_simple-graph.ipynb) introduces basic LangGraph concepts like State, Nodes, Edges, and Graphs
- [`2_react_agent.ipynb`](notebooks/2_react_agent.ipynb) builds a simple ReAct agent that calls math tools to perform arithmetic operations

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
- AI Agents: https://www.anthropic.com/engineering/building-effective-agents
- LangGraph: https://academy.langchain.com/courses/intro-to-langgraph
- Ollama: https://github.com/ollama/ollama/tree/main/docs
