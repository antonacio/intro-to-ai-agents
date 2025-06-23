# Intro to AI Agents

This repository provides a tutorial on building AI agents using Ollama to run LLMs locally and LangGraph to orchestrate agentic systems. It was created for participants of the [LegalTechTalk Hackathon](https://www.legaltech-talk.com/legaltechtalk-hackathon/) taking place on June 26-27, 2025.

## Installation

### Python environment

Follow these steps to set up your Python environment:
1. Create a virtual environment with Python 3.12 using [UV](https://docs.astral.sh/uv/getting-started/installation/) or [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html):
    - UV: `uv venv --python 3.12` to create the environment and `source .venv/bin/activate` to activate it
    - conda: `conda create -n legal-hack python=3.12` to create the environment and `conda activate legal-hack` to activate it
2. Run `pip install -r requirements.txt` to install the project dependencies

### Ollama

[Ollama](https://ollama.com/) is an open-source tool that allows you to run large language models locally on your machine. Follow these steps to set up Ollama:
1. Download and install the Ollama desktop app from https://ollama.com/download
2. Run the command `ollama` in your terminal to verify the installation was successful
3. Select a model from the [Ollama library](https://ollama.com/library). Keep in mind that:
    - The model should support tool/function calling to enable agentic use cases
    - As a general rule, you should have at least 8 GB of RAM available to run 8B models, 16 GB to run 16B models, and so on...
    - We recommend using the [Qwen3 family of models](https://ollama.com/library/qwen3): `qwen3:8b` if you have 8 GB of RAM, `qwen3:14b` if you have 16 GB, or `qwen3:32b` if you have 32 GB
4. Run the command `ollama run <your_model>` in your terminal to download the model. We will use `qwen3:14b` in this repository
5. When the download is complete, a chat interface will start with the selected model. Type `/bye` to exit the chat interface
6. Start the Ollama App in the background OR run `ollama serve` in your terminal to expose an HTTP API on localhost so that you can send requests to the model via Python code

### LangGraph Studio

[LangGraph](https://langchain-ai.github.io/langgraph/) is an open-source framework for building agents and multi-agent applications. [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/) is a web UI that enables visualization, interaction, and debugging of agentic systems built with LangGraph. Follow these steps to set up LangGraph Studio:
1. [Sign up to LangSmith](https://smith.langchain.com/settings?__hstc=5909356.74979cac47c29358b9e8426e0283c1f3.1750601004239.1750619531728.1750635426214.3&__hssc=5909356.4.1750635426214&__hsfp=268443588&_gl=1*1l57sy0*_gcl_au*Nzc0NTQ0NTcyLjE3NTA2MDEwMDI.*_ga*OTQ4MzIzOTcyLjE3NDA2MTgxMTQ.*_ga_47WX3HKKY2*czE3NTA2MzU0MjQkbzYkZzEkdDE3NTA2MzY1NjYkajYwJGwwJGgw) and create an API key (Personal Access Token)
2. Rename the [`.env.example`](./.env.example) file to `.env` and paste your LangSmith API key in

## Additional References

To learn more about:
- AI Agents: https://www.anthropic.com/engineering/building-effective-agents
- LangGraph: https://academy.langchain.com/courses/intro-to-langgraph
- Ollama: https://github.com/ollama/ollama/tree/main/docs
