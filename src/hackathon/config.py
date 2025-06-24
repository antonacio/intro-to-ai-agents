import os
from enum import Enum


class ModelProviders(Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        elif isinstance(other, ModelProviders):
            return self.value == other.value
        else:
            raise ValueError(
                "Expect value to be an instance of string or ModelProviders"
            )


MODEL_PROVIDER = os.getenv("MODEL_PROVIDER")

if MODEL_PROVIDER == ModelProviders.OLLAMA:
    from langchain_ollama import ChatOllama
    from langchain_ollama import OllamaEmbeddings

    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
    # define the llm and embedding model
    llm = ChatOllama(model=OLLAMA_MODEL)
    embedding_model = OllamaEmbeddings(model=OLLAMA_MODEL)

elif MODEL_PROVIDER == ModelProviders.OPENAI:
    from langchain_openai import ChatOpenAI
    from langchain_openai import OpenAIEmbeddings

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL")
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")
    # define the llm and embedding model
    llm = ChatOpenAI(model=OPENAI_LLM_MODEL, api_key=OPENAI_API_KEY)
    embedding_model = OpenAIEmbeddings(
        model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY
    )
else:
    raise ValueError(
        f'Selected model provider is not supported: "{MODEL_PROVIDER}"\n'
        f"Supported model providers are: {[sp.value for sp in ModelProviders]}"
    )
