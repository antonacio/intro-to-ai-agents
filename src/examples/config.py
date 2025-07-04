import os
import logging
import chromadb
from enum import Enum
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env file

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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

if not MODEL_PROVIDER:
    raise ValueError(
        "MODEL_PROVIDER environment variable is not set. "
        "Please set it to one of the following values: "
        f"{[sp.value for sp in ModelProviders]}"
    )
elif MODEL_PROVIDER == ModelProviders.OLLAMA:
    from langchain_ollama import ChatOllama
    from langchain_ollama import OllamaEmbeddings

    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
    logging.info(f"Using Ollama model: {OLLAMA_MODEL}")

    # define the llm and embedding model
    llm = ChatOllama(model=OLLAMA_MODEL)
    embedding_model = OllamaEmbeddings(model=OLLAMA_MODEL)

elif MODEL_PROVIDER == ModelProviders.OPENAI:
    from langchain_openai import ChatOpenAI
    from langchain_openai import OpenAIEmbeddings

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL")
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")
    logging.info(f"Using OpenAI model: {OPENAI_LLM_MODEL}")
    logging.info(f"Using OpenAI embedding model: {OPENAI_EMBEDDING_MODEL}")

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

# Get the directory where this config file is located
_config_dir = Path(__file__).parent

# data directory (relative to project root)
data_directory = str((_config_dir / "../../data/").resolve())

# vector store
vector_store_collection_name = "rag_collection"
vector_store_directory = str((_config_dir / "../../vector_store/").resolve())
vector_store_client = chromadb.PersistentClient(
    path=vector_store_directory,
    settings=chromadb.config.Settings(anonymized_telemetry=False),
)
