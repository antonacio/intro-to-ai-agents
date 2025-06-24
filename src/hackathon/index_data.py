import logging

from langchain_community.document_loaders import DirectoryLoader
from hackathon.config import (
    data_directory,
    vector_store_collection_name,
    vector_store_directory,
    embedding_model,
)
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load documents
loader = DirectoryLoader(
    path=data_directory, show_progress=True, use_multithreading=True
)
documents = loader.load()
logging.info(f"Indexing data: Loaded {len(documents)} documents")

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
doc_splits = text_splitter.split_documents(documents)
logging.info(
    f"Indexing data: Split {len(documents)} documents into {len(doc_splits)} chunks"
)

# Embed and store documents
vector_store = Chroma(
    collection_name=vector_store_collection_name,
    embedding_function=embedding_model,
    persist_directory=vector_store_directory,
)
chunk_indexes = vector_store.add_documents(doc_splits)
logging.info(f"Indexing data: Stored {len(chunk_indexes)} chunks in the vector store")
