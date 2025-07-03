from langchain_core.tools import tool
from examples.config import (
    vector_store_collection_name,
    embedding_model,
    vector_store_directory,
)
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name=vector_store_collection_name,
    embedding_function=embedding_model,
    persist_directory=vector_store_directory,
)


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = (
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n\n"
    ) + "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs
