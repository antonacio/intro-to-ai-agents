from langchain_core.tools import tool
from textwrap import dedent
from examples.config import (
    vector_store_collection_name,
    embedding_model,
    vector_store_client,
)
from langchain_chroma import Chroma

# get the collection from the vector store to retrieve the metadata
collection = vector_store_client.get_or_create_collection(vector_store_collection_name)
# instantiate the vector store
vector_store = Chroma(
    client=vector_store_client,
    collection_name=vector_store_collection_name,
    collection_metadata=collection.metadata,
    embedding_function=embedding_model,
)

retrieval_tool_description = f"""\
Search and retrieve information from documents to answer a user query.
You have access to the following {vector_store._collection_metadata["num_files"]} document(s):
{vector_store._collection_metadata["file_names"]}
"""


@tool(response_format="content_and_artifact", description=retrieval_tool_description)
def retrieve(query: str):
    # retrieve documents from the vector store with max marginal relevance
    retrieved_chunks = vector_store.max_marginal_relevance_search(query, k=3)
    # format the retrieved chunks into a single string
    context = "\n\n".join(
        (
            f"## {i}. Retrieved Document Chunk\n\n"
            f"### Chunk Metadata:\n{doc.metadata}\n\n"
            f"### Chunk Content:\n{doc.page_content}"
        )
        for i, doc in enumerate(retrieved_chunks, start=1)
    )
    # build message with the context to be used by the LLM
    context_message = dedent(
        """\
        Use the following pieces of context retrieved from the documents to answer the question.
        If you don't have enough information to answer the question, say that you can't answer it.

        # Context

        {context}
        """
    ).format(context=context)

    return context_message, retrieved_chunks
