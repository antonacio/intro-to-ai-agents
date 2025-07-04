import logging
import os
import asyncio
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from examples.config import (
    data_directory,
    vector_store_collection_name,
    vector_store_client,
    embedding_model,
)
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


async def load_pdf_data(file_names: list[str]) -> list[Document]:
    logging.info(f"STEP 1. Loading PDF data from {data_directory} ...")
    # create a list to store all pages from all PDF documents
    docs_pages = []
    # load pages from the PDF documents
    for i, file_name in enumerate(file_names, start=1):
        if file_name.lower().endswith(".pdf"):
            logging.info(f"  [{i}/{len(file_names)}] Loading {file_name} ...")
            loader = PyPDFLoader(
                file_path=os.path.join(data_directory, file_name),
                mode="page",
                extract_images=False,
                extraction_mode="plain",
            )
            async for page in loader.alazy_load():
                docs_pages.append(page)
            logging.info(f"  Loaded {len(docs_pages)} pages from {file_name}")
    logging.info(
        f"  Loaded {len(docs_pages)} pages from {len(file_names)} PDF document(s)"
    )

    return docs_pages


def filter_metadata(docs_pages: list[Document]) -> list[Document]:
    logging.info("STEP 1.1. Filtering metadata from the loaded documents ...")
    # filter unnecessary metadata from the loaded documents
    metadata_to_remove = ["producer", "creator", "creationdate", "moddate"]

    for page in docs_pages:
        # remove unnecessary metadata
        for metadata_key in metadata_to_remove:
            page.metadata.pop(metadata_key, None)
        # make page numbers start at 1 (PyPDFLoader indexes pages from 0)
        if "page" in page.metadata and isinstance(page.metadata["page"], int):
            page.metadata["page"] += 1
    logging.info(f"  Filtered metadata from {len(docs_pages)} documents")

    return docs_pages


def split_data(docs_pages: list[Document]) -> list[Document]:
    logging.info("STEP 2. Splitting the loaded documents into chunks ...")
    # split loaded pages into chunks of 1000 characters with 200 characters of overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    doc_splits = text_splitter.split_documents(docs_pages)
    logging.info(
        f"  Splitted {len(docs_pages)} documents into {len(doc_splits)} chunks"
    )

    return doc_splits


def embed_and_store_data(doc_splits: list[Document], file_names: list[str]):
    logging.info("STEP 3. Embedding and storing the chunks in the vector store ...")
    # create the vector store
    logging.info(
        f"  Creating the vector store with collection name '{vector_store_collection_name}'"
    )
    vector_store = Chroma(
        client=vector_store_client,
        collection_name=vector_store_collection_name,
        collection_metadata={
            "num_files": len(file_names),
            "file_names": ", ".join(file_names),
        },
        embedding_function=embedding_model,
    )
    # index chunks
    chunk_indexes = vector_store.add_documents(documents=doc_splits)
    logging.info(f"  Indexed {len(chunk_indexes)} chunks in the vector store")


if __name__ == "__main__":
    logging.info("--> Starting the Data Indexing pipeline ...")
    # get file names from data directory
    file_names = os.listdir(data_directory)
    # load PDF data
    docs_pages = asyncio.run(load_pdf_data(file_names))
    # filter metadata
    docs_pages = filter_metadata(docs_pages)
    # split data
    doc_splits = split_data(docs_pages)
    # embed and store data
    embed_and_store_data(doc_splits, file_names)
    logging.info("--> Data Indexing pipeline completed successfully!")
