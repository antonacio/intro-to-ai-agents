"""Tools for the hackathon agents."""

from hackathon.tools.legal_tools import (
    search_web,
    classify_legal_area,
    extract_client_info,
    search_lawyers_db,
    end_conversation,
    retrieve_uploaded_docs,
)
from hackathon.tools.math import add, multiply, divide
from hackathon.tools.rag import retrieve

__all__ = [
    # Legal tools
    "search_web",
    "classify_legal_area",
    "extract_client_info",
    "search_lawyers_db",
    "end_conversation",
    "retrieve_uploaded_docs",
    # Math tools
    "add",
    "multiply",
    "divide",
    # RAG tools
    "retrieve",
]
