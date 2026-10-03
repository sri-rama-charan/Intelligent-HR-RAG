"""
Diagnostics module for analyzing FAISS retrieval performance, ranking tiers,
and chunk-level retrieval behavior in the Intelligent HR RAG Chatbot.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from src.retrieval.embeddings import EmbeddingManager
from src.retrieval.vector_store import FAISSVectorStore


def diagnose_query_retrieval(
    vector_store: FAISSVectorStore,
    embedder: EmbeddingManager,
    query: str,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Retrieves the Top-K chunks for a query from the vector store and formats
    the diagnostic information (rank, document, page, chunk_id, score, preview).

    Args:
        vector_store: Initialized and populated FAISSVectorStore.
        embedder: Initialized EmbeddingManager.
        query: Query string.
        top_k: Number of chunks to retrieve (default: 10).

    Returns:
        List[Dict[str, Any]]: List of diagnostic chunk records.
    """
    query_vec = embedder.embed_query(query, normalize=True)
    raw_results = vector_store.search(query_vec, top_k=top_k)

    diagnostics = []
    for item in raw_results:
        preview = item["text"][:150].replace("\n", " ").strip()
        diagnostics.append({
            "rank": item["rank"],
            "document": item["source"],
            "page": item["page"],
            "chunk_id": item["chunk_id"],
            "score": round(item["score"], 4),
            "preview": preview,
            "full_text": item["text"]
        })
    return diagnostics


def find_chunk_rank(
    vector_store: FAISSVectorStore,
    embedder: EmbeddingManager,
    query: str,
    target_chunk_id: str
) -> Optional[Dict[str, Any]]:
    """
    Searches across all vectors in the vector store to locate the exact rank
    and score of a specific chunk ID for a given query.

    Args:
        vector_store: Initialized FAISSVectorStore.
        embedder: Initialized EmbeddingManager.
        query: Query string.
        target_chunk_id: Chunk ID to find (e.g. '02_Leave_Policy_p3_c0').

    Returns:
        Optional[Dict[str, Any]]: Dict with rank, score, chunk_id, document, page if found, else None.
    """
    total = vector_store.total_vectors
    if total == 0:
        return None

    query_vec = embedder.embed_query(query, normalize=True)
    all_results = vector_store.search(query_vec, top_k=total)

    for item in all_results:
        if item.get("chunk_id") == target_chunk_id:
            return {
                "rank": item["rank"],
                "score": round(item["score"], 4),
                "chunk_id": item["chunk_id"],
                "document": item["source"],
                "page": item["page"],
                "full_text": item["text"]
            }
    return None


def evaluate_rank_tier(rank: Optional[int]) -> Dict[str, str]:
    """
    Evaluates whether a given rank falls within Top-3, Top-5, and Top-10.

    Args:
        rank: 1-indexed rank or None.

    Returns:
        Dict[str, str]: {'top_3': 'Yes'/'No', 'top_5': 'Yes'/'No', 'top_10': 'Yes'/'No'}
    """
    if rank is None or rank <= 0:
        return {"top_3": "No", "top_5": "No", "top_10": "No"}

    return {
        "top_3": "Yes" if rank <= 3 else "No",
        "top_5": "Yes" if rank <= 5 else "No",
        "top_10": "Yes" if rank <= 10 else "No"
    }
