"""
BM25 Lexical Retrieval module for the HR Help Desk RAG pipeline.
Responsible for indexing document chunks using BM25Okapi and performing
keyword/lexical similarity search while maintaining 1-to-1 chunk metadata mapping.
"""

import re
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


def default_tokenizer(text: str) -> List[str]:
    """
    Standard lowercased alphanumeric tokenizer for BM25 lexical matching.
    Splits text on non-alphanumeric characters while preserving numbers,
    acronyms, and policy codes.
    """
    if not text:
        return []
    return re.findall(r"\b\w+\b", text.lower())


class BM25Store:
    """
    Manages an in-memory BM25Okapi lexical index, linking each indexed document
    back to its source LangChain Document and chunk metadata.
    """

    def __init__(self, chunks: Optional[List[Document]] = None):
        """
        Initializes the BM25 store.

        Args:
            chunks (Optional[List[Document]]): Initial chunks to index.
        """
        self.chunks: List[Document] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

        if chunks:
            self.add_documents(chunks)

    @property
    def total_documents(self) -> int:
        """Returns the total number of documents currently stored in the BM25 index."""
        return len(self.chunks)

    def add_documents(self, chunks: List[Document]) -> None:
        """
        Adds document chunks and indexes them using BM25Okapi.

        Args:
            chunks (List[Document]): The list of chunked Document objects.
        """
        if not chunks:
            return

        self.chunks.extend(chunks)
        new_tokens = [default_tokenizer(chunk.page_content) for chunk in chunks]
        self.corpus_tokens.extend(new_tokens)
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches the BM25 index for the Top-K most relevant chunks to the query string.

        Args:
            query (str): The search query.
            top_k (int): Number of top relevant chunks to retrieve. Must be >= 1.

        Returns:
            List[Dict[str, Any]]: Ranked list of matching chunks with lexical scores and metadata.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        if not query or not query.strip() or self.total_documents == 0 or self.bm25 is None:
            return []

        query_tokens = default_tokenizer(query)
        if not query_tokens:
            return []

        # Compute BM25 scores across all indexed chunks
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices sorted by score descending
        # Argsort returns ascending, so take reverse slice
        sorted_indices = scores.argsort()[::-1]
        k = min(top_k, len(sorted_indices))

        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(sorted_indices[:k], start=1):
            chunk = self.chunks[idx]
            results.append({
                "rank": rank,
                "score": float(scores[idx]),
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "unknown"),
                "page": chunk.metadata.get("page", 0),
                "chunk_id": chunk.metadata.get("chunk_id", "unknown"),
                "document_title": chunk.metadata.get("document_title", ""),
                "chunk": chunk
            })

        return results

    @classmethod
    def build_from_chunks(cls, chunks: List[Document]) -> "BM25Store":
        """
        Factory helper to index chunks and return a populated BM25Store.

        Args:
            chunks (List[Document]): The chunked documents to index.

        Returns:
            BM25Store: Populated BM25 store ready for search.
        """
        return cls(chunks=chunks)
