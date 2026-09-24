"""
Retrieval module: responsible for embeddings generation and vector store search (FAISS).
"""
from src.retrieval.embeddings import EmbeddingManager, DEFAULT_EMBEDDING_MODEL
from src.retrieval.vector_store import FAISSVectorStore

__all__ = ["EmbeddingManager", "DEFAULT_EMBEDDING_MODEL", "FAISSVectorStore"]
