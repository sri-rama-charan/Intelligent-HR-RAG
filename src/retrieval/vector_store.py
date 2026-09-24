"""
FAISS Vector Store module for the HR Help Desk RAG pipeline.
Responsible for indexing normalized document embeddings using FAISS IndexFlatIP (Inner Product)
and performing exact Top-K similarity search while maintaining 1-to-1 chunk metadata mapping.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import faiss
import numpy as np
from langchain_core.documents import Document

from src.retrieval.embeddings import EmbeddingManager, DEFAULT_EMBEDDING_MODEL


class FAISSVectorStore:
    """
    Manages an in-memory FAISS IndexFlatIP vector store, linking each indexed vector
    back to its source LangChain Document and chunk metadata.
    """

    def __init__(self, dimension: int = 384):
        """
        Initializes an empty FAISS IndexFlatIP store.

        Args:
            dimension (int): Vector dimension (default 384 for all-MiniLM-L6-v2).
        """
        self.dimension = dimension
        # IndexFlatIP performs exact brute-force inner product search.
        # Since vectors are normalized (length = 1), inner product == cosine similarity.
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Parallel list holding the Document object for each vector in the index.
        # Index 0 in self.index corresponds to self.chunks[0], and so on.
        self.chunks: List[Document] = []

    @property
    def total_vectors(self) -> int:
        """Returns the total number of vectors currently stored in the FAISS index."""
        return self.index.ntotal

    def add_documents(self, chunks: List[Document], embeddings: np.ndarray) -> None:
        """
        Adds document chunks and their pre-computed embeddings to the FAISS index.

        Args:
            chunks (List[Document]): The list of chunked Document objects.
            embeddings (np.ndarray): 2D float32 array of shape (len(chunks), dimension).
        """
        if len(chunks) == 0:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: Got {len(chunks)} chunks but {len(embeddings)} embedding vectors."
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Dimension mismatch: Expected {self.dimension}, got {embeddings.shape[1]}."
            )

        # Ensure embeddings are contiguous float32 for FAISS C++ engine
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)

        # Add vectors to FAISS index
        self.index.add(vectors)

        # Store chunk documents in the same order
        self.chunks.extend(chunks)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Searches the FAISS index for the Top-K most similar chunks to the query vector.

        Args:
            query_vector (np.ndarray): 1D array of shape (dimension,) or 2D array of shape (1, dimension).
            top_k (int): Number of top relevant chunks to retrieve. Must be >= 1.

        Returns:
            List[Dict[str, Any]]: Ranked list of matching chunks with similarity scores and metadata.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        if self.index.ntotal == 0:
            return []

        # Prepare query vector format
        q_vec = np.ascontiguousarray(query_vector, dtype=np.float32)
        if q_vec.ndim == 1:
            q_vec = q_vec.reshape(1, -1)

        if q_vec.shape[1] != self.dimension:
            raise ValueError(
                f"Query vector dimension {q_vec.shape[1]} does not match index dimension {self.dimension}."
            )

        # Limit top_k to the total available vectors
        k = min(top_k, self.index.ntotal)

        # FAISS search: returns 2D arrays of shape (1, k)
        scores, indices = self.index.search(q_vec, k)

        results: List[Dict[str, Any]] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx == -1:  # FAISS padding if fewer than k vectors exist
                continue

            chunk = self.chunks[idx]
            results.append({
                "rank": rank,
                "score": float(score),
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "unknown"),
                "page": chunk.metadata.get("page", 0),
                "chunk_id": chunk.metadata.get("chunk_id", "unknown"),
                "document_title": chunk.metadata.get("document_title", ""),
                "chunk": chunk
            })

        return results

    @classmethod
    def build_from_chunks(
        cls,
        chunks: List[Document],
        embedder: EmbeddingManager
    ) -> "FAISSVectorStore":
        """
        Factory helper to embed chunks and build a populated FAISSVectorStore in one step.

        Args:
            chunks (List[Document]): The chunked documents to index.
            embedder (EmbeddingManager): Initialized embedding manager.

        Returns:
            FAISSVectorStore: Populated vector store ready for search.
        """
        store = cls(dimension=embedder.dimension)
        vectors, mapped_chunks = embedder.embed_chunks(chunks, normalize=True)
        store.add_documents(mapped_chunks, vectors)
        return store

    def save(self, directory_path: Union[str, Path]) -> None:
        """
        Saves the FAISS index and the chunks metadata list to disk.

        Args:
            directory_path (Union[str, Path]): Destination directory.
        """
        save_dir = Path(directory_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        index_file = save_dir / "faiss.index"
        chunks_file = save_dir / "chunks.pkl"

        # Save FAISS binary index
        faiss.write_index(self.index, str(index_file))

        # Save Python metadata & chunks list
        with open(chunks_file, "wb") as f:
            pickle.dump(self.chunks, f)

        print(f"Vector store saved to: {save_dir}")

    @classmethod
    def load(cls, directory_path: Union[str, Path]) -> "FAISSVectorStore":
        """
        Loads a previously saved FAISS index and its chunks metadata from disk.

        Args:
            directory_path (Union[str, Path]): Directory containing saved files.

        Returns:
            FAISSVectorStore: Restored vector store.
        """
        load_dir = Path(directory_path)
        index_file = load_dir / "faiss.index"
        chunks_file = load_dir / "chunks.pkl"

        if not index_file.exists() or not chunks_file.exists():
            raise FileNotFoundError(f"Missing FAISS files in directory: {load_dir}")

        index = faiss.read_index(str(index_file))
        with open(chunks_file, "rb") as f:
            chunks = pickle.load(f)

        store = cls(dimension=index.d)
        store.index = index
        store.chunks = chunks
        return store
