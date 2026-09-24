"""
Embedding module for the HR Help Desk RAG pipeline.
Responsible for converting text chunks and user queries into dense numerical vectors
using a local SentenceTransformer model (all-MiniLM-L6-v2).
"""

from typing import List, Tuple, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

# Default lightweight, fast, CPU-friendly embedding model
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class EmbeddingManager:
    """
    Manages loading the embedding model and converting text (chunks or queries)
    into numerical vectors.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        """
        Initializes and loads the SentenceTransformer embedding model.

        Args:
            model_name (str): Hugging Face model identifier.
        """
        self.model_name = model_name
        print(f"Loading embedding model: '{self.model_name}' (runs locally on CPU)...")
        self.model = SentenceTransformer(self.model_name)
        
        # Get the fixed vector dimension produced by this model
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension = self.model.get_embedding_dimension()
        else:
            self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Embedding model loaded successfully. Vector dimension: {self.dimension}")

    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Converts a list of document chunk texts into a 2D matrix of vectors.

        Args:
            texts (List[str]): List of chunk texts to encode.
            batch_size (int): Batch size for encoding.
            normalize (bool): If True, output vectors are normalized to unit length
                             (so cosine similarity equals dot product).

        Returns:
            np.ndarray: 2D array of shape (len(texts), dimension).
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Converts a single user question into a 1D vector.
        IMPORTANT: Uses the exact same model and normalization as embed_documents.

        Args:
            query (str): User question text.
            normalize (bool): If True, normalizes vector to unit length.

        Returns:
            np.ndarray: 1D array of shape (dimension,).
        """
        if not query.strip():
            raise ValueError("Query string cannot be empty.")

        embedding = self.model.encode(
            query,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        return embedding.astype(np.float32)

    def embed_chunks(
        self,
        chunks: List[Document],
        batch_size: int = 32,
        normalize: bool = True
    ) -> Tuple[np.ndarray, List[Document]]:
        """
        Embeds a list of LangChain Document chunks while preserving the strict
        1-to-1 index mapping between each vector, chunk text, and chunk metadata.

        Args:
            chunks (List[Document]): The list of chunked Document objects.
            batch_size (int): Batch size for encoding.
            normalize (bool): Whether to normalize vectors.

        Returns:
            Tuple[np.ndarray, List[Document]]:
                - embeddings matrix of shape (num_chunks, dimension)
                - original chunks in the exact matching order
        """
        texts = [chunk.page_content for chunk in chunks]
        vectors = self.embed_documents(texts, batch_size=batch_size, normalize=normalize)
        return vectors, chunks
