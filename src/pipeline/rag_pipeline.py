"""
Unified RAG Pipeline module for the HR Help Desk Chatbot.
Orchestrates the entire query-to-answer lifecycle by connecting:
  1. Embedding model (query vectorization)
  2. FAISS vector store (Top-K similarity retrieval)
  3. Gemini generator (grounded answer generation + citation extraction)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path

from src.ingestion.pdf_loader import load_all_pdfs
from src.ingestion.chunker import chunk_documents
from src.retrieval.embeddings import EmbeddingManager, DEFAULT_EMBEDDING_MODEL
from src.retrieval.vector_store import FAISSVectorStore
from src.generation.gemini_generator import GeminiGenerator, DEFAULT_GEMINI_MODEL


@dataclass
class RAGResponse:
    """
    Structured response object returned by RAGPipeline.ask().
    Provides clean user-facing answers and sources, while retaining
    debug/retrieval metadata.
    """
    question: str
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    model: str = ""
    top_k: int = 3


class RAGPipeline:
    """
    Orchestrates the end-to-end RAG workflow.
    Acts as the manager delegating tasks to retrieval and generation components
    without duplicating their internal logic.
    """

    def __init__(
        self,
        embedder: Optional[EmbeddingManager] = None,
        vector_store: Optional[FAISSVectorStore] = None,
        generator: Optional[GeminiGenerator] = None,
        top_k: int = 3
    ):
        """
        Initializes the unified pipeline using dependency injection.

        Args:
            embedder (Optional[EmbeddingManager]): Embedding component.
            vector_store (Optional[FAISSVectorStore]): FAISS vector store.
            generator (Optional[GeminiGenerator]): Gemini LLM generator.
            top_k (int): Number of relevant chunks to retrieve (default: 3).
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator
        self.top_k = top_k

    def ask(self, question: str, top_k: Optional[int] = None) -> RAGResponse:
        """
        Answers a user question through the complete RAG lifecycle:
          Validate question -> Embed -> FAISS Retrieve -> Ground Prompt -> Generate Answer.

        Args:
            question (str): The employee's question.
            top_k (Optional[int]): Override default Top-K if provided.

        Returns:
            RAGResponse: Structured result with answer, citations, and debug chunks.
        """
        # 1. Question Validation
        if question is None:
            raise ValueError("Question cannot be None.")
        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("Question cannot be empty or whitespace only.")

        # Ensure components are configured
        if self.embedder is None:
            raise RuntimeError("Pipeline embedder is not initialized.")
        if self.vector_store is None:
            raise RuntimeError("Pipeline vector_store is not initialized.")
        if self.generator is None:
            raise RuntimeError("Pipeline generator is not initialized.")

        k = top_k if (top_k is not None and top_k > 0) else self.top_k

        # 2. Embedding & Retrieval Stage
        try:
            query_vector = self.embedder.embed_query(cleaned_question, normalize=True)
            retrieved_chunks = self.vector_store.search(query_vector, top_k=k)
        except Exception as e:
            raise RuntimeError(f"Retrieval stage failed: {e}") from e

        # 3. Generation Stage
        try:
            answer = self.generator.generate(cleaned_question, retrieved_chunks)
        except Exception as e:
            raise RuntimeError(f"Generation stage failed: {e}") from e

        # 4. Source Citation Extraction (Deduplicated)
        unique_sources = self._extract_unique_sources(retrieved_chunks)

        # 5. Return structured result
        return RAGResponse(
            question=cleaned_question,
            answer=answer,
            sources=unique_sources,
            retrieved_chunks=retrieved_chunks,
            model=getattr(self.generator, "model_name", "unknown"),
            top_k=k
        )

    @staticmethod
    def _extract_unique_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extracts distinct (source_file, page) pairs for clean user citations.
        """
        unique_sources = []
        seen = set()

        for chunk in chunks:
            source = chunk.get("source", "Unknown Document")
            page = chunk.get("page", 0)
            key = (source, page)
            if key not in seen:
                seen.add(key)
                unique_sources.append({"source": source, "page": page})

        return unique_sources

    @classmethod
    def from_corpus(
        cls,
        corpus_dir: Optional[Path] = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        gemini_model: str = DEFAULT_GEMINI_MODEL,
        api_key: Optional[str] = None,
        top_k: int = 3
    ) -> "RAGPipeline":
        """
        Convenience factory: loads the corpus, chunks documents, builds the in-memory
        FAISS index, initializes GeminiGenerator, and returns a ready-to-use RAGPipeline.
        """
        pages = load_all_pdfs(corpus_dir)
        chunks = chunk_documents(pages, chunk_size=800, chunk_overlap=100)

        embedder = EmbeddingManager(model_name=embedding_model)
        vector_store = FAISSVectorStore.build_from_chunks(chunks, embedder)
        generator = GeminiGenerator(api_key=api_key, model_name=gemini_model)

        return cls(
            embedder=embedder,
            vector_store=vector_store,
            generator=generator,
            top_k=top_k
        )
