"""
Unified RAG Pipeline module for the HR Help Desk Chatbot.
Orchestrates the entire query-to-answer lifecycle by connecting:
  1. Embedding model (query vectorization)
  2. FAISS vector store (Top-K similarity retrieval)
  3. BM25 store + Hybrid retriever (sparse-dense fusion via RRF)
  4. Cross-Encoder reranker (deep relevance scoring of candidate pools)
  5. Gemini generator (grounded answer generation + citation extraction)
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
    top_k: int = 5
    retrieval_mode: str = "faiss"


class RAGPipeline:
    """
    Orchestrates the end-to-end RAG workflow.
    Acts as the manager delegating tasks to retrieval, reranking, and generation components
    without duplicating their internal logic.
    """

    def __init__(
        self,
        embedder: Optional[EmbeddingManager] = None,
        vector_store: Optional[FAISSVectorStore] = None,
        generator: Optional[GeminiGenerator] = None,
        retriever: Optional[Any] = None,
        reranker: Optional[Any] = None,
        retrieval_mode: str = "faiss",
        top_k: int = 5,
        candidate_pool_size: int = 20
    ):
        """
        Initializes the unified pipeline using dependency injection.

        Args:
            embedder (Optional[EmbeddingManager]): Embedding component.
            vector_store (Optional[FAISSVectorStore]): FAISS vector store.
            generator (Optional[GeminiGenerator]): Gemini LLM generator.
            retriever (Optional[Any]): Custom or hybrid retriever instance (e.g. HybridRetriever).
            reranker (Optional[Any]): Cross-encoder reranker instance (e.g. Reranker).
            retrieval_mode (str): Retrieval mode: 'faiss' (default), 'hybrid', or 'hybrid_rerank'.
            top_k (int): Number of final relevant chunks to provide to the generator (default: 5).
            candidate_pool_size (int): Size of the candidate pool fetched prior to reranking (default: 20).
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")
        if candidate_pool_size <= 0:
            raise ValueError(f"candidate_pool_size must be a positive integer, got {candidate_pool_size}")
        if retrieval_mode not in ("faiss", "hybrid", "hybrid_rerank"):
            raise ValueError(
                f"Unsupported retrieval_mode '{retrieval_mode}'. Must be 'faiss', 'hybrid', or 'hybrid_rerank'."
            )

        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator
        self.retriever = retriever
        self.reranker = reranker
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k
        self.candidate_pool_size = candidate_pool_size

    def ask(self, question: str, top_k: Optional[int] = None) -> RAGResponse:
        """
        Answers a user question through the complete RAG lifecycle:
          Validate question -> Retrieve (FAISS, Hybrid, or Hybrid+Rerank) -> Ground Prompt -> Generate Answer.

        Args:
            question (str): The employee's question.
            top_k (Optional[int]): Override default Top-K if provided.

        Returns:
            RAGResponse: Structured result with answer, citations, debug chunks, and retrieval_mode.
        """
        # 1. Question Validation
        if question is None:
            raise ValueError("Question cannot be None.")
        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("Question cannot be empty or whitespace only.")

        # Ensure generator is configured
        if self.generator is None:
            raise RuntimeError("Pipeline generator is not initialized.")

        k = top_k if (top_k is not None and top_k > 0) else self.top_k

        # 2. Retrieval Stage (FAISS, Hybrid, or Hybrid + Reranker)
        try:
            if self.retrieval_mode == "hybrid_rerank":
                if self.retriever is None:
                    raise RuntimeError("Pipeline retriever is not initialized for hybrid_rerank mode.")
                if self.reranker is None:
                    raise RuntimeError("Pipeline reranker is not initialized for hybrid_rerank mode.")
                # Fetch candidate pool (e.g. Top-20 from FAISS+BM25 via RRF)
                pool_size = max(self.candidate_pool_size, k)
                candidates = self.retriever.search(cleaned_question, top_k=pool_size)
                # Rerank candidates down to the final Top-K
                retrieved_chunks = self.reranker.rerank(cleaned_question, candidates, top_k=k)
            elif self.retrieval_mode == "hybrid":
                if self.retriever is None:
                    raise RuntimeError("Pipeline retriever is not initialized for hybrid mode.")
                retrieved_chunks = self.retriever.search(cleaned_question, top_k=k)
            else:  # "faiss" mode
                if self.embedder is None:
                    raise RuntimeError("Pipeline embedder is not initialized.")
                if self.vector_store is None:
                    raise RuntimeError("Pipeline vector_store is not initialized.")
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
            top_k=k,
            retrieval_mode=self.retrieval_mode
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
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        api_key: Optional[str] = None,
        retrieval_mode: str = "faiss",
        top_k: int = 5,
        candidate_pool_size: int = 20
    ) -> "RAGPipeline":
        """
        Convenience factory: loads the corpus, chunks documents, builds the in-memory
        retrieval backend (FAISS, Hybrid, or Hybrid+Rerank), initializes GeminiGenerator,
        and returns a ready-to-use RAGPipeline.
        """
        if retrieval_mode not in ("faiss", "hybrid", "hybrid_rerank"):
            raise ValueError(
                f"Unsupported retrieval_mode '{retrieval_mode}'. Must be 'faiss', 'hybrid', or 'hybrid_rerank'."
            )

        pages = load_all_pdfs(corpus_dir)
        chunks = chunk_documents(pages, chunk_size=800, chunk_overlap=100)

        embedder = EmbeddingManager(model_name=embedding_model)
        vector_store = FAISSVectorStore.build_from_chunks(chunks, embedder)
        generator = GeminiGenerator(api_key=api_key, model_name=gemini_model)

        retriever = None
        reranker = None

        if retrieval_mode in ("hybrid", "hybrid_rerank"):
            from src.retrieval.bm25_store import BM25Store
            from src.retrieval.hybrid_retriever import HybridRetriever
            bm25_store = BM25Store.build_from_chunks(chunks)
            default_k = candidate_pool_size if retrieval_mode == "hybrid_rerank" else top_k
            retriever = HybridRetriever(
                vector_store=vector_store,
                bm25_store=bm25_store,
                embedder=embedder,
                rrf_k=60,
                default_top_k=default_k
            )

        if retrieval_mode == "hybrid_rerank":
            from src.retrieval.reranker import Reranker
            reranker = Reranker(model_name=reranker_model)

        return cls(
            embedder=embedder,
            vector_store=vector_store,
            generator=generator,
            retriever=retriever,
            reranker=reranker,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size
        )
