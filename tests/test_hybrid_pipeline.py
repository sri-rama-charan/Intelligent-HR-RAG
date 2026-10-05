"""
Unit tests for Phase 6: Hybrid Retrieval Integration into RAGPipeline.
Validates:
  1. Default FAISS mode works identically to baseline.
  2. Hybrid mode queries HybridRetriever with raw query string and top_k.
  3. Response contains retrieval_mode metadata ('faiss' vs 'hybrid').
  4. Deduplication of source citations across hybrid retrieved chunks.
  5. Dependency injection of custom/mock retrievers without coupling to BM25.
  6. Factory method (from_corpus) wiring for both modes.
  7. Error handling for missing components and invalid retrieval modes.
All tests are 100% deterministic (zero network/Gemini calls).
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.pipeline.rag_pipeline import RAGPipeline, RAGResponse


class TestHybridPipeline(unittest.TestCase):

    def setUp(self):
        # 1. Mock Embedder
        self.mock_embedder = MagicMock()
        self.mock_embedder.embed_query.return_value = np.zeros(384, dtype=np.float32)

        # 2. Mock FAISS Vector Store
        self.mock_faiss_chunks = [
            {
                "rank": 1,
                "score": 0.88,
                "text": "FAISS chunk 1: Leave policy details.",
                "source": "01_Leave_Policy.pdf",
                "page": 1,
                "chunk_id": "01_Leave_p1_c1"
            },
            {
                "rank": 2,
                "score": 0.75,
                "text": "FAISS chunk 2: Leave policy continuation.",
                "source": "01_Leave_Policy.pdf",
                "page": 1,
                "chunk_id": "01_Leave_p1_c2"
            }
        ]
        self.mock_vector_store = MagicMock()
        self.mock_vector_store.search.return_value = self.mock_faiss_chunks

        # 3. Mock Hybrid Retriever
        self.mock_hybrid_chunks = [
            {
                "rank": 1,
                "score": 0.0328,
                "text": "Hybrid chunk 1: WFH policy eligibility.",
                "source": "04_Work_From_Home_Policy.pdf",
                "page": 2,
                "chunk_id": "04_WFH_p2_c1",
                "faiss_rank": 3,
                "bm25_rank": 1
            },
            {
                "rank": 2,
                "score": 0.0164,
                "text": "Hybrid chunk 2: WFH equipment guidelines.",
                "source": "04_Work_From_Home_Policy.pdf",
                "page": 2,
                "chunk_id": "04_WFH_p2_c2",
                "faiss_rank": 5,
                "bm25_rank": 2
            },
            {
                "rank": 3,
                "score": 0.0161,
                "text": "Hybrid chunk 3: General code of conduct.",
                "source": "03_Code_of_Conduct.pdf",
                "page": 1,
                "chunk_id": "03_Conduct_p1_c1",
                "faiss_rank": None,
                "bm25_rank": 3
            }
        ]
        self.mock_hybrid_retriever = MagicMock()
        self.mock_hybrid_retriever.search.return_value = self.mock_hybrid_chunks

        # 4. Mock Gemini Generator
        self.mock_generator = MagicMock()
        self.mock_generator.model_name = "gemini-2.5-flash"
        self.mock_generator.generate.return_value = "Employees may work from home up to 2 days per week."

    # --- A. FAISS Mode Tests ---

    def test_default_retrieval_mode_is_faiss(self):
        """Pipeline constructed without retrieval_mode should default to 'faiss'."""
        pipeline = RAGPipeline(
            embedder=self.mock_embedder,
            vector_store=self.mock_vector_store,
            generator=self.mock_generator
        )
        self.assertEqual(pipeline.retrieval_mode, "faiss")
        response = pipeline.ask("What is the leave policy?")
        self.assertEqual(response.retrieval_mode, "faiss")
        self.mock_embedder.embed_query.assert_called_once()
        self.mock_vector_store.search.assert_called_once()

    def test_faiss_mode_backward_compatibility(self):
        """FAISS mode retains exact structured response and deduplication."""
        pipeline = RAGPipeline(
            embedder=self.mock_embedder,
            vector_store=self.mock_vector_store,
            generator=self.mock_generator,
            retrieval_mode="faiss",
            top_k=5
        )
        response = pipeline.ask("What is the leave policy?")
        self.assertIsInstance(response, RAGResponse)
        self.assertEqual(response.retrieval_mode, "faiss")
        self.assertEqual(len(response.retrieved_chunks), 2)
        # Deduplicated: both mock chunks are from (01_Leave_Policy.pdf, page 1)
        self.assertEqual(len(response.sources), 1)
        self.assertEqual(response.sources[0], {"source": "01_Leave_Policy.pdf", "page": 1})

    # --- B. Hybrid Mode Tests ---

    def test_hybrid_mode_uses_hybrid_retriever(self):
        """In hybrid mode, search delegates directly to retriever without pipeline embedding."""
        pipeline = RAGPipeline(
            generator=self.mock_generator,
            retriever=self.mock_hybrid_retriever,
            retrieval_mode="hybrid",
            top_k=5
        )
        question = "Can I work from home?"
        response = pipeline.ask(question)

        # In hybrid mode, pipeline delegates query string directly to retriever.search
        self.mock_hybrid_retriever.search.assert_called_once_with(question, top_k=5)
        self.mock_embedder.embed_query.assert_not_called()
        self.assertEqual(response.retrieval_mode, "hybrid")
        self.assertEqual(response.retrieved_chunks, self.mock_hybrid_chunks)

    def test_hybrid_mode_respects_top_k(self):
        """Custom top_k override is passed correctly to retriever in hybrid mode."""
        pipeline = RAGPipeline(
            generator=self.mock_generator,
            retriever=self.mock_hybrid_retriever,
            retrieval_mode="hybrid",
            top_k=3
        )
        pipeline.ask("Can I work from home?", top_k=10)
        self.mock_hybrid_retriever.search.assert_called_once_with("Can I work from home?", top_k=10)

    def test_hybrid_mode_chunks_passed_to_generator(self):
        """Generator receives the hybrid retrieved chunks in order."""
        pipeline = RAGPipeline(
            generator=self.mock_generator,
            retriever=self.mock_hybrid_retriever,
            retrieval_mode="hybrid",
            top_k=3
        )
        question = "Can I work from home?"
        pipeline.ask(question)
        self.mock_generator.generate.assert_called_once_with(question, self.mock_hybrid_chunks)

    def test_hybrid_mode_sources_deduplicated(self):
        """Sources from hybrid chunks are deduplicated by (source, page)."""
        pipeline = RAGPipeline(
            generator=self.mock_generator,
            retriever=self.mock_hybrid_retriever,
            retrieval_mode="hybrid",
            top_k=5
        )
        response = pipeline.ask("Can I work from home?")
        # 3 chunks: chunk 1 & 2 are (04_Work_From_Home_Policy.pdf, page 2); chunk 3 is (03_Code_of_Conduct.pdf, page 1)
        self.assertEqual(len(response.sources), 2)
        self.assertEqual(response.sources[0], {"source": "04_Work_From_Home_Policy.pdf", "page": 2})
        self.assertEqual(response.sources[1], {"source": "03_Code_of_Conduct.pdf", "page": 1})

    def test_hybrid_mode_reports_metadata(self):
        """RAGResponse exposes retrieval_mode='hybrid'."""
        pipeline = RAGPipeline(
            generator=self.mock_generator,
            retriever=self.mock_hybrid_retriever,
            retrieval_mode="hybrid"
        )
        response = pipeline.ask("Can I work from home?")
        self.assertEqual(response.retrieval_mode, "hybrid")
        self.assertEqual(response.model, "gemini-2.5-flash")

    # --- C. Dependency Injection & Error Handling ---

    def test_custom_retriever_dependency_injection(self):
        """Any custom retriever object implementing search(query, top_k) can be injected."""
        class CustomRetriever:
            def search(self, query: str, top_k: int = 5):
                return [{"rank": 1, "score": 1.0, "text": "custom", "source": "custom.pdf", "page": 1}]

        custom = CustomRetriever()
        pipeline = RAGPipeline(
            generator=self.mock_generator,
            retriever=custom,
            retrieval_mode="hybrid"
        )
        response = pipeline.ask("Custom question")
        self.assertEqual(response.retrieval_mode, "hybrid")
        self.assertEqual(len(response.retrieved_chunks), 1)
        self.assertEqual(response.retrieved_chunks[0]["source"], "custom.pdf")

    def test_hybrid_mode_missing_retriever_raises_error(self):
        """Hybrid mode with retriever=None raises RuntimeError when ask() is called."""
        pipeline = RAGPipeline(
            generator=self.mock_generator,
            retriever=None,
            retrieval_mode="hybrid"
        )
        with self.assertRaises(RuntimeError) as ctx:
            pipeline.ask("Question")
        self.assertIn("retriever is not initialized for hybrid mode", str(ctx.exception))

    def test_faiss_mode_missing_components_raises_error(self):
        """FAISS mode with missing embedder or vector_store raises RuntimeError."""
        pipeline_no_store = RAGPipeline(
            embedder=self.mock_embedder,
            vector_store=None,
            generator=self.mock_generator,
            retrieval_mode="faiss"
        )
        with self.assertRaises(RuntimeError) as ctx:
            pipeline_no_store.ask("Question")
        self.assertIn("vector_store is not initialized", str(ctx.exception))

        pipeline_no_embedder = RAGPipeline(
            embedder=None,
            vector_store=self.mock_vector_store,
            generator=self.mock_generator,
            retrieval_mode="faiss"
        )
        with self.assertRaises(RuntimeError) as ctx:
            pipeline_no_embedder.ask("Question")
        self.assertIn("embedder is not initialized", str(ctx.exception))

    def test_invalid_retrieval_mode_raises_value_error(self):
        """Unsupported retrieval_mode raises ValueError on construction."""
        with self.assertRaises(ValueError) as ctx:
            RAGPipeline(
                generator=self.mock_generator,
                retrieval_mode="bm25_only"
            )
        self.assertIn("Unsupported retrieval_mode", str(ctx.exception))

    # --- D. Factory Method Tests (from_corpus) ---

    @patch("src.pipeline.rag_pipeline.load_all_pdfs")
    @patch("src.pipeline.rag_pipeline.chunk_documents")
    @patch("src.pipeline.rag_pipeline.EmbeddingManager")
    @patch("src.pipeline.rag_pipeline.FAISSVectorStore")
    @patch("src.pipeline.rag_pipeline.GeminiGenerator")
    def test_from_corpus_faiss_mode(self, mock_gen_cls, mock_vs_cls, mock_emb_cls, mock_chunk, mock_load):
        """from_corpus with retrieval_mode='faiss' builds FAISS pipeline without BM25."""
        mock_load.return_value = [{"text": "page1"}]
        mock_chunk.return_value = [{"chunk_id": "c1", "text": "chunk1"}]

        pipeline = RAGPipeline.from_corpus(
            corpus_dir=None,
            retrieval_mode="faiss",
            top_k=5
        )

        self.assertEqual(pipeline.retrieval_mode, "faiss")
        self.assertIsNone(pipeline.retriever)
        mock_emb_cls.assert_called_once()
        mock_vs_cls.build_from_chunks.assert_called_once()

    @patch("src.retrieval.hybrid_retriever.HybridRetriever")
    @patch("src.retrieval.bm25_store.BM25Store")
    @patch("src.pipeline.rag_pipeline.load_all_pdfs")
    @patch("src.pipeline.rag_pipeline.chunk_documents")
    @patch("src.pipeline.rag_pipeline.EmbeddingManager")
    @patch("src.pipeline.rag_pipeline.FAISSVectorStore")
    @patch("src.pipeline.rag_pipeline.GeminiGenerator")
    def test_from_corpus_hybrid_mode(self, mock_gen_cls, mock_vs_cls, mock_emb_cls, mock_chunk, mock_load, mock_bm25_cls, mock_hybrid_cls):
        """from_corpus with retrieval_mode='hybrid' builds both stores and wires HybridRetriever."""
        mock_load.return_value = [{"text": "page1"}]
        sample_chunks = [{"chunk_id": "c1", "text": "chunk1"}]
        mock_chunk.return_value = sample_chunks

        pipeline = RAGPipeline.from_corpus(
            corpus_dir=None,
            retrieval_mode="hybrid",
            top_k=5
        )

        self.assertEqual(pipeline.retrieval_mode, "hybrid")
        self.assertIsNotNone(pipeline.retriever)
        # Verify both stores received the exact same chunks
        mock_vs_cls.build_from_chunks.assert_called_once_with(sample_chunks, mock_emb_cls.return_value)
        mock_bm25_cls.build_from_chunks.assert_called_once_with(sample_chunks)
        mock_hybrid_cls.assert_called_once_with(
            vector_store=mock_vs_cls.build_from_chunks.return_value,
            bm25_store=mock_bm25_cls.build_from_chunks.return_value,
            embedder=mock_emb_cls.return_value,
            rrf_k=60,
            default_top_k=5
        )

    # --- E. Hybrid + Reranker Mode Tests ---

    def test_hybrid_rerank_mode_fetches_pool_and_reduces_to_top_k(self):
        """In hybrid_rerank mode, retriever gets candidate_pool_size (20), and reranker reduces to top_k (5)."""
        mock_reranker = MagicMock()
        reranked_chunks = self.mock_hybrid_chunks[:2]
        mock_reranker.rerank.return_value = reranked_chunks

        pipeline = RAGPipeline(
            generator=self.mock_generator,
            retriever=self.mock_hybrid_retriever,
            reranker=mock_reranker,
            retrieval_mode="hybrid_rerank",
            top_k=2,
            candidate_pool_size=20
        )

        question = "Can I work from home?"
        response = pipeline.ask(question)

        # 1. Retriever fetched candidate pool of size 20
        self.mock_hybrid_retriever.search.assert_called_once_with(question, top_k=20)
        # 2. Reranker received the candidates and reduced to top_k=2
        mock_reranker.rerank.assert_called_once_with(question, self.mock_hybrid_chunks, top_k=2)
        # 3. Generator received only the reranked chunks (2, not 20)
        self.mock_generator.generate.assert_called_once_with(question, reranked_chunks)
        # 4. Response metadata reflects hybrid_rerank
        self.assertEqual(response.retrieval_mode, "hybrid_rerank")
        self.assertEqual(len(response.retrieved_chunks), 2)

    def test_hybrid_rerank_mode_missing_reranker_raises_error(self):
        """hybrid_rerank mode with reranker=None raises RuntimeError."""
        pipeline = RAGPipeline(
            generator=self.mock_generator,
            retriever=self.mock_hybrid_retriever,
            reranker=None,
            retrieval_mode="hybrid_rerank"
        )
        with self.assertRaises(RuntimeError) as ctx:
            pipeline.ask("Question")
        self.assertIn("reranker is not initialized for hybrid_rerank mode", str(ctx.exception))

    @patch("src.retrieval.reranker.Reranker")
    @patch("src.retrieval.hybrid_retriever.HybridRetriever")
    @patch("src.retrieval.bm25_store.BM25Store")
    @patch("src.pipeline.rag_pipeline.load_all_pdfs")
    @patch("src.pipeline.rag_pipeline.chunk_documents")
    @patch("src.pipeline.rag_pipeline.EmbeddingManager")
    @patch("src.pipeline.rag_pipeline.FAISSVectorStore")
    @patch("src.pipeline.rag_pipeline.GeminiGenerator")
    def test_from_corpus_hybrid_rerank_mode(
        self, mock_gen_cls, mock_vs_cls, mock_emb_cls, mock_chunk, mock_load, mock_bm25_cls, mock_hybrid_cls, mock_rerank_cls
    ):
        """from_corpus with retrieval_mode='hybrid_rerank' wires up retriever and reranker."""
        mock_load.return_value = [{"text": "page1"}]
        sample_chunks = [{"chunk_id": "c1", "text": "chunk1"}]
        mock_chunk.return_value = sample_chunks

        pipeline = RAGPipeline.from_corpus(
            corpus_dir=None,
            retrieval_mode="hybrid_rerank",
            top_k=5,
            candidate_pool_size=20
        )

        self.assertEqual(pipeline.retrieval_mode, "hybrid_rerank")
        self.assertIsNotNone(pipeline.retriever)
        self.assertIsNotNone(pipeline.reranker)
        mock_rerank_cls.assert_called_once_with(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")


if __name__ == "__main__":
    unittest.main()
