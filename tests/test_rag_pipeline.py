"""
Unit tests for Phase 2E: Unified RAG Pipeline.
Validates orchestration, input validation, Top-K delegation, source extraction,
and error propagation using deterministic mocks (zero network calls).
"""

import unittest
from unittest.mock import MagicMock
import numpy as np

from src.pipeline.rag_pipeline import RAGPipeline, RAGResponse


class TestRAGPipeline(unittest.TestCase):

    def setUp(self):
        # 1. Mock EmbeddingManager
        self.mock_embedder = MagicMock()
        self.mock_embedder.embed_query.return_value = np.zeros(384, dtype=np.float32)

        # 2. Mock FAISSVectorStore with 3 sample chunks (two from the same doc/page to test deduplication)
        self.mock_chunks = [
            {
                "rank": 1,
                "score": 0.5239,
                "text": "Salaries are credited on the 7th of the following month.",
                "source": "06_Compensation_and_Benefits_Policy.pdf",
                "page": 1,
                "chunk_id": "06_Comp_p1_c1"
            },
            {
                "rank": 2,
                "score": 0.4753,
                "text": "The payroll cut-off date is the 24th of each month.",
                "source": "06_Compensation_and_Benefits_Policy.pdf",
                "page": 1,
                "chunk_id": "06_Comp_p1_c2"
            },
            {
                "rank": 3,
                "score": 0.4120,
                "text": "Salary deductions can be made for unpaid leave.",
                "source": "06_Compensation_and_Benefits_Policy.pdf",
                "page": 2,
                "chunk_id": "06_Comp_p2_c1"
            }
        ]
        self.mock_vector_store = MagicMock()
        self.mock_vector_store.search.return_value = self.mock_chunks

        # 3. Mock GeminiGenerator
        self.mock_generator = MagicMock()
        self.mock_generator.model_name = "gemini-test-mock"
        self.mock_generator.generate.return_value = (
            "Salaries are credited to the employee bank account by the 7th of the following month."
        )

        # 4. Construct Pipeline under test
        self.pipeline = RAGPipeline(
            embedder=self.mock_embedder,
            vector_store=self.mock_vector_store,
            generator=self.mock_generator,
            top_k=3
        )

    def test_valid_question_reaches_retrieval(self):
        """Verify pipeline embeds question and queries vector store."""
        question = "When does salary get credited?"
        response = self.pipeline.ask(question)

        self.mock_embedder.embed_query.assert_called_once_with(question, normalize=True)
        self.mock_vector_store.search.assert_called_once()
        self.assertEqual(response.question, question)

    def test_retrieved_chunks_passed_to_generator(self):
        """Verify retrieved chunks from vector store are passed directly to generator."""
        question = "When does salary get credited?"
        self.pipeline.ask(question)

        self.mock_generator.generate.assert_called_once_with(question, self.mock_chunks)

    def test_final_answer_returned_in_structured_response(self):
        """Verify RAGResponse contains expected answer, model name, and question."""
        response = self.pipeline.ask("When does salary get credited?")

        self.assertIsInstance(response, RAGResponse)
        self.assertEqual(
            response.answer,
            "Salaries are credited to the employee bank account by the 7th of the following month."
        )
        self.assertEqual(response.model, "gemini-test-mock")
        self.assertEqual(len(response.retrieved_chunks), 3)

    def test_sources_extracted_correctly_with_deduplication(self):
        """Verify user-facing citations deduplicate identical (source, page) entries."""
        response = self.pipeline.ask("When does salary get credited?")

        # In mock data: chunk 1 & 2 are from (Policy.pdf, Page 1), chunk 3 is from (Policy.pdf, Page 2).
        # Total chunks = 3, but unique citations = 2.
        self.assertEqual(len(response.sources), 2)
        self.assertEqual(response.sources[0], {"source": "06_Compensation_and_Benefits_Policy.pdf", "page": 1})
        self.assertEqual(response.sources[1], {"source": "06_Compensation_and_Benefits_Policy.pdf", "page": 2})

    def test_top_k_configuration_respected(self):
        """Verify custom top_k parameter overrides default pipeline top_k."""
        self.pipeline.ask("When does salary get credited?", top_k=5)
        # Verify vector_store.search was called with k=5
        args, kwargs = self.mock_vector_store.search.call_args
        self.assertEqual(kwargs.get("top_k"), 5)

    def test_empty_question_raises_value_error(self):
        """Verify empty string and whitespace questions raise ValueError."""
        with self.assertRaises(ValueError):
            self.pipeline.ask("")
        with self.assertRaises(ValueError):
            self.pipeline.ask("     \n\t  ")

    def test_none_question_raises_value_error(self):
        """Verify passing None raises ValueError."""
        with self.assertRaises(ValueError):
            self.pipeline.ask(None)

    def test_invalid_top_k_init_raises_value_error(self):
        """Verify top_k <= 0 raises ValueError on pipeline construction."""
        with self.assertRaises(ValueError):
            RAGPipeline(self.mock_embedder, self.mock_vector_store, self.mock_generator, top_k=0)
        with self.assertRaises(ValueError):
            RAGPipeline(self.mock_embedder, self.mock_vector_store, self.mock_generator, top_k=-2)

    def test_retrieval_errors_surfaced_clearly(self):
        """Verify exceptions during retrieval raise RuntimeError."""
        self.mock_vector_store.search.side_effect = Exception("FAISS index corrupted")
        with self.assertRaises(RuntimeError) as ctx:
            self.pipeline.ask("When does salary get credited?")
        self.assertIn("Retrieval stage failed", str(ctx.exception))

    def test_generation_errors_surfaced_clearly(self):
        """Verify exceptions during generation raise RuntimeError."""
        self.mock_generator.generate.side_effect = Exception("API rate limit exceeded")
        with self.assertRaises(RuntimeError) as ctx:
            self.pipeline.ask("When does salary get credited?")
        self.assertIn("Generation stage failed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
