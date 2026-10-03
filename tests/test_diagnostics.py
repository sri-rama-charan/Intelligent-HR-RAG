"""
Unit tests for the retrieval diagnostics module (src/evaluation/diagnostics.py).
Ensures zero network calls, deterministic behavior, and proper ranking/tier evaluation.
"""

import unittest
from unittest.mock import MagicMock
import numpy as np
from langchain_core.documents import Document

from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.embeddings import EmbeddingManager
from src.evaluation.diagnostics import (
    diagnose_query_retrieval,
    find_chunk_rank,
    evaluate_rank_tier
)


class TestDiagnosticsModule(unittest.TestCase):
    """Test suite for retrieval diagnostics functions."""

    def setUp(self):
        # Create mock embedder
        self.mock_embedder = MagicMock(spec=EmbeddingManager)
        self.mock_embedder.dimension = 4
        self.mock_embedder.embed_query.return_value = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Create real FAISS vector store with mock documents
        self.vector_store = FAISSVectorStore(dimension=4)
        chunks = [
            Document(
                page_content="Earned Leave can be carried forward up to 45 days.",
                metadata={"source": "02_Leave_Policy.pdf", "page": 3, "chunk_id": "02_Leave_Policy_p3_c0"}
            ),
            Document(
                page_content="Maternity leave entitlement is 26 weeks.",
                metadata={"source": "02_Leave_Policy.pdf", "page": 3, "chunk_id": "02_Leave_Policy_p3_c1"}
            ),
            Document(
                page_content="Domestic travel daily allowance is Rs 500.",
                metadata={"source": "10_Travel_and_Expense_Policy.pdf", "page": 2, "chunk_id": "10_Travel_p2_c0"}
            ),
        ]
        # Vectors where chunk 0 has highest similarity to query [1.0, 0.0, 0.0, 0.0]
        vectors = np.array([
            [1.0, 0.0, 0.0, 0.0],  # inner product = 1.0 (rank 1)
            [0.5, 0.5, 0.5, 0.5],  # inner product = 0.5 (rank 2)
            [0.0, 1.0, 0.0, 0.0],  # inner product = 0.0 (rank 3)
        ], dtype=np.float32)

        self.vector_store.add_documents(chunks, vectors)

    def test_diagnose_query_retrieval(self):
        """Test retrieving top-k formatted diagnostics."""
        results = diagnose_query_retrieval(
            self.vector_store,
            self.mock_embedder,
            query="carry forward",
            top_k=2
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["rank"], 1)
        self.assertEqual(results[0]["chunk_id"], "02_Leave_Policy_p3_c0")
        self.assertAlmostEqual(results[0]["score"], 1.0, places=3)
        self.assertIn("Earned Leave", results[0]["preview"])

    def test_find_chunk_rank_found(self):
        """Test finding exact rank of a chunk that exists in index."""
        result = find_chunk_rank(
            self.vector_store,
            self.mock_embedder,
            query="carry forward",
            target_chunk_id="02_Leave_Policy_p3_c0"
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["rank"], 1)
        self.assertEqual(result["chunk_id"], "02_Leave_Policy_p3_c0")
        self.assertAlmostEqual(result["score"], 1.0, places=3)

    def test_find_chunk_rank_not_found(self):
        """Test finding rank for non-existent chunk ID returns None."""
        result = find_chunk_rank(
            self.vector_store,
            self.mock_embedder,
            query="carry forward",
            target_chunk_id="non_existent_chunk_id"
        )
        self.assertIsNone(result)

    def test_find_chunk_rank_empty_store(self):
        """Test finding rank on empty store returns None."""
        empty_store = FAISSVectorStore(dimension=4)
        result = find_chunk_rank(
            empty_store,
            self.mock_embedder,
            query="test",
            target_chunk_id="any"
        )
        self.assertIsNone(result)

    def test_evaluate_rank_tier(self):
        """Test rank tier categorization."""
        tier_1 = evaluate_rank_tier(1)
        self.assertEqual(tier_1, {"top_3": "Yes", "top_5": "Yes", "top_10": "Yes"})

        tier_4 = evaluate_rank_tier(4)
        self.assertEqual(tier_4, {"top_3": "No", "top_5": "Yes", "top_10": "Yes"})

        tier_8 = evaluate_rank_tier(8)
        self.assertEqual(tier_8, {"top_3": "No", "top_5": "No", "top_10": "Yes"})

        tier_15 = evaluate_rank_tier(15)
        self.assertEqual(tier_15, {"top_3": "No", "top_5": "No", "top_10": "No"})

        tier_none = evaluate_rank_tier(None)
        self.assertEqual(tier_none, {"top_3": "No", "top_5": "No", "top_10": "No"})


if __name__ == "__main__":
    unittest.main()
