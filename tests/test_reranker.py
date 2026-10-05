"""
Unit tests for Phase 7: Cross-Encoder Reranker.
Validates:
  1. Component initialization and dependency injection.
  2. Query-chunk pair formation and scoring.
  3. Correct descending re-ordering by cross-encoder score.
  4. Top-K truncation (e.g. 20 candidates reduced to Top-5).
  5. Edge cases: empty candidates list, fewer candidates than K, zero/negative top_k.
  6. Comprehensive metadata preservation (chunk_id, source, page, text, prior retrieval metrics).
All tests are 100% deterministic (zero network/model download calls via mock injection).
"""

import unittest
from unittest.mock import MagicMock
import numpy as np

from src.retrieval.reranker import Reranker


class TestReranker(unittest.TestCase):

    def setUp(self):
        # 1. Create a mock CrossEncoder model
        self.mock_model = MagicMock()
        # predict returns an array of scores corresponding to candidate pairs
        self.mock_model.predict.return_value = np.array([0.15, 0.92, 0.45, 0.88, 0.05], dtype=np.float32)

        # 2. Inject mock model into Reranker
        self.reranker = Reranker(model_name="mock-reranker", model=self.mock_model)

        # 3. Create 5 sample candidate chunks with rich upstream metadata
        self.sample_candidates = [
            {
                "rank": 1,
                "score": 0.033,
                "chunk_id": "c1",
                "source": "01_Leave_Policy.pdf",
                "page": 1,
                "text": "General leave policy introduction.",
                "faiss_rank": 1,
                "bm25_rank": 4
            },
            {
                "rank": 2,
                "score": 0.031,
                "chunk_id": "c2",
                "source": "01_Leave_Policy.pdf",
                "page": 2,
                "text": "Earned leave carry-forward is capped at 45 days.",
                "faiss_rank": 5,
                "bm25_rank": 1
            },
            {
                "rank": 3,
                "score": 0.030,
                "chunk_id": "c3",
                "source": "02_Work_From_Home.pdf",
                "page": 1,
                "text": "Remote work guidelines.",
                "faiss_rank": 2,
                "bm25_rank": 8
            },
            {
                "rank": 4,
                "score": 0.029,
                "chunk_id": "c4",
                "source": "01_Leave_Policy.pdf",
                "page": 2,
                "text": "Carry-forward leave calculation rules.",
                "faiss_rank": 4,
                "bm25_rank": 3
            },
            {
                "rank": 5,
                "score": 0.025,
                "chunk_id": "c5",
                "source": "03_Travel_Policy.pdf",
                "page": 1,
                "text": "Domestic per diem details.",
                "faiss_rank": 12,
                "bm25_rank": 15
            }
        ]

    def test_dependency_injection_initialization(self):
        """Verify Reranker initializes with custom/mock model without downloading."""
        self.assertEqual(self.reranker.model_name, "mock-reranker")
        self.assertIs(self.reranker.model, self.mock_model)

    def test_pair_creation_and_scoring(self):
        """Verify query-chunk pairs are built and passed to model.predict."""
        query = "How much leave can I carry forward?"
        self.reranker.rerank(query, self.sample_candidates, top_k=3)

        expected_pairs = [
            [query, self.sample_candidates[0]["text"]],
            [query, self.sample_candidates[1]["text"]],
            [query, self.sample_candidates[2]["text"]],
            [query, self.sample_candidates[3]["text"]],
            [query, self.sample_candidates[4]["text"]]
        ]
        self.mock_model.predict.assert_called_once_with(expected_pairs)

    def test_descending_ranking_by_score(self):
        """Verify candidates are re-ordered descending by reranker score."""
        # Predicted scores: [0.15, 0.92, 0.45, 0.88, 0.05]
        # Expected order: c2 (0.92), c4 (0.88), c3 (0.45), c1 (0.15), c5 (0.05)
        query = "How much leave can I carry forward?"
        results = self.reranker.rerank(query, self.sample_candidates, top_k=5)

        self.assertEqual(len(results), 5)
        self.assertEqual(results[0]["chunk_id"], "c2")
        self.assertAlmostEqual(results[0]["rerank_score"], 0.92, places=4)
        self.assertEqual(results[0]["rank"], 1)

        self.assertEqual(results[1]["chunk_id"], "c4")
        self.assertAlmostEqual(results[1]["rerank_score"], 0.88, places=4)
        self.assertEqual(results[1]["rank"], 2)

        self.assertEqual(results[2]["chunk_id"], "c3")
        self.assertAlmostEqual(results[2]["rerank_score"], 0.45, places=4)
        self.assertEqual(results[2]["rank"], 3)

        self.assertEqual(results[3]["chunk_id"], "c1")
        self.assertAlmostEqual(results[3]["rerank_score"], 0.15, places=4)
        self.assertEqual(results[3]["rank"], 4)

        self.assertEqual(results[4]["chunk_id"], "c5")
        self.assertAlmostEqual(results[4]["rerank_score"], 0.05, places=4)
        self.assertEqual(results[4]["rank"], 5)

    def test_top_k_truncation(self):
        """Verify top_k parameter strictly limits the number of returned chunks."""
        query = "How much leave can I carry forward?"
        results = self.reranker.rerank(query, self.sample_candidates, top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chunk_id"], "c2")
        self.assertEqual(results[1]["chunk_id"], "c4")

    def test_metadata_preservation(self):
        """Verify all original chunk metadata fields and text are intact."""
        query = "How much leave can I carry forward?"
        results = self.reranker.rerank(query, self.sample_candidates, top_k=1)
        top = results[0]

        self.assertEqual(top["chunk_id"], "c2")
        self.assertEqual(top["source"], "01_Leave_Policy.pdf")
        self.assertEqual(top["page"], 2)
        self.assertEqual(top["text"], "Earned leave carry-forward is capped at 45 days.")
        self.assertEqual(top["faiss_rank"], 5)
        self.assertEqual(top["bm25_rank"], 1)
        self.assertEqual(top["score"], 0.031)  # Prior RRF score
        self.assertIn("rerank_score", top)
        self.assertEqual(top["rank"], 1)

    def test_empty_candidates_returns_empty(self):
        """Verify passing an empty candidate list returns []."""
        results = self.reranker.rerank("Query", [], top_k=5)
        self.assertEqual(results, [])
        self.mock_model.predict.assert_not_called()

    def test_fewer_candidates_than_k(self):
        """Verify when candidates < top_k, all candidates are reranked and returned."""
        short_candidates = self.sample_candidates[:2]
        self.mock_model.predict.return_value = np.array([0.2, 0.8], dtype=np.float32)

        results = self.reranker.rerank("Query", short_candidates, top_k=5)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chunk_id"], "c2")
        self.assertEqual(results[1]["chunk_id"], "c1")

    def test_invalid_top_k_returns_empty(self):
        """Verify top_k <= 0 returns []."""
        results_zero = self.reranker.rerank("Query", self.sample_candidates, top_k=0)
        self.assertEqual(results_zero, [])
        results_neg = self.reranker.rerank("Query", self.sample_candidates, top_k=-3)
        self.assertEqual(results_neg, [])

    def test_original_candidate_list_not_mutated(self):
        """Verify the input candidates list dictionaries are not mutated in place."""
        original_score = self.sample_candidates[0]["score"]
        self.reranker.rerank("Query", self.sample_candidates, top_k=3)
        self.assertEqual(self.sample_candidates[0]["score"], original_score)
        self.assertNotIn("rerank_score", self.sample_candidates[0])


if __name__ == "__main__":
    unittest.main()
