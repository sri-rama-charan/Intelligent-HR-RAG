"""
Unit tests for HybridRetriever and Reciprocal Rank Fusion (src/retrieval/hybrid_retriever.py).
Validates rank fusion logic, duplicate chunk deduplication, metadata preservation,
and end-to-end hybrid retrieval without external network calls.
"""

import unittest
from unittest.mock import MagicMock
import numpy as np
from langchain_core.documents import Document

from src.retrieval.embeddings import EmbeddingManager
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid_retriever import (
    HybridRetriever,
    reciprocal_rank_fusion,
    DEFAULT_RRF_K
)


class TestHybridRetriever(unittest.TestCase):
    """Test suite for HybridRetriever and RRF fusion."""

    def test_reciprocal_rank_fusion_logic(self):
        """Verify RRF formula: 1 / (60 + rank) and score addition for overlapping chunks."""
        faiss_results = [
            {"rank": 1, "score": 0.9, "chunk_id": "chunk_A", "text": "A", "source": "doc1.pdf", "page": 1},
            {"rank": 2, "score": 0.8, "chunk_id": "chunk_B", "text": "B", "source": "doc1.pdf", "page": 2},
        ]
        bm25_results = [
            {"rank": 1, "score": 12.5, "chunk_id": "chunk_B", "text": "B", "source": "doc1.pdf", "page": 2},
            {"rank": 2, "score": 8.0, "chunk_id": "chunk_C", "text": "C", "source": "doc2.pdf", "page": 1},
        ]

        # chunk_B appears in both: FAISS rank 2, BM25 rank 1
        # RRF(chunk_B) = 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = ~0.032522
        # RRF(chunk_A) = 1/(60+1) = 1/61 = ~0.016393
        # RRF(chunk_C) = 1/(60+2) = 1/62 = ~0.016129
        # So chunk_B should be Rank 1!
        fused = reciprocal_rank_fusion(faiss_results, bm25_results, rrf_k=60, top_k=3)

        self.assertEqual(len(fused), 3)
        self.assertEqual(fused[0]["chunk_id"], "chunk_B")
        self.assertEqual(fused[0]["rank"], 1)
        self.assertAlmostEqual(fused[0]["rrf_score"], (1/62) + (1/61), places=5)
        self.assertEqual(fused[0]["faiss_rank"], 2)
        self.assertEqual(fused[0]["bm25_rank"], 1)

        self.assertEqual(fused[1]["chunk_id"], "chunk_A")
        self.assertEqual(fused[1]["rank"], 2)

        self.assertEqual(fused[2]["chunk_id"], "chunk_C")
        self.assertEqual(fused[2]["rank"], 3)

    def test_duplicate_chunks_are_never_returned_twice(self):
        """Verify that a chunk present in both retrievers is merged into one entry."""
        faiss_results = [
            {"rank": 1, "score": 0.85, "chunk_id": "chunk_1", "text": "content", "source": "doc.pdf", "page": 1}
        ]
        bm25_results = [
            {"rank": 1, "score": 10.0, "chunk_id": "chunk_1", "text": "content", "source": "doc.pdf", "page": 1}
        ]
        fused = reciprocal_rank_fusion(faiss_results, bm25_results, top_k=5)
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["chunk_id"], "chunk_1")

    def test_rrf_invalid_parameters_raise_error(self):
        """Verify ValueError on non-positive top_k or rrf_k."""
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([], [], top_k=0)
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([], [], rrf_k=0)

    def test_hybrid_retriever_search_integration(self):
        """Verify HybridRetriever orchestrates mock FAISS and mock BM25 and returns Top-K."""
        mock_vs = MagicMock(spec=FAISSVectorStore)
        mock_bm25 = MagicMock(spec=BM25Store)
        mock_embedder = MagicMock(spec=EmbeddingManager)
        mock_embedder.embed_query.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        mock_vs.search.return_value = [
            {"rank": 1, "score": 0.9, "chunk_id": "c1", "text": "Text 1", "source": "s1.pdf", "page": 1},
            {"rank": 2, "score": 0.7, "chunk_id": "c2", "text": "Text 2", "source": "s1.pdf", "page": 2},
        ]
        mock_bm25.search.return_value = [
            {"rank": 1, "score": 15.0, "chunk_id": "c2", "text": "Text 2", "source": "s1.pdf", "page": 2},
            {"rank": 2, "score": 11.0, "chunk_id": "c3", "text": "Text 3", "source": "s2.pdf", "page": 1},
        ]

        retriever = HybridRetriever(
            vector_store=mock_vs,
            bm25_store=mock_bm25,
            embedder=mock_embedder,
            default_top_k=2
        )

        results = retriever.search("salary credit date", top_k=2)

        self.assertEqual(len(results), 2)
        # c2 had rank 2 in FAISS and rank 1 in BM25 -> top RRF score
        self.assertEqual(results[0]["chunk_id"], "c2")
        self.assertEqual(results[0]["rank"], 1)
        self.assertEqual(results[1]["rank"], 2)

        mock_embedder.embed_query.assert_called_once_with("salary credit date", normalize=True)
        mock_vs.search.assert_called_once()
        mock_bm25.search.assert_called_once()

    def test_hybrid_retriever_empty_query_returns_empty_list(self):
        """Verify empty query produces empty results without calling backends."""
        retriever = HybridRetriever(MagicMock(), MagicMock(), MagicMock())
        self.assertEqual(retriever.search(""), [])
        self.assertEqual(retriever.search("   "), [])


if __name__ == "__main__":
    unittest.main()
