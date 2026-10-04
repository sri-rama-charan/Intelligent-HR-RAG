"""
Unit tests for BM25Store lexical retrieval (src/retrieval/bm25_store.py).
Validates indexing, metadata preservation, top-k retrieval, and edge-case handling.
"""

import unittest
from pathlib import Path
from langchain_core.documents import Document

from src.ingestion.pdf_loader import load_all_pdfs
from src.ingestion.chunker import chunk_documents
from src.retrieval.bm25_store import BM25Store, default_tokenizer


class TestBM25Store(unittest.TestCase):
    """Test suite for BM25Store functionality."""

    @classmethod
    def setUpClass(cls):
        # Load the real 107 corpus chunks for comprehensive testing
        cls.project_root = Path(__file__).resolve().parent.parent
        corpus_dir = cls.project_root / "data" / "hr_corpus"
        pages = load_all_pdfs(corpus_dir)
        cls.chunks = chunk_documents(pages, chunk_size=800, chunk_overlap=100)
        cls.bm25_store = BM25Store.build_from_chunks(cls.chunks)

    def test_bm25_indexes_all_107_chunks(self):
        """Verify that BM25 indexes exactly all 107 chunks in the corpus."""
        self.assertEqual(self.bm25_store.total_documents, 107)
        self.assertEqual(len(self.bm25_store.chunks), 107)
        self.assertEqual(len(self.bm25_store.corpus_tokens), 107)

    def test_bm25_returns_requested_top_k(self):
        """Verify that search returns exactly the requested Top-K results."""
        for k in [1, 3, 5, 10]:
            results = self.bm25_store.search("Earned leave carry forward", top_k=k)
            self.assertEqual(len(results), k)

    def test_returned_chunk_ids_map_correctly_to_metadata(self):
        """Verify that returned chunk items contain valid metadata matching corpus chunks."""
        results = self.bm25_store.search("Maternity leave entitlement", top_k=3)
        self.assertTrue(len(results) > 0)
        for r in results:
            self.assertIn("chunk_id", r)
            self.assertIn("score", r)
            self.assertIn("rank", r)
            self.assertIn("text", r)
            self.assertIn("source", r)
            self.assertIn("page", r)
            self.assertTrue(r["chunk_id"].startswith("02_Leave_Policy") or r["source"].endswith(".pdf"))
            self.assertIsInstance(r["score"], float)

    def test_score_ordering_descending(self):
        """Verify that returned BM25 scores are monotonically non-increasing."""
        results = self.bm25_store.search("Annual performance review March", top_k=5)
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_empty_or_whitespace_query_returns_empty_list(self):
        """Verify that empty queries return an empty list without errors."""
        self.assertEqual(self.bm25_store.search(""), [])
        self.assertEqual(self.bm25_store.search("   "), [])

    def test_invalid_top_k_raises_error(self):
        """Verify that non-positive top_k raises ValueError."""
        with self.assertRaises(ValueError):
            self.bm25_store.search("test", top_k=0)
        with self.assertRaises(ValueError):
            self.bm25_store.search("test", top_k=-3)

    def test_tokenizer_clean_tokens(self):
        """Verify default_tokenizer cleans text and isolates words."""
        tokens = default_tokenizer("Earned Leave (EL): Up to 45 days!")
        self.assertEqual(tokens, ["earned", "leave", "el", "up", "to", "45", "days"])


if __name__ == "__main__":
    unittest.main()
