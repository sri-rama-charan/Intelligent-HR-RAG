"""
Unit tests for Phase 2C: FAISS Vector Store and Top-K Retrieval.
Validates FAISS index creation, vector count, top-k search boundaries,
metadata mapping integrity, score ordering, and persistence.
"""

import unittest
import tempfile
from pathlib import Path
import numpy as np

from src.ingestion.pdf_loader import load_all_pdfs
from src.ingestion.chunker import chunk_documents
from src.retrieval.embeddings import EmbeddingManager, DEFAULT_EMBEDDING_MODEL
from src.retrieval.vector_store import FAISSVectorStore


class TestVectorStore(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parent.parent
        corpus_dir = cls.project_root / "data" / "hr_corpus"
        if not corpus_dir.exists():
            corpus_dir = cls.project_root / "project-2-intelligent-rag" / "zyro-dynamics-hr-corpus"

        pages = load_all_pdfs(corpus_dir)
        cls.chunks = chunk_documents(pages, chunk_size=800, chunk_overlap=100)
        cls.embedder = EmbeddingManager(model_name=DEFAULT_EMBEDDING_MODEL)
        
        # Build vector store once for testing
        cls.vector_store = FAISSVectorStore.build_from_chunks(cls.chunks, cls.embedder)

    def test_faiss_index_creation_and_count(self):
        """Verify FAISS index is created with correct total vector count (107)."""
        self.assertIsNotNone(self.vector_store.index)
        self.assertEqual(self.vector_store.total_vectors, len(self.chunks))
        self.assertEqual(self.vector_store.dimension, 384)

    def test_query_dimension_matches_index(self):
        """Verify query vector dimension matches FAISS index dimension."""
        q_vec = self.embedder.embed_query("When is salary credited?")
        self.assertEqual(q_vec.shape[0], self.vector_store.dimension)

    def test_top_k_returns_exact_count(self):
        """Verify search returns at most K results."""
        q_vec = self.embedder.embed_query("casual leaves")
        for k in [1, 3, 5, 10]:
            results = self.vector_store.search(q_vec, top_k=k)
            self.assertEqual(len(results), k, f"Expected {k} results, got {len(results)}")

    def test_results_contain_required_metadata(self):
        """Verify every returned result contains score, text, source, page, chunk_id."""
        q_vec = self.embedder.embed_query("leave rules")
        results = self.vector_store.search(q_vec, top_k=5)
        self.assertTrue(len(results) > 0)

        required_keys = ["rank", "score", "text", "source", "page", "chunk_id"]
        for item in results:
            for key in required_keys:
                self.assertIn(key, item, f"Missing key '{key}' in search result")
            self.assertTrue(len(item["text"].strip()) > 0)
            self.assertTrue(item["source"].endswith(".pdf"))
            self.assertIsInstance(item["page"], int)
            self.assertIsInstance(item["score"], float)

    def test_returned_chunk_ids_exist_in_corpus(self):
        """Verify that returned chunk_ids belong to the original corpus chunks."""
        known_chunk_ids = {c.metadata["chunk_id"] for c in self.chunks}
        q_vec = self.embedder.embed_query("work from home policy")
        results = self.vector_store.search(q_vec, top_k=5)

        for item in results:
            self.assertIn(item["chunk_id"], known_chunk_ids)

    def test_similarity_scores_in_descending_order(self):
        """Verify that similarity scores are sorted from highest to lowest."""
        q_vec = self.embedder.embed_query("IT security password")
        results = self.vector_store.search(q_vec, top_k=5)

        scores = [item["score"] for item in results]
        self.assertEqual(scores, sorted(scores, reverse=True), "Scores are not sorted in descending order.")

    def test_search_across_different_queries(self):
        """Verify search produces distinct top results for different query topics."""
        q_leave = self.embedder.embed_query("sick leave medical certificate")
        q_travel = self.embedder.embed_query("travel daily allowance hotel expenses")

        res_leave = self.vector_store.search(q_leave, top_k=1)
        res_travel = self.vector_store.search(q_travel, top_k=1)

        self.assertNotEqual(res_leave[0]["chunk_id"], res_travel[0]["chunk_id"])
        self.assertEqual(res_leave[0]["source"], "02_Leave_Policy.pdf")
        self.assertEqual(res_travel[0]["source"], "10_Travel_and_Expense_Policy.pdf")

    def test_invalid_top_k_raises_error(self):
        """Verify that non-positive top_k values raise ValueError."""
        q_vec = self.embedder.embed_query("salary")
        with self.assertRaises(ValueError):
            self.vector_store.search(q_vec, top_k=0)
        with self.assertRaises(ValueError):
            self.vector_store.search(q_vec, top_k=-5)

    def test_save_and_load_round_trip(self):
        """Verify FAISS vector store can be saved to disk and reloaded with identical behavior."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.vector_store.save(tmp_dir)
            reloaded_store = FAISSVectorStore.load(tmp_dir)

            self.assertEqual(reloaded_store.total_vectors, self.vector_store.total_vectors)
            self.assertEqual(reloaded_store.dimension, self.vector_store.dimension)

            q_vec = self.embedder.embed_query("maternity leave")
            res_orig = self.vector_store.search(q_vec, top_k=3)
            res_loaded = reloaded_store.search(q_vec, top_k=3)

            for r_orig, r_loaded in zip(res_orig, res_loaded):
                self.assertEqual(r_orig["chunk_id"], r_loaded["chunk_id"])
                self.assertAlmostEqual(r_orig["score"], r_loaded["score"], places=5)


if __name__ == "__main__":
    unittest.main()
