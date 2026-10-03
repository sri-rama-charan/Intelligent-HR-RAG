"""
Unit tests for Phase 2B: Embeddings and Vector Representation.
Validates model initialization, vector dimension consistency, query embedding,
and chunk-to-vector mapping.
"""

import unittest
from pathlib import Path
import numpy as np

from src.ingestion.pdf_loader import load_all_pdfs
from src.ingestion.chunker import chunk_documents
from src.retrieval.embeddings import EmbeddingManager, DEFAULT_EMBEDDING_MODEL


class TestEmbeddings(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Locate project root and load a small subset or full corpus
        cls.project_root = Path(__file__).resolve().parent.parent
        corpus_dir = cls.project_root / "data" / "hr_corpus"
        
        pages = load_all_pdfs(corpus_dir)
        cls.chunks = chunk_documents(pages, chunk_size=800, chunk_overlap=100)
        
        # Initialize embedding manager once for the test suite
        cls.embedder = EmbeddingManager(model_name=DEFAULT_EMBEDDING_MODEL)
        cls.vectors, cls.mapped_chunks = cls.embedder.embed_chunks(cls.chunks)

    def test_model_loads_and_has_valid_dimension(self):
        """Verify model loads and has positive integer dimension (384 for MiniLM)."""
        self.assertIsNotNone(self.embedder.model)
        self.assertEqual(self.embedder.dimension, 384)

    def test_expected_number_of_embeddings_produced(self):
        """Verify that every chunk has a corresponding vector."""
        self.assertEqual(len(self.vectors), len(self.chunks))
        self.assertEqual(self.vectors.shape[0], len(self.chunks))

    def test_every_embedding_has_same_dimension(self):
        """Verify every row in the vector matrix has exactly 384 dimensions."""
        self.assertEqual(self.vectors.shape[1], 384)
        for vec in self.vectors:
            self.assertEqual(len(vec), 384)

    def test_query_embedding_works_and_matches_dimension(self):
        """Verify query embedding produces a 1D vector of matching dimension."""
        query = "How many casual leaves can an employee take?"
        q_vec = self.embedder.embed_query(query)
        self.assertEqual(q_vec.ndim, 1)
        self.assertEqual(len(q_vec), self.embedder.dimension)
        self.assertEqual(q_vec.dtype, np.float32)

    def test_chunk_to_vector_mapping_intact(self):
        """Verify 1-to-1 index mapping between vectors and chunk metadata."""
        self.assertEqual(len(self.mapped_chunks), len(self.vectors))
        for i, chunk in enumerate(self.mapped_chunks):
            self.assertIn("chunk_id", chunk.metadata)
            self.assertIn("source", chunk.metadata)
            self.assertIn("page", chunk.metadata)
            # Verify the chunk matches the input chunk at the same index
            self.assertEqual(chunk.metadata["chunk_id"], self.chunks[i].metadata["chunk_id"])

    def test_empty_query_raises_error(self):
        """Verify that an empty or whitespace-only query raises ValueError."""
        with self.assertRaises(ValueError):
            self.embedder.embed_query("   ")


if __name__ == "__main__":
    unittest.main()
