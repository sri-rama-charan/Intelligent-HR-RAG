"""
Unit tests for Phase 2A: Document Ingestion and Chunking.
Validates PDF loading, page extraction, chunk generation, and metadata integrity.
"""

import unittest
from pathlib import Path
from src.ingestion.pdf_loader import load_all_pdfs, load_single_pdf, extract_title_from_filename
from src.ingestion.chunker import chunk_documents, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


class TestIngestion(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Locate project root and corpus directory
        cls.project_root = Path(__file__).resolve().parent.parent
        cls.corpus_dir = cls.project_root / "data" / "hr_corpus"
        
        # Load pages and create chunks once for tests
        cls.pages = load_all_pdfs(cls.corpus_dir)
        cls.chunks = chunk_documents(cls.pages, chunk_size=800, chunk_overlap=100)

    def test_all_11_pdfs_loaded(self):
        """Verify that pages from all 11 PDFs were loaded."""
        distinct_sources = {p.metadata["source"] for p in self.pages}
        self.assertEqual(len(distinct_sources), 11, f"Expected 11 distinct PDF sources, got {len(distinct_sources)}")

    def test_no_empty_pages(self):
        """Verify that every loaded page document has non-empty text content."""
        self.assertGreater(len(self.pages), 0, "No pages loaded.")
        for doc in self.pages:
            self.assertTrue(len(doc.page_content.strip()) > 0, f"Page in {doc.metadata.get('source')} has empty content.")

    def test_page_metadata_present(self):
        """Verify that every page has source, page number, and document_title."""
        for doc in self.pages:
            self.assertIn("source", doc.metadata)
            self.assertIn("page", doc.metadata)
            self.assertIn("document_title", doc.metadata)
            self.assertIsInstance(doc.metadata["page"], int)
            self.assertGreater(doc.metadata["page"], 0)

    def test_chunks_generated(self):
        """Verify that chunking produces more chunks than the number of raw pages."""
        self.assertGreater(len(self.chunks), len(self.pages), "Expected more chunks than original pages.")

    def test_chunk_content_non_empty(self):
        """Verify that every chunk has non-empty text."""
        for c in self.chunks:
            self.assertTrue(len(c.page_content.strip()) > 0, f"Chunk {c.metadata.get('chunk_id')} is empty.")

    def test_chunk_metadata_completeness(self):
        """Verify that every chunk preserves source, page, and chunk_id."""
        for c in self.chunks:
            self.assertIn("source", c.metadata)
            self.assertIn("page", c.metadata)
            self.assertIn("chunk_id", c.metadata)
            self.assertIn("chunk_size", c.metadata)
            self.assertTrue(c.metadata["source"].endswith(".pdf"))
            self.assertIsInstance(c.metadata["page"], int)

    def test_unique_chunk_ids(self):
        """Verify that all chunk IDs across the entire corpus are strictly unique."""
        chunk_ids = [c.metadata["chunk_id"] for c in self.chunks]
        unique_ids = set(chunk_ids)
        self.assertEqual(len(chunk_ids), len(unique_ids), f"Duplicate chunk_ids found! {len(chunk_ids)} vs {len(unique_ids)}")

    def test_chunk_size_respects_bounds(self):
        """Verify that chunk sizes stay reasonably within the configured chunk size."""
        configured_max = 800
        # Recursive text splitter usually keeps chunks <= configured_max unless an unbreakable word exists
        # Allow a small 10% margin for boundary edge cases if any
        for c in self.chunks:
            self.assertLessEqual(
                len(c.page_content),
                configured_max + 100,
                f"Chunk {c.metadata['chunk_id']} exceeded maximum size: {len(c.page_content)}"
            )

    def test_extract_title_from_filename(self):
        """Verify helper derives clean title from filenames."""
        self.assertEqual(extract_title_from_filename("02_Leave_Policy.pdf"), "Leave Policy")
        self.assertEqual(extract_title_from_filename("00_Company_Profile.pdf"), "Company Profile")
        self.assertEqual(extract_title_from_filename("10_Travel_and_Expense_Policy.pdf"), "Travel and Expense Policy")


if __name__ == "__main__":
    unittest.main()
