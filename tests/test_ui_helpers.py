"""
Unit tests for the UI helper functions (src/ui/helpers.py).
Tests citation formatting, grouping, score display, and mode mapping without Streamlit or Gemini calls.
"""

import unittest
from src.ui.helpers import (
    get_mode_key,
    get_mode_label,
    format_source_citation,
    format_chunk_score,
    clean_document_title,
    format_grouped_source,
    group_and_format_sources,
    format_compact_chunk_score,
    MODE_OPTIONS,
    DEFAULT_MODE,
)


class TestUIHelpers(unittest.TestCase):
    def test_mode_mappings(self):
        self.assertEqual(get_mode_key("Hybrid + Reranker"), "hybrid_rerank")
        self.assertEqual(get_mode_key("Hybrid"), "hybrid")
        self.assertEqual(get_mode_key("FAISS"), "faiss")
        self.assertEqual(get_mode_key("Unknown Mode"), "hybrid_rerank")

        self.assertEqual(get_mode_label("hybrid_rerank"), "Hybrid + Reranker")
        self.assertEqual(get_mode_label("hybrid"), "Hybrid")
        self.assertEqual(get_mode_label("faiss"), "FAISS")
        self.assertEqual(get_mode_label("unknown"), "Hybrid + Reranker")

        self.assertIn(DEFAULT_MODE, MODE_OPTIONS)
        self.assertEqual(len(MODE_OPTIONS), 3)

    def test_clean_document_title(self):
        # Known mapping
        self.assertEqual(clean_document_title("02_Leave_Policy.pdf"), "Leave Policy")
        self.assertEqual(clean_document_title("06_Compensation_and_Benefits_Policy.pdf"), "Compensation & Benefits Policy")
        self.assertEqual(clean_document_title("10_Travel_and_Expense_Policy.pdf"), "Travel & Expense Policy")
        # Generic cleaning fallback
        self.assertEqual(clean_document_title("15_Remote_Work_Guidelines.pdf"), "Remote Work Guidelines")
        # Empty/None
        self.assertEqual(clean_document_title(""), "Unknown Document")
        self.assertEqual(clean_document_title(None), "Unknown Document")

    def test_format_grouped_source(self):
        # Multiple pages
        res_multi = format_grouped_source("Leave Policy", [3, 2, 3])
        self.assertEqual(res_multi, "📄 Leave Policy — pp. 2, 3")

        # Single page
        res_single = format_grouped_source("Leave Policy", [2])
        self.assertEqual(res_single, "📄 Leave Policy — p. 2")

        # No page or 0
        res_none = format_grouped_source("Leave Policy", [0, None])
        self.assertEqual(res_none, "📄 Leave Policy")

    def test_group_and_format_sources(self):
        raw_sources = [
            {"source": "06_Compensation_and_Benefits_Policy.pdf", "page": 1},
            {"source": "02_Leave_Policy.pdf", "page": 2},
            {"source": "06_Compensation_and_Benefits_Policy.pdf", "page": 3},
            {"source": "02_Leave_Policy.pdf", "page": 3},
        ]
        formatted = group_and_format_sources(raw_sources)
        self.assertEqual(len(formatted), 2)
        self.assertEqual(formatted[0], "📄 Compensation & Benefits Policy — pp. 1, 3")
        self.assertEqual(formatted[1], "📄 Leave Policy — pp. 2, 3")

        # Empty
        self.assertEqual(group_and_format_sources([]), [])
        self.assertEqual(group_and_format_sources(None), [])

    def test_format_compact_chunk_score(self):
        # Rerank score
        chunk1 = {"rerank_score": 6.548512}
        self.assertEqual(format_compact_chunk_score(chunk1), "6.5485")

        # Dense score
        chunk2 = {"score": 0.81234}
        self.assertEqual(format_compact_chunk_score(chunk2), "0.8123")

        # RRF score
        chunk3 = {"rrf_score": 0.0327868}
        self.assertEqual(format_compact_chunk_score(chunk3), "0.032787")

        # Missing / invalid
        self.assertEqual(format_compact_chunk_score({}), "N/A")
        self.assertEqual(format_compact_chunk_score(None), "N/A")

    def test_format_source_citation_legacy(self):
        c1 = {"source": "Leave_Policy.pdf", "page": 3}
        self.assertEqual(format_source_citation(c1), "📄 **Leave_Policy.pdf** (Page 3)")

        c2 = {"source": "Travel_Policy.pdf", "page": 0}
        self.assertEqual(format_source_citation(c2), "📄 **Travel_Policy.pdf**")

        self.assertEqual(format_source_citation({}), "📄 **Unknown Document**")
        self.assertEqual(format_source_citation(None), "📄 Unknown Document")

    def test_format_chunk_score_legacy(self):
        chunk_rerank = {"rerank_score": 4.12345}
        score_str = format_chunk_score(chunk_rerank, retrieval_mode="hybrid_rerank")
        self.assertEqual(score_str, "Cross-Encoder Score: 4.1235")


if __name__ == "__main__":
    unittest.main()
