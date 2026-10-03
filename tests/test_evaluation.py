"""
Unit tests for Phase 3: Baseline Evaluation.
Tests question loading, evaluation iteration, statistical metric calculations,
CSV export, and Markdown report generation using deterministic mocks (zero network calls).
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.pipeline.rag_pipeline import RAGResponse
from src.evaluation.evaluator import (
    load_evaluation_questions,
    evaluate_pipeline,
    calculate_retrieval_stats,
    save_results_csv,
    generate_markdown_report
)


class TestEvaluation(unittest.TestCase):

    def setUp(self):
        self.project_root = Path(__file__).resolve().parent.parent
        self.test_csv_path = self.project_root / "data" / "evaluation" / "test.csv"

        # Mock RAGPipeline
        self.mock_pipeline = MagicMock()
        self.mock_response = RAGResponse(
            question="When does salary get credited?",
            answer="Salaries are credited on the 7th.",
            sources=[{"source": "06_Compensation_and_Benefits_Policy.pdf", "page": 1}],
            retrieved_chunks=[
                {
                    "rank": 1,
                    "score": 0.5239,
                    "chunk_id": "06_Comp_p1_c1",
                    "source": "06_Compensation_and_Benefits_Policy.pdf",
                    "page": 1,
                    "text": "Salaries are processed by the 7th."
                }
            ],
            model="gemini-mock"
        )
        self.mock_pipeline.ask.return_value = self.mock_response

    def test_load_evaluation_questions_reads_test_csv(self):
        """Verify load_evaluation_questions loads all 20 questions with question_id and question."""
        questions = load_evaluation_questions(self.test_csv_path)
        self.assertEqual(len(questions), 20)
        self.assertEqual(questions[0]["question_id"], "Q01")
        self.assertEqual(questions[-1]["question_id"], "Q20")
        for q in questions:
            self.assertIn("question_id", q)
            self.assertIn("question", q)
            self.assertTrue(len(q["question_id"]) > 0)
            self.assertTrue(len(q["question"]) > 0)

    def test_evaluate_pipeline_iterates_and_formats_records(self):
        """Verify evaluate_pipeline runs all provided questions through pipeline.ask."""
        sample_questions = [
            {"question_id": "Q01", "question": "Question 1"},
            {"question_id": "Q16", "question": "Question 16"}
        ]
        results = evaluate_pipeline(self.mock_pipeline, sample_questions, delay_seconds=0.0)

        self.assertEqual(len(results), 2)
        self.assertEqual(self.mock_pipeline.ask.call_count, 2)
        
        # In-scope flag test
        self.assertTrue(results[0]["in_scope"])
        self.assertFalse(results[1]["in_scope"])
        self.assertEqual(results[0]["status"], "OK")
        self.assertEqual(results[0]["generated_answer"], "Salaries are credited on the 7th.")

    def test_evaluate_pipeline_handles_errors_gracefully(self):
        """Verify that an exception in pipeline.ask is caught and marked as ERROR."""
        self.mock_pipeline.ask.side_effect = Exception("API rate limit error")
        sample_questions = [{"question_id": "Q01", "question": "Question 1"}]

        results = evaluate_pipeline(self.mock_pipeline, sample_questions, delay_seconds=0.0)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "ERROR")
        self.assertIn("ERROR:", results[0]["generated_answer"])

    def test_calculate_retrieval_stats(self):
        """Verify calculate_retrieval_stats computes in-scope and out-of-scope averages accurately."""
        sample_results = [
            {"question_id": "Q01", "in_scope": True, "status": "OK", "top_1_score": 0.60},
            {"question_id": "Q02", "in_scope": True, "status": "OK", "top_1_score": 0.40},
            {"question_id": "Q16", "in_scope": False, "status": "OK", "top_1_score": 0.30},
            {"question_id": "Q17", "in_scope": False, "status": "OK", "top_1_score": 0.20}
        ]
        stats = calculate_retrieval_stats(sample_results)

        self.assertEqual(stats["total_questions"], 4)
        self.assertEqual(stats["successful_generations"], 4)
        self.assertEqual(stats["in_scope_count"], 2)
        self.assertEqual(stats["out_of_scope_count"], 2)
        self.assertAlmostEqual(stats["avg_top1_in_scope"], 0.50)
        self.assertAlmostEqual(stats["avg_top1_out_of_scope"], 0.25)

    def test_save_results_csv(self):
        """Verify save_results_csv writes expected columns and rows."""
        sample_results = [
            {
                "question_id": "Q01",
                "question": "When does salary get credited?",
                "generated_answer": "On the 7th.",
                "model": "gemini-mock",
                "retrieved_chunk_ids": "chunk_1",
                "retrieval_scores": "0.5239",
                "source_documents": "Policy.pdf",
                "source_pages": "1"
            }
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "test_out.csv"
            save_results_csv(sample_results, csv_path)

            self.assertTrue(csv_path.exists())
            with open(csv_path, mode="r", encoding="utf-8") as f:
                content = f.read()
                self.assertIn("question_id,question,generated_answer", content)
                self.assertIn("Q01", content)
                self.assertIn("On the 7th.", content)

    def test_generate_markdown_report(self):
        """Verify generate_markdown_report writes summary, in-scope, out-of-scope sections."""
        sample_results = [
            {
                "question_id": "Q01",
                "question": "When does salary get credited?",
                "in_scope": True,
                "status": "OK",
                "generated_answer": "On the 7th.",
                "source_documents": "Policy.pdf",
                "source_pages": "1",
                "retrieval_scores": "0.5239"
            },
            {
                "question_id": "Q16",
                "question": "What is the pet policy?",
                "in_scope": False,
                "status": "OK",
                "generated_answer": "I am sorry, but the provided HR policy documents do not contain information regarding this topic.",
                "source_documents": "Other.pdf",
                "source_pages": "2",
                "retrieval_scores": "0.3120"
            }
        ]
        stats = {
            "total_questions": 2,
            "successful_generations": 2,
            "failed_generations": 0,
            "in_scope_count": 1,
            "out_of_scope_count": 1,
            "avg_top1_in_scope": 0.5239,
            "avg_top1_out_of_scope": 0.3120
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            md_path = Path(tmp_dir) / "test_report.md"
            generate_markdown_report(sample_results, stats, md_path)

            self.assertTrue(md_path.exists())
            with open(md_path, mode="r", encoding="utf-8") as f:
                content = f.read()
                self.assertIn("Phase 3: Baseline Evaluation Report", content)
                self.assertIn("Q01–Q15: In-Scope Results", content)
                self.assertIn("Q16–Q20: Out-of-Scope Results", content)
                self.assertIn("Polite Refusal", content)


if __name__ == "__main__":
    unittest.main()
