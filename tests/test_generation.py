"""
Unit tests for Phase 2D: Generation & Gemini LLM Integration.
Tests prompt construction, context formatting, input validation, error handling,
and response parsing using deterministic mock objects (no live API key required).
"""

import os
import unittest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.generation.prompts import (
    build_grounded_prompt,
    format_retrieved_context,
    GROUNDED_HR_SYSTEM_PROMPT
)
from src.generation.gemini_generator import (
    GeminiGenerator,
    EMPTY_CONTEXT_MESSAGE,
    DEFAULT_GEMINI_MODEL
)


class TestGeneration(unittest.TestCase):

    def setUp(self):
        # Sample retrieved chunks for testing
        self.sample_chunks = [
            {
                "rank": 1,
                "score": 0.5239,
                "text": "Salaries are credited to the employee bank account by the 7th of the following month.",
                "source": "06_Compensation_and_Benefits_Policy.pdf",
                "page": 1,
                "chunk_id": "06_Compensation_p1_c1"
            },
            {
                "rank": 2,
                "score": 0.4753,
                "text": "The payroll cut-off date is the 24th of each month.",
                "source": "06_Compensation_and_Benefits_Policy.pdf",
                "page": 1,
                "chunk_id": "06_Compensation_p1_c2"
            }
        ]

        self.sample_document_chunks = [
            Document(
                page_content="Permanent employees at grade L3 and above are eligible for WFH.",
                metadata={"source": "03_Work_From_Home_Policy.pdf", "page": 1}
            )
        ]

    # --- Prompt & Formatting Tests ---

    def test_prompt_contains_question(self):
        """Verify prompt template embeds the exact user question."""
        question = "When does salary get credited?"
        prompt = build_grounded_prompt(question, self.sample_chunks)
        self.assertIn("<QUESTION>", prompt)
        self.assertIn("When does salary get credited?", prompt)
        self.assertIn("</QUESTION>", prompt)

    def test_prompt_contains_retrieved_context(self):
        """Verify prompt template embeds the retrieved chunk text inside <HR_CONTEXT> tags."""
        question = "When does salary get credited?"
        prompt = build_grounded_prompt(question, self.sample_chunks)
        self.assertIn("<HR_CONTEXT>", prompt)
        self.assertIn("Salaries are credited to the employee bank account", prompt)
        self.assertIn("The payroll cut-off date is the 24th", prompt)
        self.assertIn("</HR_CONTEXT>", prompt)

    def test_source_and_page_metadata_in_formatted_context(self):
        """Verify formatted context displays source filename and page numbers clearly."""
        context_str = format_retrieved_context(self.sample_chunks)
        self.assertIn("[Document: 06_Compensation_and_Benefits_Policy.pdf | Page: 1]", context_str)

    def test_format_retrieved_context_supports_langchain_documents(self):
        """Verify formatting works interchangeably with LangChain Document objects."""
        context_str = format_retrieved_context(self.sample_document_chunks)
        self.assertIn("[Document: 03_Work_From_Home_Policy.pdf | Page: 1]", context_str)
        self.assertIn("Permanent employees at grade L3 and above", context_str)

    def test_empty_question_raises_value_error(self):
        """Verify empty or whitespace-only questions raise ValueError."""
        with self.assertRaises(ValueError):
            build_grounded_prompt("", self.sample_chunks)
        with self.assertRaises(ValueError):
            build_grounded_prompt("   \n\t  ", self.sample_chunks)

    # --- Generator Initialization & Error Handling Tests ---

    def test_missing_api_key_detected(self):
        """Verify initializing GeminiGenerator without API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                GeminiGenerator()
            self.assertIn("GEMINI_API_KEY is not set", str(ctx.exception))

    def test_empty_retrieved_context_returns_standard_refusal(self):
        """Verify that passing empty context returns clean refusal without calling the API."""
        with patch("google.genai.Client") as mock_client:
            generator = GeminiGenerator(api_key="test_fake_api_key")
            response = generator.generate("What is the pet policy?", [])
            
            self.assertEqual(response, EMPTY_CONTEXT_MESSAGE)
            # Ensure generate_content was NOT called
            mock_client.return_value.models.generate_content.assert_not_called()

    def test_empty_question_in_generator_raises_value_error(self):
        """Verify generator.generate raises ValueError on empty question."""
        with patch("google.genai.Client"):
            generator = GeminiGenerator(api_key="test_fake_api_key")
            with self.assertRaises(ValueError):
                generator.generate("   ", self.sample_chunks)

    def test_api_response_converted_to_clean_string(self):
        """Verify response from Gemini is cleanly extracted and returned."""
        with patch("google.genai.Client") as mock_client_cls:
            mock_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "Salaries are credited by the 7th of the following month."
            mock_instance.models.generate_content.return_value = mock_response
            mock_client_cls.return_value = mock_instance

            generator = GeminiGenerator(api_key="test_fake_api_key")
            answer = generator.generate("When does salary get credited?", self.sample_chunks)

            self.assertEqual(answer, "Salaries are credited by the 7th of the following month.")
            mock_instance.models.generate_content.assert_called_once()

    def test_api_failure_handled_cleanly(self):
        """Verify API exceptions are caught and raised as RuntimeError without crashing unhandled."""
        with patch("google.genai.Client") as mock_client_cls:
            mock_instance = MagicMock()
            mock_instance.models.generate_content.side_effect = Exception("API connection quota exceeded")
            mock_client_cls.return_value = mock_instance

            generator = GeminiGenerator(api_key="test_fake_api_key")
            with self.assertRaises(RuntimeError) as ctx:
                generator.generate("When does salary get credited?", self.sample_chunks)
            self.assertIn("Gemini API call failed", str(ctx.exception))

    def test_generate_with_sources_aggregates_metadata(self):
        """Verify generate_with_sources returns answer along with distinct source citations."""
        with patch("google.genai.Client") as mock_client_cls:
            mock_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "Salaries are credited by the 7th."
            mock_instance.models.generate_content.return_value = mock_response
            mock_client_cls.return_value = mock_instance

            generator = GeminiGenerator(api_key="test_fake_api_key")
            result = generator.generate_with_sources("When does salary get credited?", self.sample_chunks)

            self.assertIn("answer", result)
            self.assertIn("sources", result)
            self.assertEqual(result["answer"], "Salaries are credited by the 7th.")
            # Both sample chunks came from same document & page, so sources list should have 1 distinct citation
            self.assertEqual(len(result["sources"]), 1)
            self.assertEqual(result["sources"][0]["source"], "06_Compensation_and_Benefits_Policy.pdf")
            self.assertEqual(result["sources"][0]["page"], 1)


if __name__ == "__main__":
    unittest.main()
