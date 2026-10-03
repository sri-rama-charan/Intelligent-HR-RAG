"""
Gemini Generator module for the HR Help Desk RAG pipeline.
Responsible for interacting with the Google Gemini LLM using the google-genai SDK,
formatting grounded prompts from retrieved chunks, and generating policy-accurate answers.
"""

import os
from typing import Any, Dict, List, Optional, Union
from dotenv import load_dotenv
from google import genai
from langchain_core.documents import Document

from src.generation.prompts import build_grounded_prompt

# Load environment variables from .env if present
load_dotenv()

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
EMPTY_CONTEXT_MESSAGE = (
    "I am sorry, but the provided HR policy documents do not contain information regarding this topic."
)


class GeminiGenerator:
    """
    Manages communication with the Google Gemini LLM to produce grounded answers.
    Completely decoupled from retrieval mechanisms (FAISS).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """
        Initializes the Gemini client using the google-genai SDK.

        Args:
            api_key (Optional[str]): Gemini API key. Defaults to GEMINI_API_KEY or GOOGLE_API_KEY from env.
            model_name (Optional[str]): Gemini model name. Defaults to GEMINI_MODEL env or 'gemini-2.5-flash'.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please set GEMINI_API_KEY in your .env file or environment."
            )

        self.model_name = model_name or DEFAULT_GEMINI_MODEL

        # Initialize the official Google GenAI client
        self.client = genai.Client(api_key=self.api_key)

    def generate(
        self,
        question: str,
        retrieved_chunks: List[Union[Dict[str, Any], Document]]
    ) -> str:
        """
        Generates a grounded natural language answer for the user's question
        using the provided retrieved HR policy chunks.

        Args:
            question (str): The employee's question.
            retrieved_chunks: List of retrieved chunk dictionaries or Document objects.

        Returns:
            str: Plain text answer grounded in the provided context.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        # Guardrail: If no context was retrieved at all, return the standard refusal directly
        if not retrieved_chunks:
            return EMPTY_CONTEXT_MESSAGE

        # Build grounded prompt with strict instructions and explicit context boundaries
        prompt = build_grounded_prompt(question, retrieved_chunks)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            # Extract plain text from response
            answer_text = response.text if response and hasattr(response, "text") and response.text else ""
            return answer_text.strip()

        except Exception as e:
            # Wrap error cleanly without leaking sensitive details or API keys
            raise RuntimeError(f"Gemini API call failed: {e}") from None

    def generate_with_sources(
        self,
        question: str,
        retrieved_chunks: List[Union[Dict[str, Any], Document]]
    ) -> Dict[str, Any]:
        """
        Generates an answer and bundles it with the cited source document metadata.

        Args:
            question (str): The employee's question.
            retrieved_chunks: List of retrieved chunk dictionaries or Document objects.

        Returns:
            Dict[str, Any]: Dictionary containing answer, question, model name, and unique sources.
        """
        answer = self.generate(question, retrieved_chunks)

        # Extract unique source and page citations from retrieved chunks
        sources = []
        seen = set()
        for item in retrieved_chunks:
            if isinstance(item, dict):
                src = item.get("source", "Unknown")
                pg = item.get("page", 0)
            elif isinstance(item, Document):
                src = item.metadata.get("source", "Unknown")
                pg = item.metadata.get("page", 0)
            else:
                src, pg = "Unknown", 0

            key = (src, pg)
            if key not in seen:
                seen.add(key)
                sources.append({"source": src, "page": pg})

        return {
            "question": question,
            "answer": answer,
            "model": self.model_name,
            "sources": sources
        }
