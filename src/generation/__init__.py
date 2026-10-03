"""
Generation module: responsible for prompt templates and Gemini LLM integration.
"""

from src.generation.prompts import (
    GROUNDED_HR_SYSTEM_PROMPT,
    format_retrieved_context,
    build_grounded_prompt
)
from src.generation.gemini_generator import (
    GeminiGenerator,
    DEFAULT_GEMINI_MODEL,
    EMPTY_CONTEXT_MESSAGE
)

__all__ = [
    "GROUNDED_HR_SYSTEM_PROMPT",
    "format_retrieved_context",
    "build_grounded_prompt",
    "GeminiGenerator",
    "DEFAULT_GEMINI_MODEL",
    "EMPTY_CONTEXT_MESSAGE"
]
