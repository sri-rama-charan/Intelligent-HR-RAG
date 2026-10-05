"""
UI helper package for the Streamlit Chat Interface.
"""

from src.ui.helpers import (
    format_source_citation,
    format_chunk_score,
    format_compact_chunk_score,
    clean_document_title,
    format_grouped_source,
    group_and_format_sources,
    get_mode_label,
    get_mode_key,
    MODE_OPTIONS,
    DEFAULT_MODE,
)

__all__ = [
    "format_source_citation",
    "format_chunk_score",
    "format_compact_chunk_score",
    "clean_document_title",
    "format_grouped_source",
    "group_and_format_sources",
    "get_mode_label",
    "get_mode_key",
    "MODE_OPTIONS",
    "DEFAULT_MODE",
]
