"""
UI Helper functions and constants for the HR RAG Streamlit Chat Interface.
Pure helper logic kept separate from Streamlit display rendering for testability.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# Supported retrieval modes mapping
MODE_DISPLAY_MAP = {
    "Hybrid + Reranker": "hybrid_rerank",
    "Hybrid": "hybrid",
    "FAISS": "faiss",
}

# Reverse mapping: internal key to display label
DISPLAY_NAME_MAP = {
    "hybrid_rerank": "Hybrid + Reranker",
    "hybrid": "Hybrid",
    "faiss": "FAISS",
}

MODE_OPTIONS = list(MODE_DISPLAY_MAP.keys())
DEFAULT_MODE = "Hybrid + Reranker"

# Clean presentation mapping for official HR policy documents
POLICY_NAME_MAP = {
    "00_Company_Profile.pdf": "Company Profile",
    "01_Employee_Handbook.pdf": "Employee Handbook",
    "02_Leave_Policy.pdf": "Leave Policy",
    "03_Work_From_Home_Policy.pdf": "Work From Home Policy",
    "04_Code_of_Conduct.pdf": "Code of Conduct",
    "05_Performance_Review_Policy.pdf": "Performance Review Policy",
    "06_Compensation_and_Benefits_Policy.pdf": "Compensation & Benefits Policy",
    "07_IT_and_Data_Security_Policy.pdf": "IT & Data Security Policy",
    "08_Prevention_of_Sexual_Harassment_Policy.pdf": "Prevention of Sexual Harassment Policy",
    "09_Onboarding_and_Separation_Policy.pdf": "Onboarding & Separation Policy",
    "10_Travel_and_Expense_Policy.pdf": "Travel & Expense Policy",
}


def clean_document_title(filename: str) -> str:
    """
    Cleans a PDF filename into a clean, human-readable document title.

    Args:
        filename (str): Document filename (e.g. '02_Leave_Policy.pdf').

    Returns:
        str: Human-readable document name (e.g. 'Leave Policy').
    """
    if not filename:
        return "Unknown Document"

    fname = Path(str(filename)).name
    if fname in POLICY_NAME_MAP:
        return POLICY_NAME_MAP[fname]

    # Clean fallback: remove leading numbering, extension, and underscores
    clean = re.sub(r"^\d+_", "", fname)
    if clean.lower().endswith(".pdf"):
        clean = clean[:-4]
    clean = clean.replace("_and_", " & ").replace("_", " ")
    return clean.strip() or fname


def format_grouped_source(doc_title: str, pages: List[Any]) -> str:
    """
    Formats a document title and list of page numbers into a compact citation string.

    Args:
        doc_title (str): Clean document title.
        pages (List[Any]): List of page numbers (integers or numeric strings).

    Returns:
        str: Compact citation (e.g., '📄 Leave Policy — pp. 2, 3' or '📄 Leave Policy — p. 2').
    """
    valid_pages = []
    for p in pages:
        if p is not None:
            try:
                page_int = int(p)
                if page_int > 0 and page_int not in valid_pages:
                    valid_pages.append(page_int)
            except (ValueError, TypeError):
                continue

    valid_pages.sort()

    if not valid_pages:
        return f"📄 {doc_title}"
    if len(valid_pages) == 1:
        return f"📄 {doc_title} — p. {valid_pages[0]}"
    pages_str = ", ".join(str(p) for p in valid_pages)
    return f"📄 {doc_title} — pp. {pages_str}"


def group_and_format_sources(sources: List[Dict[str, Any]]) -> List[str]:
    """
    Deduplicates and groups source entries by document, combining page numbers.

    Args:
        sources (List[Dict[str, Any]]): List of raw source dicts with 'source' and 'page'.

    Returns:
        List[str]: List of formatted strings grouped by document.
    """
    if not sources:
        return []

    # Map doc_title -> list of pages, preserving order of appearance
    doc_order: List[str] = []
    doc_pages: Dict[str, List[Any]] = {}

    for item in sources:
        if not isinstance(item, dict):
            continue
        raw_source = item.get("source", "Unknown Document")
        doc_title = clean_document_title(raw_source)
        page = item.get("page")

        if doc_title not in doc_pages:
            doc_order.append(doc_title)
            doc_pages[doc_title] = []

        if page is not None:
            doc_pages[doc_title].append(page)

    formatted_list = []
    for doc in doc_order:
        formatted_list.append(format_grouped_source(doc, doc_pages[doc]))

    return formatted_list


def format_compact_chunk_score(chunk: Dict[str, Any]) -> str:
    """
    Returns a clean numerical score string for context inspection.

    Args:
        chunk (Dict[str, Any]): Retrieved chunk dictionary.

    Returns:
        str: Formatted score value (e.g., '6.5485').
    """
    if not isinstance(chunk, dict):
        return "N/A"

    if "rerank_score" in chunk:
        try:
            return f"{float(chunk['rerank_score']):.4f}"
        except (ValueError, TypeError):
            return str(chunk["rerank_score"])

    if "score" in chunk:
        try:
            return f"{float(chunk['score']):.4f}"
        except (ValueError, TypeError):
            return str(chunk["score"])

    if "rrf_score" in chunk:
        try:
            return f"{float(chunk['rrf_score']):.6f}"
        except (ValueError, TypeError):
            return str(chunk["rrf_score"])

    return "N/A"


def get_mode_key(display_label: str) -> str:
    """
    Maps a user-friendly UI display label to internal pipeline retrieval_mode key.

    Args:
        display_label (str): Selected option.

    Returns:
        str: Internal mode key ('hybrid_rerank', 'hybrid', or 'faiss').
    """
    return MODE_DISPLAY_MAP.get(display_label, "hybrid_rerank")


def get_mode_label(mode_key: str) -> str:
    """
    Maps an internal pipeline retrieval_mode key to user-friendly UI display label.

    Args:
        mode_key (str): Internal mode key ('hybrid_rerank', 'hybrid', or 'faiss').

    Returns:
        str: UI display label.
    """
    return DISPLAY_NAME_MAP.get(mode_key, "Hybrid + Reranker")


def format_source_citation(source_dict: Dict[str, Any]) -> str:
    """
    Formats a single source dictionary into a clean markdown citation string.
    Retained for backward compatibility.
    """
    if not isinstance(source_dict, dict):
        return "📄 Unknown Document"

    source = source_dict.get("source", "Unknown Document")
    page = source_dict.get("page")

    if page is not None and str(page).strip() and str(page) != "0":
        return f"📄 **{source}** (Page {page})"
    return f"📄 **{source}**"


def format_chunk_score(chunk: Dict[str, Any], retrieval_mode: Optional[str] = None) -> str:
    """
    Extracts and formats the score label and value. Retained for backward compatibility.
    """
    if not isinstance(chunk, dict):
        return "Score: N/A"

    mode = retrieval_mode or chunk.get("retrieval_mode")

    if mode == "hybrid_rerank" and "rerank_score" in chunk:
        score_val = chunk["rerank_score"]
        try:
            return f"Cross-Encoder Score: {float(score_val):.4f}"
        except (ValueError, TypeError):
            return f"Cross-Encoder Score: {score_val}"

    if mode == "hybrid" and "rrf_score" in chunk:
        score_val = chunk["rrf_score"]
        try:
            return f"RRF Score: {float(score_val):.6f}"
        except (ValueError, TypeError):
            return f"RRF Score: {score_val}"

    if "rerank_score" in chunk:
        try:
            return f"Cross-Encoder Score: {float(chunk['rerank_score']):.4f}"
        except (ValueError, TypeError):
            return f"Cross-Encoder Score: {chunk['rerank_score']}"

    if "rrf_score" in chunk:
        try:
            return f"RRF Score: {float(chunk['rrf_score']):.6f}"
        except (ValueError, TypeError):
            return f"RRF Score: {chunk['rrf_score']}"

    if "score" in chunk:
        score_val = chunk["score"]
        try:
            return f"Similarity Score: {float(score_val):.4f}"
        except (ValueError, TypeError):
            return f"Similarity Score: {score_val}"

    return "Score: N/A"
