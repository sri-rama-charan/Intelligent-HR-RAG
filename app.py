"""
HR Help Desk — Intelligent RAG Chat Application.
Streamlit presentation layer connecting directly to RAGPipeline.

Production retrieval pipeline:
  User Query -> Hybrid Retrieval (FAISS + BM25) -> RRF Fusion -> Cross-Encoder Reranker -> Grounded Answer
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import streamlit as st

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

from src.pipeline.rag_pipeline import RAGPipeline, RAGResponse
from src.ui.helpers import (
    clean_document_title,
    group_and_format_sources,
    format_compact_chunk_score,
)

# Production retrieval configuration
PRODUCTION_RETRIEVAL_MODE = "hybrid_rerank"

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="HR Help Desk — Policy Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# 2. Pipeline Initialization (Cached Resource)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading HR policy documents and initializing retrieval models...")
def initialize_pipeline() -> RAGPipeline:
    """
    Initializes the production RAGPipeline once with all components:
    FAISS vector store, BM25 store, Cross-Encoder reranker, and Gemini generator.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Please add GEMINI_API_KEY to your .env file."
        )

    pipeline = RAGPipeline.from_corpus(
        corpus_dir=PROJECT_ROOT / "data" / "hr_corpus",
        retrieval_mode=PRODUCTION_RETRIEVAL_MODE,
        top_k=5,
        candidate_pool_size=20,
    )
    return pipeline


# -----------------------------------------------------------------------------
# 3. Session State Management
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------------------------------------------------------
# 4. Compact Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🏢 HR Help Desk")
    st.caption("AI-Powered Policy Assistant")
    st.markdown("---")

    st.markdown("**STATUS**")
    st.success("✓ Hybrid Retrieval + Reranking")
    st.markdown("---")

    st.markdown("**SYSTEM**")
    st.markdown(
        "✓ 11 HR Policies  \n"
        "✓ Hybrid Retrieval  \n"
        "✓ Cross-Encoder Reranking  \n"
        "✓ Gemini Generation"
    )
    st.markdown("---")

    with st.expander("ℹ️ How it works", expanded=False):
        st.markdown(
            "**Question**  \n"
            "&nbsp;&nbsp;&nbsp;&nbsp;↓  \n"
            "**Hybrid Retrieval**  \n"
            "&nbsp;&nbsp;&nbsp;&nbsp;↓  \n"
            "**RRF Fusion**  \n"
            "&nbsp;&nbsp;&nbsp;&nbsp;↓  \n"
            "**Reranking**  \n"
            "&nbsp;&nbsp;&nbsp;&nbsp;↓  \n"
            "**Grounded Answer**"
        )
    st.markdown("---")

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# -----------------------------------------------------------------------------
# 5. Main Chat Area Header
# -----------------------------------------------------------------------------
st.title("🏢 HR Policy Help Desk")
st.markdown(
    "Ask any question regarding company leave, travel, medical benefits, work from home, "
    "or code of conduct. Answers are strictly grounded in official HR policy documents."
)
st.markdown("---")


# -----------------------------------------------------------------------------
# 6. Pipeline Loading & Health Check
# -----------------------------------------------------------------------------
pipeline = None
init_error = None

try:
    pipeline = initialize_pipeline()
except Exception as e:
    init_error = str(e)

if init_error:
    st.error(
        f"**Failed to initialize RAG Pipeline:** {init_error}\n\n"
        "Please check your `.env` configuration and ensure `GEMINI_API_KEY` is set correctly."
    )
    st.stop()


# -----------------------------------------------------------------------------
# 7. Render Conversation History & Components
# -----------------------------------------------------------------------------
def render_sources(sources: List[Dict[str, Any]]) -> None:
    """Renders compact, grouped source citations."""
    if not sources:
        return
    grouped = group_and_format_sources(sources)
    if not grouped:
        return

    doc_count = len(grouped)
    doc_label = "document" if doc_count == 1 else "documents"

    if doc_count <= 2:
        st.markdown(f"**📚 Sources · {doc_count} {doc_label}**")
        for item in grouped:
            st.markdown(item)
    else:
        with st.expander(f"📚 Sources · {doc_count} {doc_label}", expanded=False):
            for item in grouped:
                st.markdown(item)


def render_retrieved_context(chunks: List[Dict[str, Any]]) -> None:
    """Renders compact context inspection without clutter."""
    if not chunks:
        return
    with st.expander(f"🔍 View Retrieved Context ({len(chunks)} chunks)", expanded=False):
        for idx, chunk in enumerate(chunks, 1):
            raw_source = chunk.get("source", "Unknown Document")
            page_num = chunk.get("page", 0)
            score_str = format_compact_chunk_score(chunk)
            chunk_text = chunk.get("text", "").strip()

            st.markdown(
                f"**Chunk #{idx}**  \n"
                f"Source: `{raw_source}` · Page {page_num}  \n"
                f"Score: `{score_str}`"
            )
            st.markdown(f"> {chunk_text}")
            if idx < len(chunks):
                st.divider()


def render_assistant_response(msg: Dict[str, Any]) -> None:
    """Renders an assistant message with answer, citations, and expandable context."""
    # 1. Answer (primary visual emphasis)
    st.markdown(msg["content"])

    # 2. Sources (secondary, compact)
    render_sources(msg.get("sources", []))

    # 3. Retrieved context (optional, collapsed)
    render_retrieved_context(msg.get("retrieved_chunks", []))


for msg_idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_assistant_response(message)


# -----------------------------------------------------------------------------
# 8. Chat Input & Query Execution Flow
# -----------------------------------------------------------------------------
user_query = st.chat_input("Ask an HR policy question (e.g., 'How many Earned Leaves can I carry forward?')...")

if user_query is not None:
    cleaned_query = user_query.strip()
    if not cleaned_query:
        st.warning("Please enter a question.")
    else:
        # 1. Display User Message
        st.session_state.messages.append({"role": "user", "content": cleaned_query})
        with st.chat_message("user"):
            st.markdown(cleaned_query)

        # 2. Execute RAG Pipeline with production hybrid_rerank mode
        with st.chat_message("assistant"):
            with st.spinner("Searching policies and generating answer..."):
                try:
                    pipeline.retrieval_mode = PRODUCTION_RETRIEVAL_MODE
                    response: RAGResponse = pipeline.ask(cleaned_query)

                    assistant_msg = {
                        "id": f"msg_{len(st.session_state.messages)}",
                        "role": "assistant",
                        "content": response.answer,
                        "sources": response.sources,
                        "retrieved_chunks": response.retrieved_chunks,
                        "retrieval_mode": response.retrieval_mode,
                    }
                    render_assistant_response(assistant_msg)
                    st.session_state.messages.append(assistant_msg)

                except Exception as err:
                    error_msg = f"An error occurred while answering your question: {err}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {
                            "id": f"msg_err_{len(st.session_state.messages)}",
                            "role": "assistant",
                            "content": f"⚠️ {error_msg}",
                            "sources": [],
                            "retrieved_chunks": [],
                            "retrieval_mode": PRODUCTION_RETRIEVAL_MODE,
                        }
                    )
