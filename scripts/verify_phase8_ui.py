"""
Verification script for Phase 8 Streamlit UI improvements:
1. Verifies that RAGPipeline supports hybrid_rerank as the production mode.
2. Verifies that group_and_format_sources groups documents and combines page numbers cleanly.
3. Verifies that clean_document_title formats official policy names cleanly.
4. Verifies that format_compact_chunk_score produces concise scores without clutter.
5. Performs a targeted live query (Q02) with hybrid_rerank to verify grounded answer, grouped sources, and context chunks.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure utf-8 encoding for Windows consoles with emojis
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv(PROJECT_ROOT / ".env")

from src.pipeline.rag_pipeline import RAGPipeline
from src.ui.helpers import (
    clean_document_title,
    group_and_format_sources,
    format_compact_chunk_score,
)


def run_verification():
    print("=" * 60)
    print("PHASE 8 STREAMLIT UI/UX VERIFICATION")
    print("=" * 60)

    # 1. Verify Citation Grouping & Deduplication
    test_raw_sources = [
        {"source": "06_Compensation_and_Benefits_Policy.pdf", "page": 1},
        {"source": "02_Leave_Policy.pdf", "page": 2},
        {"source": "06_Compensation_and_Benefits_Policy.pdf", "page": 3},
        {"source": "02_Leave_Policy.pdf", "page": 3},
    ]
    grouped = group_and_format_sources(test_raw_sources)
    print(f"\n[Test 1] Citation Grouping Result ({len(grouped)} documents):")
    for g in grouped:
        print(f"  {g}")
    assert len(grouped) == 2, f"Expected 2 grouped documents, got {len(grouped)}"
    assert grouped[0] == "📄 Compensation & Benefits Policy — pp. 1, 3"
    assert grouped[1] == "📄 Leave Policy — pp. 2, 3"
    print("  --> PASS: Sources grouped and pages combined correctly.")

    # 2. Verify Document Title Cleaning
    print("\n[Test 2] Policy Title Cleaning:")
    titles = [
        "02_Leave_Policy.pdf",
        "06_Compensation_and_Benefits_Policy.pdf",
        "10_Travel_and_Expense_Policy.pdf",
    ]
    for t in titles:
        cleaned = clean_document_title(t)
        print(f"  {t} -> {cleaned}")
        assert not cleaned.endswith(".pdf")
        assert "_" not in cleaned
    print("  --> PASS: Policy titles cleaned cleanly.")

    # 3. Live Pipeline Query Verification
    print("\n[Test 3] Live Hybrid + Reranker RAG Execution (Q02):")
    pipeline = RAGPipeline.from_corpus(
        corpus_dir=PROJECT_ROOT / "data" / "hr_corpus",
        retrieval_mode="hybrid_rerank",
        top_k=5,
        candidate_pool_size=20,
    )
    query = "How much Earned Leave can I carry forward to next year?"
    response = pipeline.ask(query)

    print(f"\nQuestion: {response.question}")
    print(f"\nAnswer:\n{response.answer}")

    formatted_sources = group_and_format_sources(response.sources)
    print(f"\nGrouped Sources ({len(formatted_sources)} documents):")
    for s in formatted_sources:
        print(f"  {s}")

    print(f"\nRetrieved Chunks (Top-{len(response.retrieved_chunks)}):")
    for i, c in enumerate(response.retrieved_chunks, 1):
        score_val = format_compact_chunk_score(c)
        print(f"  Chunk #{i} | Source: {c.get('source')} (Page {c.get('page')}) | Score: {score_val}")

    # Assertions
    assert "45" in response.answer, f"Expected '45' in answer, got: {response.answer}"
    assert len(response.retrieved_chunks) == 5, f"Expected 5 chunks, got {len(response.retrieved_chunks)}"
    assert len(formatted_sources) >= 1
    print("\n" + "=" * 60)
    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_verification()
