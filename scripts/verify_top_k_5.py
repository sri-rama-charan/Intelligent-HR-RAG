"""
Script: scripts/verify_top_k_5.py
Purpose: Phase 4B Retrieval-only verification for Top-K = 5.
Evaluates Q02, Q15, and Q10 with Top-K=5.
Makes ZERO Gemini API calls.
"""

from pathlib import Path
import sys

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.pdf_loader import load_all_pdfs
from src.ingestion.chunker import chunk_documents
from src.retrieval.embeddings import EmbeddingManager
from src.retrieval.vector_store import FAISSVectorStore


def verify_top_k_5():
    print("=" * 70)
    print("PHASE 4B: TOP-K = 5 RETRIEVAL VERIFICATION")
    print("Zero Gemini API calls will be made.")
    print("=" * 70)

    corpus_dir = project_root / "data" / "hr_corpus"
    documents = load_all_pdfs(corpus_dir)
    chunks = chunk_documents(documents)
    embedder = EmbeddingManager()
    vector_store = FAISSVectorStore.build_from_chunks(chunks, embedder)

    queries = {
        "Q02": "How much Earned Leave can I carry forward to next year?",
        "Q15": "What's my hotel and daily allowance for international travel?",
        "Q10": "Am I eligible to work from home at my grade?"
    }

    for qid, qtext in queries.items():
        print("\n" + "=" * 70)
        print(f"[{qid}] \"{qtext}\" (Top-K = 5)")
        print("=" * 70)
        q_vec = embedder.embed_query(qtext, normalize=True)
        results = vector_store.search(q_vec, top_k=5)

        print(f"{'Rank':<5} | {'Score':<7} | {'Chunk ID':<35} | {'Doc':<30} | {'Pg':<3}")
        print("-" * 88)
        for r in results:
            print(f"{r['rank']:<5} | {r['score']:<7.4f} | {r['chunk_id']:<35} | {r['source']:<30} | {r['page']:<3}")
            preview = r['text'][:180].replace('\n', ' ')
            print(f"       Preview: {preview}...\n")


if __name__ == "__main__":
    verify_top_k_5()
