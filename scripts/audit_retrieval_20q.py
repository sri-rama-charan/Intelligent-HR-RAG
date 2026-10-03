"""
Local retrieval audit script across all 20 evaluation questions.
Computes top-3 retrieval metrics without consuming any Gemini API quota.
"""

import sys
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ingestion.pdf_loader import load_all_pdfs
from src.ingestion.chunker import chunk_documents
from src.retrieval.embeddings import EmbeddingManager
from src.retrieval.vector_store import FAISSVectorStore
from src.evaluation.evaluator import load_evaluation_questions


def run_audit():
    questions = load_evaluation_questions(Path("data/evaluation/test.csv"))
    pages = load_all_pdfs()
    chunks = chunk_documents(pages, chunk_size=800, chunk_overlap=100)
    embedder = EmbeddingManager(model_name="all-MiniLM-L6-v2")
    vector_store = FAISSVectorStore.build_from_chunks(chunks, embedder)

    in_scope_scores = []
    out_scope_scores = []

    print("=" * 80)
    print("LOCAL RETRIEVAL AUDIT ACROSS ALL 20 QUESTIONS (ZERO API QUOTA USED)")
    print("=" * 80)

    for q in questions:
        qid = q["question_id"]
        is_in_scope = int(qid[1:]) <= 15
        q_vec = embedder.embed_query(q["question"], normalize=True)
        res = vector_store.search(q_vec, top_k=3)
        top1 = res[0]
        score = top1["score"]

        if is_in_scope:
            in_scope_scores.append(score)
        else:
            out_scope_scores.append(score)

        category = "IN-SCOPE " if is_in_scope else "OUT-SCOPE"
        sources_str = ", ".join([f"{c['source']} (p.{c['page']}, s:{c['score']:.2f})" for c in res])
        print(f"[{qid}] ({category}) Top Score: {score:.4f} | Top Source: {top1['source']} (p.{top1['page']})")
        print(f"      Q: \"{q['question']}\"")
        print(f"      Top-3: {sources_str}\n")

    print("=" * 80)
    print(f"Average Top-1 Similarity for In-Scope (Q01-Q15):      {np.mean(in_scope_scores):.4f}")
    print(f"Average Top-1 Similarity for Out-of-Scope (Q16-Q20):  {np.mean(out_scope_scores):.4f}")
    print("=" * 80)


if __name__ == "__main__":
    run_audit()
