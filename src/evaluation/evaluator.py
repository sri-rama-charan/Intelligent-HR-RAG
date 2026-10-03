"""
Baseline Evaluator module for the HR Help Desk RAG pipeline.
Executes the test suite of 20 evaluation questions through the unified RAGPipeline,
records retrieval metrics and generated answers, and exports results to CSV and Markdown.
"""

import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from src.pipeline.rag_pipeline import RAGPipeline, RAGResponse


def load_evaluation_questions(csv_path: Path) -> List[Dict[str, str]]:
    """
    Loads evaluation questions from test.csv.

    Args:
        csv_path (Path): Path to test.csv.

    Returns:
        List[Dict[str, str]]: List of dicts with 'question_id' and 'question'.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Evaluation questions file not found: {csv_path}")

    questions = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row.get("question_id", "").strip()
            q_text = row.get("question", "").strip()
            if qid and q_text:
                questions.append({"question_id": qid, "question": q_text})

    return questions


def evaluate_pipeline(
    pipeline: RAGPipeline,
    questions: List[Dict[str, str]],
    delay_seconds: float = 1.0,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Runs each evaluation question through the unified RAGPipeline and records answers
    and retrieval details.

    Args:
        pipeline (RAGPipeline): The unified pipeline instance.
        questions (List[Dict[str, str]]): List of questions to evaluate.
        delay_seconds (float): Pause between API calls to avoid rate limits (default: 1.0s).
        top_k (int): Top-K parameter for retrieval.

    Returns:
        List[Dict[str, Any]]: Evaluated records with answer, citations, and scores.
    """
    results: List[Dict[str, Any]] = []

    for i, item in enumerate(questions, start=1):
        qid = item["question_id"]
        q_text = item["question"]
        is_in_scope = int(qid[1:]) <= 15 if qid.startswith("Q") and qid[1:].isdigit() else True

        try:
            response: RAGResponse = pipeline.ask(q_text, top_k=top_k)
            answer = response.answer
            model = response.model
            retrieved_chunks = response.retrieved_chunks
            status = "OK"
        except Exception as e:
            answer = f"ERROR: {e}"
            model = "error"
            retrieved_chunks = []
            status = "ERROR"

        # Format retrieval metadata strings for CSV
        chunk_ids = ";".join([c.get("chunk_id", "") for c in retrieved_chunks])
        scores = ";".join([f"{c.get('score', 0.0):.4f}" for c in retrieved_chunks])
        docs = ";".join([c.get("source", "") for c in retrieved_chunks])
        pages = ";".join([str(c.get("page", "")) for c in retrieved_chunks])

        # Top-1 score for statistical analysis
        top_1_score = retrieved_chunks[0].get("score", 0.0) if retrieved_chunks else 0.0

        results.append({
            "question_id": qid,
            "question": q_text,
            "in_scope": is_in_scope,
            "status": status,
            "generated_answer": answer,
            "model": model,
            "retrieved_chunk_ids": chunk_ids,
            "retrieval_scores": scores,
            "source_documents": docs,
            "source_pages": pages,
            "top_1_score": top_1_score,
            "raw_chunks": retrieved_chunks
        })

        # Throttle between live LLM requests
        if delay_seconds > 0 and i < len(questions):
            time.sleep(delay_seconds)

    return results


def calculate_retrieval_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes summary metrics from the evaluation results.

    Args:
        results: The evaluated result records.

    Returns:
        Dict[str, Any]: Aggregated stats (in-scope avg top-1 score, out-of-scope avg top-1 score, counts).
    """
    in_scope_scores = [r["top_1_score"] for r in results if r["in_scope"] and r["status"] == "OK"]
    out_scope_scores = [r["top_1_score"] for r in results if not r["in_scope"] and r["status"] == "OK"]

    avg_in_scope = float(np.mean(in_scope_scores)) if in_scope_scores else 0.0
    avg_out_scope = float(np.mean(out_scope_scores)) if out_scope_scores else 0.0

    successful = sum(1 for r in results if r["status"] == "OK")
    failed = sum(1 for r in results if r["status"] != "OK")

    return {
        "total_questions": len(results),
        "successful_generations": successful,
        "failed_generations": failed,
        "in_scope_count": sum(1 for r in results if r["in_scope"]),
        "out_of_scope_count": sum(1 for r in results if not r["in_scope"]),
        "avg_top1_in_scope": avg_in_scope,
        "avg_top1_out_of_scope": avg_out_scope
    }


def save_results_csv(results: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Saves baseline evaluation results to CSV with all required columns.
    """
    fieldnames = [
        "question_id",
        "question",
        "generated_answer",
        "model",
        "retrieved_chunk_ids",
        "retrieval_scores",
        "source_documents",
        "source_pages"
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r[k] for k in fieldnames}
            writer.writerow(row)


def generate_markdown_report(
    results: List[Dict[str, Any]],
    stats: Dict[str, Any],
    output_path: Path
) -> None:
    """
    Generates a structured Markdown baseline evaluation report.
    """
    lines = [
        "# Phase 3: Baseline Evaluation Report",
        "",
        "## Summary",
        f"- **Total Questions Evaluated:** {stats['total_questions']}",
        f"- **Successful Generations:** {stats['successful_generations']}",
        f"- **Failed Generations:** {stats['failed_generations']}",
        f"- **In-Scope HR Policy Questions (Q01–Q15):** {stats['in_scope_count']}",
        f"- **Out-of-Scope / Edge-Case Questions (Q16–Q20):** {stats['out_of_scope_count']}",
        "",
        "## Local Retrieval Analysis",
        f"- **Average Top-1 Similarity (Q01–Q15, In-Scope):** `{stats['avg_top1_in_scope']:.4f}`",
        f"- **Average Top-1 Similarity (Q16–Q20, Out-of-Scope):** `{stats['avg_top1_out_of_scope']:.4f}`",
        "",
        "> [!NOTE]",
        "> Similarity scores represent geometric closeness in the embedding space (`all-MiniLM-L6-v2`),",
        "> not statistical probability or answer correctness.",
        "",
        "---",
        "",
        "## Q01–Q15: In-Scope Results",
        ""
    ]

    in_scope_results = [r for r in results if r["in_scope"]]
    for r in in_scope_results:
        lines.extend([
            f"### [{r['question_id']}] {r['question']}",
            f"- **Generated Answer:** {r['generated_answer']}",
            f"- **Top Retrieved Sources:** `{r['source_documents']}`",
            f"- **Source Pages:** `{r['source_pages']}`",
            f"- **Retrieval Similarity Scores:** `{r['retrieval_scores']}`",
            ""
        ])

    lines.extend([
        "---",
        "",
        "## Q16–Q20: Out-of-Scope Results",
        ""
    ])

    out_scope_results = [r for r in results if not r["in_scope"]]
    for r in out_scope_results:
        # Check if the answer contains refusal wording
        answer_lower = r["generated_answer"].lower()
        refusal_keywords = ["not available", "do not contain", "sorry", "cannot", "no information", "not found"]
        is_refusal = any(kw in answer_lower for kw in refusal_keywords)
        refusal_label = "✅ Polite Refusal (Appropriate)" if is_refusal else "⚠️ Answered / Attempted without direct policy"

        lines.extend([
            f"### [{r['question_id']}] {r['question']}",
            f"- **Refusal Status:** {refusal_label}",
            f"- **Generated Answer:** {r['generated_answer']}",
            f"- **Top Retrieved Sources:** `{r['source_documents']}`",
            f"- **Source Pages:** `{r['source_pages']}`",
            f"- **Retrieval Similarity Scores:** `{r['retrieval_scores']}`",
            ""
        ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, mode="w", encoding="utf-8") as f:
        f.write("\n".join(lines))
