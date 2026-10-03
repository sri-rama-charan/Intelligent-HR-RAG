"""
Executable script for Phase 3: Baseline Evaluation.
Runs all 20 Kaggle competition questions through the unified RAGPipeline,
records generated answers and retrieval metrics, and exports results to:
  - evaluation/baseline_results.csv
  - evaluation/baseline_report.md
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv()

from src.pipeline.rag_pipeline import RAGPipeline
from src.evaluation.evaluator import (
    load_evaluation_questions,
    evaluate_pipeline,
    calculate_retrieval_stats,
    save_results_csv,
    generate_markdown_report
)


def run_baseline_evaluation():
    print("=" * 70)
    print("PHASE 3: BASELINE EVALUATION (20 QUESTIONS)")
    print("=" * 70)

    # 1. Check API Key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\nERROR: GEMINI_API_KEY is not set in environment or .env file.")
        print("Please configure GEMINI_API_KEY to run baseline evaluation.")
        sys.exit(1)

    # 2. Locate and load test.csv
    test_csv_path = project_root / "data" / "evaluation" / "test.csv"
    if not test_csv_path.exists():
        print(f"\nERROR: test.csv not found at {test_csv_path}")
        sys.exit(1)

    questions = load_evaluation_questions(test_csv_path)
    print(f"\nLoaded {len(questions)} evaluation questions from: {test_csv_path}")

    # 3. Initialize Unified RAG Pipeline
    print("\nInitializing RAGPipeline from corpus...")
    pipeline = RAGPipeline.from_corpus(
        corpus_dir=project_root / "data" / "hr_corpus",
        api_key=api_key,
        top_k=3
    )
    print("Pipeline initialized successfully.")

    # 4. Run Evaluation across all 20 questions
    print(f"\nEvaluating {len(questions)} questions (delay: 1.0s between calls)...")
    results = evaluate_pipeline(pipeline, questions, delay_seconds=1.0, top_k=3)

    # 5. Save Artifacts immediately
    csv_out = project_root / "evaluation" / "baseline_results.csv"
    md_out = project_root / "evaluation" / "baseline_report.md"

    save_results_csv(results, csv_out)
    generate_markdown_report(results, stats_dummy := calculate_retrieval_stats(results), md_out)

    for r in results:
        status_symbol = "[OK]" if r["status"] == "OK" else "[FAIL]"
        print(f"[{r['question_id']}] {status_symbol} Top-1 Score: {r['top_1_score']:.4f} | {r['question'][:45]}...")

    # 6. Calculate Metrics and display summary
    stats = stats_dummy
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY:")
    print("=" * 70)
    print(f"Total Questions:          {stats['total_questions']}")
    print(f"Successful Generations:   {stats['successful_generations']}")
    print(f"Failed Generations:       {stats['failed_generations']}")
    print(f"In-Scope Questions:       {stats['in_scope_count']} (Q01-Q15)")
    print(f"Out-of-Scope Questions:   {stats['out_of_scope_count']} (Q16-Q20)")
    print(f"Avg Top-1 In-Scope:       {stats['avg_top1_in_scope']:.4f}")
    print(f"Avg Top-1 Out-of-Scope:   {stats['avg_top1_out_of_scope']:.4f}")

    print(f"\nResults saved to:  {csv_out}")
    print(f"Report saved to:   {md_out}")
    print("\n" + "=" * 70)
    print("Phase 3 baseline evaluation completed.")
    print("=" * 70)


if __name__ == "__main__":
    run_baseline_evaluation()
