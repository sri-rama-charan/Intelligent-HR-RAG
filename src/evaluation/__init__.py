"""
Evaluation module: benchmarks retrieval accuracy and generation quality.
"""

from src.evaluation.evaluator import (
    load_evaluation_questions,
    evaluate_pipeline,
    calculate_retrieval_stats,
    save_results_csv,
    generate_markdown_report
)

__all__ = [
    "load_evaluation_questions",
    "evaluate_pipeline",
    "calculate_retrieval_stats",
    "save_results_csv",
    "generate_markdown_report"
]
