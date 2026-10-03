"""
Pipeline module: connects retrieval and generation into a unified RAG workflow.
"""

from src.pipeline.rag_pipeline import RAGPipeline, RAGResponse

__all__ = ["RAGPipeline", "RAGResponse"]
