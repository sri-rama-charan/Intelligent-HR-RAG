"""
End-to-end integration test script for Phase 2E: Unified RAG Pipeline.
Initializes the RAGPipeline orchestrator, executes the complete query lifecycle,
and displays the final structured answer along with cited sources and debug ranks.
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

from src.ingestion.pdf_loader import load_all_pdfs
from src.ingestion.chunker import chunk_documents
from src.retrieval.embeddings import EmbeddingManager
from src.retrieval.vector_store import FAISSVectorStore
from src.generation.gemini_generator import GeminiGenerator, DEFAULT_GEMINI_MODEL
from src.pipeline.rag_pipeline import RAGPipeline


def run_pipeline_test():
    print("=" * 70)
    print("PHASE 2E: UNIFIED RAG PIPELINE END-TO-END TEST")
    print("=" * 70)

    # 1. Check for Gemini API Key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\nCONFIGURATION ERROR: GEMINI_API_KEY is not set.")
        print("Please configure GEMINI_API_KEY in your .env file to run the unified pipeline.")
        sys.exit(1)

    # 2. Build Components
    print("\n[Step 1] Initializing Pipeline Components...")
    pages = load_all_pdfs()
    chunks = chunk_documents(pages, chunk_size=800, chunk_overlap=100)
    embedder = EmbeddingManager(model_name="all-MiniLM-L6-v2")
    vector_store = FAISSVectorStore.build_from_chunks(chunks, embedder)
    generator = GeminiGenerator(api_key=api_key, model_name=DEFAULT_GEMINI_MODEL)

    # 3. Construct Unified Orchestrator
    pipeline = RAGPipeline(
        embedder=embedder,
        vector_store=vector_store,
        generator=generator,
        top_k=3
    )
    print("RAGPipeline initialized successfully.")

    # 4. Ask Test Question through Unified Pipeline
    test_question = "When does salary get credited?"
    print("\n" + "=" * 70)
    print(f"PIPELINE QUERY: \"{test_question}\"")
    print("=" * 70)

    print("\nExecuting pipeline.ask()...")
    response = pipeline.ask(test_question)

    # 5. Display Structured Output
    print("\n" + "=" * 70)
    print("UNIFIED PIPELINE RESPONSE:")
    print("=" * 70)
    print(f"Question: {response.question}")
    print(f"\nFinal Answer:\n{response.answer}")
    print(f"\nModel: {response.model}")
    print(f"\nUser-Facing Sources (Deduplicated):")
    for s in response.sources:
        print(f"  • {s['source']} — Page {s['page']}")

    print("\nUnderlying Retrieved Chunks (Debug / Evaluation):")
    for c in response.retrieved_chunks:
        print(f"  [Rank {c['rank']}] Score: {c['score']:.4f} | Chunk ID: {c['chunk_id']} | Source: {c['source']} (Page {c['page']})")

    print("\n" + "=" * 70)
    print("Phase 2E unified pipeline execution completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline_test()
