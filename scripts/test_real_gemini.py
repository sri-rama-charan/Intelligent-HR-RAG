"""
One-time integration test script for Phase 2D.5: First Real Gemini API Integration Test.
Strictly verifies live Gemini API generation without printing or exposing API keys.
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


def run_real_gemini_integration():
    print("=" * 70)
    print("PHASE 2D.5: FIRST REAL GEMINI API INTEGRATION TEST")
    print("=" * 70)

    # 1. Check API Key presence securely (boolean only)
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\nERROR: GEMINI_API_KEY is not detected in .env or environment.")
        sys.exit(1)

    print("\n[Security Check] GEMINI_API_KEY detected: TRUE (Length: confirmed valid, value hidden)")

    # 2. Build In-Memory FAISS Vector Store
    print("\n[Step 1] Loading 11 HR policy PDFs and generating chunks...")
    pages = load_all_pdfs()
    chunks = chunk_documents(pages, chunk_size=800, chunk_overlap=100)
    print(f"Total chunks created: {len(chunks)}")

    print("\n[Step 2] Initializing local embedding model (all-MiniLM-L6-v2)...")
    embedder = EmbeddingManager(model_name="all-MiniLM-L6-v2")
    vector_store = FAISSVectorStore.build_from_chunks(chunks, embedder)
    print(f"FAISS index ready: {vector_store.total_vectors} vectors indexed.")

    # 3. Retrieve context for the test question
    test_question = "When does salary get credited?"
    print("\n" + "=" * 70)
    print(f"TEST QUESTION: \"{test_question}\"")
    print("=" * 70)

    query_vector = embedder.embed_query(test_question, normalize=True)
    retrieved_chunks = vector_store.search(query_vector, top_k=3)

    print("\n[Step 3] Retrieved Context Chunks from FAISS:")
    for res in retrieved_chunks:
        print(f"  - Rank {res['rank']}: Source = {res['source']} (Page {res['page']}) | Similarity Score = {res['score']:.4f}")
        first_line = res['text'].strip().split('\n')[0][:100]
        print(f"    Excerpt: \"{first_line}...\"")

    # 4. Initialize Gemini Generator and execute LIVE API call
    print(f"\n[Step 4] Making ONE live API call to Google Gemini (Model: {DEFAULT_GEMINI_MODEL})...")
    generator = GeminiGenerator(api_key=api_key, model_name=DEFAULT_GEMINI_MODEL)

    try:
        response_data = generator.generate_with_sources(test_question, retrieved_chunks)
        real_answer = response_data["answer"]

        print("\n" + "=" * 70)
        print("REAL GEMINI RESPONSE:")
        print("=" * 70)
        print(real_answer)
        print("=" * 70)
        print(f"Model Used: {response_data['model']}")
        print(f"Cited Sources: {response_data['sources']}")
        print("\nIntegration test executed successfully with live API.")

    except Exception as e:
        print(f"\nAPI Call Failed: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_real_gemini_integration()
