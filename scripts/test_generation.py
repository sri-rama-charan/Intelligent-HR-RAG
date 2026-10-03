"""
Generation testing script for Phase 2D: Generation & Gemini LLM Integration.
Executes the retrieval-to-generation pipeline for 3 key HR questions plus
1 grounding/hallucination test on an uncovered topic (pet insurance).
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
from src.generation.prompts import build_grounded_prompt
from src.generation.gemini_generator import GeminiGenerator, DEFAULT_GEMINI_MODEL


def run_generation_tests():
    print("=" * 70)
    print("PHASE 2D: GENERATION & GEMINI LLM INTEGRATION TEST")
    print("=" * 70)

    # 1. Initialize Retrieval Pipeline (FAISS + all-MiniLM-L6-v2)
    print("\n[Step 1] Loading corpus and building in-memory FAISS index...")
    pages = load_all_pdfs()
    chunks = chunk_documents(pages, chunk_size=800, chunk_overlap=100)
    embedder = EmbeddingManager(model_name="all-MiniLM-L6-v2")
    vector_store = FAISSVectorStore.build_from_chunks(chunks, embedder)
    print(f"Index built: {vector_store.total_vectors} chunks indexed.")

    # 2. Check for Gemini API Key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    generator = None
    if api_key:
        print(f"\n[Step 2] Gemini API Key detected. Initializing GeminiGenerator ({DEFAULT_GEMINI_MODEL})...")
        generator = GeminiGenerator(api_key=api_key, model_name=DEFAULT_GEMINI_MODEL)
    else:
        print("\n[Step 2] NOTE: GEMINI_API_KEY is not set in environment or .env file.")
        print("         The script will retrieve real chunks and display the exact grounded")
        print("         prompt prepared for Gemini.")

    # 3. Test Questions: 3 In-Scope + 1 Grounding/Hallucination Test
    test_questions = [
        {
            "id": "Test 1 (Salary Credit Date)",
            "query": "When does salary get credited?",
            "in_scope": True
        },
        {
            "id": "Test 2 (WFH Eligibility)",
            "query": "Who is eligible to work from home?",
            "in_scope": True
        },
        {
            "id": "Test 3 (Casual Leave Limit)",
            "query": "How many casual leaves can an employee take?",
            "in_scope": True
        },
        {
            "id": "Test 4 (Grounding / Hallucination Test)",
            "query": "What is the company's policy on employee pet insurance?",
            "in_scope": False
        }
    ]

    for item in test_questions:
        print("\n" + "=" * 70)
        print(f"QUESTION [{item['id']}]: \"{item['query']}\"")
        print("=" * 70)

        # Retrieve Top-3 relevant chunks from FAISS
        query_vector = embedder.embed_query(item["query"], normalize=True)
        retrieved_results = vector_store.search(query_vector, top_k=3)

        print("\n--- Retrieved Sources Passed as Context ---")
        for res in retrieved_results:
            print(f"- [Rank {res['rank']}] {res['source']} (Page {res['page']}) | Similarity: {res['score']:.4f}")
            # Show a brief 2-line snippet of what the LLM sees
            snippet = res['text'].replace('\n', ' ').strip()[:140]
            print(f"  Snippet: \"{snippet}...\"")

        # Format the exact prompt
        prompt = build_grounded_prompt(item["query"], retrieved_results)

        # Call Gemini if API key is configured
        if generator:
            print("\n--- Gemini Generated Answer ---")
            try:
                answer = generator.generate(item["query"], retrieved_results)
                print(answer)
            except Exception as e:
                print(f"Error during generation: {e}")
        else:
            print("\n--- Prompt Preview (Ready to send to Gemini) ---")
            print(prompt[:400] + "\n  ... [Full context included] ...\n" + prompt[-120:])

    print("\n" + "=" * 70)
    print("Generation experiment completed.")
    print("=" * 70)


if __name__ == "__main__":
    run_generation_tests()
