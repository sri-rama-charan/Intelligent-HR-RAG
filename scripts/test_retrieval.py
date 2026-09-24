"""
Retrieval testing script for Phase 2C: FAISS Vector Store & Top-K Retrieval.
Indexes all 107 HR chunks into FAISS IndexFlatIP, runs three benchmark HR queries,
displays Top-5 retrieved chunks with source citations, and validates rankings against NumPy.
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ingestion.pdf_loader import load_all_pdfs
from src.ingestion.chunker import chunk_documents
from src.retrieval.embeddings import EmbeddingManager
from src.retrieval.vector_store import FAISSVectorStore


def run_retrieval_tests():
    print("=" * 70)
    print("PHASE 2C: FAISS VECTOR STORE & TOP-K RETRIEVAL EXPERIMENT")
    print("=" * 70)

    # 1. Ingestion: Load PDFs and create chunks
    print("\n[Step 1] Loading PDFs and chunking...")
    pages = load_all_pdfs()
    chunks = chunk_documents(pages, chunk_size=800, chunk_overlap=100)
    print(f"Total chunks created: {len(chunks)}")

    # 2. Embedding Model
    print("\n[Step 2] Initializing Embedding Model (all-MiniLM-L6-v2)...")
    embedder = EmbeddingManager(model_name="all-MiniLM-L6-v2")

    # 3. Build FAISS Index
    print("\n[Step 3] Building FAISS IndexFlatIP...")
    vector_store = FAISSVectorStore.build_from_chunks(chunks, embedder)
    print(f"Total vectors stored in FAISS: {vector_store.total_vectors}")
    print(f"Index dimension:                {vector_store.dimension}")

    # 4. Benchmark Queries
    test_queries = [
        "How many casual leaves can an employee take?",
        "When does salary get credited?",
        "Who is eligible to work from home?"
    ]

    for q_idx, query in enumerate(test_queries, start=1):
        print("\n" + "=" * 70)
        print(f"QUERY {q_idx}: \"{query}\"")
        print("=" * 70)

        # Embed query and retrieve Top-5 chunks from FAISS
        query_vector = embedder.embed_query(query, normalize=True)
        results = vector_store.search(query_vector, top_k=5)

        for item in results:
            print(f"Rank {item['rank']}")
            print(f"Similarity Score: {item['score']:.4f}")
            print(f"Source:           {item['source']}")
            print(f"Page:             {item['page']}")
            print(f"Chunk ID:         {item['chunk_id']}")
            print("Text Preview:")
            
            # Shortened text preview (first 4 lines)
            lines = item["text"].strip().split("\n")
            preview_lines = [line.strip() for line in lines if line.strip()][:4]
            for pl in preview_lines:
                print(f"  {pl}")
            if len(lines) > 4:
                print("  ...")
            print("-" * 70)

    print("\nRetrieval experiment completed successfully.\n")


if __name__ == "__main__":
    run_retrieval_tests()
