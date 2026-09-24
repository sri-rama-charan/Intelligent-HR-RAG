"""
Inspection and experimentation script for Phase 2B: Embeddings & Vector Representation.
Loads the 107 chunks, converts them into 384-dimensional vectors,
and performs a manual cosine similarity experiment for a sample query.
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


def run_embedding_inspection():
    print("=" * 70)
    print("PHASE 2B: EMBEDDINGS & VECTOR REPRESENTATION INSPECTION")
    print("=" * 70)

    # 1. Load documents and generate chunks
    print("\n[Step 1] Loading PDFs and generating chunks...")
    pages = load_all_pdfs()
    chunks = chunk_documents(pages, chunk_size=800, chunk_overlap=100)
    print(f"Total chunks created: {len(chunks)}")

    # 2. Initialize embedding manager
    print("\n[Step 2] Initializing SentenceTransformer embedding model...")
    embedder = EmbeddingManager(model_name="all-MiniLM-L6-v2")

    # 3. Generate embeddings for all chunks
    print("\n[Step 3] Generating embeddings for all chunks...")
    vectors, mapped_chunks = embedder.embed_chunks(chunks, normalize=True)

    print(f"Number of chunks:        {len(mapped_chunks)}")
    print(f"Number of vectors:       {len(vectors)}")
    print(f"Embedding dimension:     {embedder.dimension}")
    print(f"Embedding matrix shape:  {vectors.shape}")

    # 4. Preview sample chunks and their numerical vectors
    print("\n[Step 4] Previewing Sample Chunks & Vector Representations:")
    
    sample_indices = [10, 25]  # Take two distinct chunks
    for idx in sample_indices:
        chunk = mapped_chunks[idx]
        vec = vectors[idx]
        
        # Format first 6 numbers for preview
        preview_vals = [f"{v:.4f}" for v in vec[:6]]
        preview_str = f"[{', '.join(preview_vals)}, ... ({len(vec)} total dimensions)]"

        print("-" * 70)
        print(f"Chunk ID: {chunk.metadata['chunk_id']}")
        print(f"Source:   {chunk.metadata['source']}")
        print(f"Page:     {chunk.metadata['page']}")
        print(f"Text Snippet:\n  \"{chunk.page_content[:180].replace(chr(10), ' ')}...\"")
        print(f"Vector Preview: {preview_str}")
    print("-" * 70)

    # 5. Semantic Similarity Experiment (WITHOUT FAISS)
    test_query = "How many casual leaves can an employee take?"
    print(f"\n[Step 5] Educational Similarity Experiment")
    print(f"User Query: \"{test_query}\"")
    print("-" * 70)

    # Embed query using the EXACT SAME model and normalization
    query_vector = embedder.embed_query(test_query, normalize=True)
    print(f"Query vector generated with shape: {query_vector.shape}")

    # Compute cosine similarity manually using dot product
    # Note: Because both vectors and query_vector are L2-normalized (length=1),
    # the dot product directly equals the cosine similarity!
    cosine_similarities = np.dot(vectors, query_vector)

    # Find the top 5 indices with highest similarity scores
    top_5_indices = np.argsort(cosine_similarities)[::-1][:5]

    print("\nTop 5 Most Semantically Similar Chunks:")
    print("=" * 70)

    for rank, idx in enumerate(top_5_indices, start=1):
        score = cosine_similarities[idx]
        matched_chunk = mapped_chunks[idx]
        
        print(f"Rank {rank}")
        print(f"Similarity Score: {score:.4f}")
        print(f"Source:           {matched_chunk.metadata['source']}")
        print(f"Page:             {matched_chunk.metadata['page']}")
        print(f"Chunk ID:         {matched_chunk.metadata['chunk_id']}")
        print("Text Content:")
        # Indent text slightly for readability
        lines = matched_chunk.page_content.strip().split("\n")
        for line in lines[:6]:  # Show first 6 lines
            print(f"  {line}")
        if len(lines) > 6:
            print("  ...")
        print("-" * 70)

    print("\nEmbedding inspection and experiment completed successfully.\n")


if __name__ == "__main__":
    run_embedding_inspection()
