"""
Inspection script to verify chunking behavior across all 11 HR policy documents.
Displays overall statistics (pages, total chunks, chunks per document)
and prints representative chunks with metadata from different policy areas.
"""

import sys
from pathlib import Path
from collections import defaultdict

# Add project root to sys.path so 'src' can be imported directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ingestion.pdf_loader import load_all_pdfs
from src.ingestion.chunker import chunk_documents


def inspect_chunks():
    print("=" * 70)
    print("PHASE 2A: DOCUMENT INGESTION & CHUNKING INSPECTION")
    print("=" * 70)

    # 1. Load all PDFs
    print("\n[Step 1] Loading all PDF documents...")
    pages = load_all_pdfs()
    print(f"Total page documents extracted: {len(pages)}")

    # 2. Chunk all documents
    print("\n[Step 2] Chunking documents (chunk_size=800, chunk_overlap=100)...")
    chunks = chunk_documents(pages, chunk_size=800, chunk_overlap=100)
    print(f"Total chunks generated: {len(chunks)}")

    # 3. Compute stats per document
    chunks_per_doc = defaultdict(list)
    pages_per_doc = defaultdict(set)
    for c in chunks:
        doc_name = c.metadata["source"]
        chunks_per_doc[doc_name].append(c)
        pages_per_doc[doc_name].add(c.metadata["page"])

    print("\n[Step 3] Breakdown per Document:")
    print("-" * 70)
    print(f"{'Document Name':<45} | {'Pages':<6} | {'Chunks':<6}")
    print("-" * 70)
    for doc_name, doc_chunks in sorted(chunks_per_doc.items()):
        num_pages = len(pages_per_doc[doc_name])
        num_chunks = len(doc_chunks)
        print(f"{doc_name:<45} | {num_pages:<6} | {num_chunks:<6}")
    print("-" * 70)

    # 4. Display representative chunks from various documents
    print("\n[Step 4] Representative Chunks from Different Policies:")
    
    # Pick sample chunks from different documents:
    # 1. Leave Policy
    # 2. Work From Home Policy
    # 3. Compensation and Benefits Policy
    # 4. IT and Data Security Policy
    # 5. POSH Policy
    target_docs = [
        "02_Leave_Policy.pdf",
        "03_Work_From_Home_Policy.pdf",
        "06_Compensation_and_Benefits_Policy.pdf",
        "07_IT_and_Data_Security_Policy.pdf",
        "08_Prevention_of_Sexual_Harassment_Policy.pdf",
    ]

    for target_doc in target_docs:
        doc_chunks = chunks_per_doc.get(target_doc, [])
        if not doc_chunks:
            continue
        
        # Select a middle chunk that usually contains specific policy rules
        sample_chunk = doc_chunks[min(1, len(doc_chunks) - 1)]
        
        print("\n" + "=" * 70)
        print(f"Chunk ID: {sample_chunk.metadata['chunk_id']}")
        print(f"Source:   {sample_chunk.metadata['source']}")
        print(f"Page:     {sample_chunk.metadata['page']} of {sample_chunk.metadata.get('total_pages', 'N/A')}")
        print(f"Length:   {sample_chunk.metadata['chunk_size']} characters")
        print("=" * 70)
        print(sample_chunk.page_content)
        print("=" * 70)

    print("\nChunk inspection completed successfully.\n")


if __name__ == "__main__":
    inspect_chunks()
