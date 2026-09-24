"""
Document Chunker module.
Responsible for breaking page-level documents into smaller, coherent text chunks
with overlap, while preserving and enriching metadata (source, page, unique chunk_id).
"""

from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100


def get_text_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: Optional[List[str]] = None
) -> RecursiveCharacterTextSplitter:
    """
    Creates and configures a RecursiveCharacterTextSplitter.
    
    Why RecursiveCharacterTextSplitter?
    It attempts to split on natural boundaries first (paragraphs: '\\n\\n',
    then line breaks: '\\n', then words: ' ', then characters) so that sentences
    and policy clauses stay together rather than getting arbitrarily cut in half.
    """
    if separators is None:
        # Paragraphs -> Lines -> Sentences/Words -> Characters
        separators = ["\n\n", "\n", ". ", " ", ""]

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False
    )


def chunk_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Document]:
    """
    Splits a list of page-level Document objects into smaller chunks
    while preserving and attaching unique chunk metadata.

    Args:
        documents (List[Document]): The raw page-level documents from pdf_loader.
        chunk_size (int): Maximum character size for each chunk. Defaults to 800.
        chunk_overlap (int): Number of overlapping characters between adjacent chunks. Defaults to 100.

    Returns:
        List[Document]: List of chunked Document objects with unique chunk_id in metadata.
    """
    if not documents:
        return []

    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # We will track chunk counts per document-page to generate stable, readable IDs
    chunked_documents: List[Document] = []
    
    # Dictionary to keep track of chunk index per (source, page) pair
    chunk_counters = {}

    for doc in documents:
        # Split the single page document into smaller chunks
        raw_chunks = splitter.split_text(doc.page_content)
        
        source = doc.metadata.get("source", "unknown_doc")
        page = doc.metadata.get("page", 1)
        stem = Path(source).stem  # e.g. "02_Leave_Policy"

        for chunk_text in raw_chunks:
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            counter_key = (source, page)
            chunk_idx = chunk_counters.get(counter_key, 0)
            chunk_counters[counter_key] = chunk_idx + 1

            # Format a unique, human-readable chunk_id
            # Example: "02_Leave_Policy_p2_c0"
            chunk_id = f"{stem}_p{page}_c{chunk_idx}"

            # Copy parent metadata and attach chunk-specific metadata
            new_metadata = dict(doc.metadata)
            new_metadata.update({
                "chunk_id": chunk_id,
                "chunk_index": chunk_idx,
                "chunk_size": len(chunk_text),
                "configured_chunk_size": chunk_size,
                "configured_chunk_overlap": chunk_overlap
            })

            chunk_doc = Document(page_content=chunk_text, metadata=new_metadata)
            chunked_documents.append(chunk_doc)

    return chunked_documents
