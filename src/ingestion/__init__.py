"""
Ingestion module: responsible for loading PDFs, extracting text and metadata, and chunking documents.
"""
from src.ingestion.pdf_loader import load_all_pdfs, load_single_pdf
from src.ingestion.chunker import chunk_documents

__all__ = ["load_all_pdfs", "load_single_pdf", "chunk_documents"]
