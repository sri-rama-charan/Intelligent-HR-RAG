"""
PDF Loader module.
Responsible for reading PDF files from disk, extracting page text,
and attaching structured metadata (source filename, page number, document title).
"""

import os
from pathlib import Path
from typing import List, Optional
from pypdf import PdfReader
from langchain_core.documents import Document


def extract_title_from_filename(filename: str) -> str:
    """
    Derives a human-readable title from the PDF filename.
    Example: '02_Leave_Policy.pdf' -> 'Leave Policy'
    """
    stem = Path(filename).stem
    # Remove leading numeric prefix if present (e.g. '02_')
    parts = stem.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        title = parts[1].replace("_", " ")
    else:
        title = stem.replace("_", " ")
    return title.strip()


def load_single_pdf(file_path: Path) -> List[Document]:
    """
    Loads a single PDF and returns a list of Document objects,
    with one Document per page.

    Args:
        file_path (Path): Path to the PDF file.

    Returns:
        List[Document]: List of documents representing individual pages.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found at: {file_path}")

    documents = []
    reader = PdfReader(str(file_path))
    total_pages = len(reader.pages)
    title = extract_title_from_filename(file_path.name)

    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1  # 1-indexed for human readability
        text = page.extract_text() or ""
        
        # Clean up text slightly: normalize carriage returns
        text = text.replace("\r\n", "\n").strip()

        # Build metadata dictionary to accompany this page
        metadata = {
            "source": file_path.name,
            "page": page_num,
            "total_pages": total_pages,
            "document_title": title,
            "file_path": str(file_path.resolve())
        }

        # Only include if text was extracted (skip completely blank pages if any)
        if text:
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)

    return documents


def load_all_pdfs(corpus_dir: Optional[Path] = None) -> List[Document]:
    """
    Loads all PDF files from the corpus directory.

    Args:
        corpus_dir (Optional[Path]): Directory containing PDFs.
            Defaults to 'data/hr_corpus'.

    Returns:
        List[Document]: List of all page-level documents across all PDFs.
    """
    if corpus_dir is None:
        # Default to data/hr_corpus relative to project root
        project_root = Path(__file__).resolve().parent.parent.parent
        corpus_dir = project_root / "data" / "hr_corpus"
        
        # Fallback to project-2-intelligent-rag/zyro-dynamics-hr-corpus if data/hr_corpus is missing
        if not corpus_dir.exists():
            corpus_dir = project_root / "project-2-intelligent-rag" / "zyro-dynamics-hr-corpus"

    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found at: {corpus_dir}")

    pdf_files = sorted([f for f in corpus_dir.iterdir() if f.suffix.lower() == ".pdf"])
    if not pdf_files:
        raise ValueError(f"No PDF files found in directory: {corpus_dir}")

    all_page_docs: List[Document] = []
    for pdf_file in pdf_files:
        docs = load_single_pdf(pdf_file)
        all_page_docs.extend(docs)

    return all_page_docs
