import os
import sys
from pathlib import Path
from pypdf import PdfReader

def inspect_corpus(corpus_dir: Path):
    """
    Standalone script to inspect and verify PDF extraction for all HR policy documents.
    Validates page counts and character lengths, and prints a text sample from each document.
    """
    if not corpus_dir.exists():
        print(f"ERROR: Corpus directory not found at: {corpus_dir}")
        sys.exit(1)

    pdf_files = sorted([f for f in corpus_dir.iterdir() if f.suffix.lower() == ".pdf"])
    
    print(f"Found {len(pdf_files)} PDF files in: {corpus_dir}\n")
    
    success_count = 0
    fail_count = 0

    for pdf_path in pdf_files:
        print("=" * 60)
        print(f"File: {pdf_path.name}")
        
        try:
            reader = PdfReader(str(pdf_path))
            num_pages = len(reader.pages)
            
            # Extract text from all pages
            all_text = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                all_text.append(page_text.strip())
            
            full_text = "\n\n".join(all_text).strip()
            char_count = len(full_text)
            
            # Prepare a short, clean sample (first ~250 characters, normalized)
            sample = full_text[:250].replace("\r\n", "\n").strip()
            
            print(f"Pages: {num_pages}")
            print(f"Characters: {char_count}")
            print(f"\nSample:\n\"{sample}...\"")
            print("=" * 60 + "\n")
            
            success_count += 1
            
        except Exception as e:
            print(f"FAILED to extract: {e}")
            print("=" * 60 + "\n")
            fail_count += 1

    print(f"Inspection Summary: {success_count} succeeded, {fail_count} failed out of {len(pdf_files)} total files.")

if __name__ == "__main__":
    # Path to the HR corpus directory
    current_dir = Path(__file__).resolve().parent
    workspace_root = current_dir.parent
    corpus_directory = workspace_root / "project-2-intelligent-rag" / "zyro-dynamics-hr-corpus"
    
    inspect_corpus(corpus_directory)
