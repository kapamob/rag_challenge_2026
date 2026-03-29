import os
import sys
from pathlib import Path
import fitz # PyMuPDF
import pdfplumber
from docling.document_converter import DocumentConverter

def compare_extractors(pdf_path: str):
    print(f"\n--- Comparing: {Path(pdf_path).name} ---")
    
    # 1. PyMuPDF
    try:
        doc = fitz.open(pdf_path)
        pymupdf_text = ""
        for page in doc:
            pymupdf_text += page.get_text()
    except Exception as e:
        pymupdf_text = f"Error: {e}"
    
    # 2. pdfplumber
    try:
        plumber_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                plumber_text += page.extract_text() or ""
    except Exception as e:
        plumber_text = f"Error: {e}"
            
    # 3. Docling
    try:
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        docling_text = result.document.export_to_markdown()
    except Exception as e:
        docling_text = f"Error: {e}"

    # Metrics
    len_py = len(pymupdf_text)
    len_pl = len(plumber_text)
    len_dl = len(docling_text)
    
    print(f"PyMuPDF length: {len_py}")
    print(f"pdfplumber length: {len_pl}")
    print(f"Docling length: {len_dl}")
    
    # Sample check for tables (Markdown tables often use |)
    table_count = docling_text.count("|")
    print(f"Docling potential table chars (|): {table_count}")

    if abs(len_py - len_dl) > 500:
        print(f"Significant difference detected with Docling: {len_dl - len_py} chars")
        # Save snippets to /tmp for analysis
        base = Path(pdf_path).stem
        with open(f"/tmp/{base}_pymupdf.txt", "w") as f: f.write(pymupdf_text[:2000])
        with open(f"/tmp/{base}_docling.md", "w") as f: f.write(docling_text[:2000])
        print(f"Saved snippets to /tmp/{base}_* for comparison.")

if __name__ == "__main__":
    docs_dir = Path("./docs_corpus")
    # Take 3 samples to save time (Docling is slow)
    samples = list(docs_dir.glob("*.pdf"))[:3]
    for sample in samples:
        compare_extractors(str(sample))
