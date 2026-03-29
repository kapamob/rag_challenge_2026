import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any
from llama_index.core.schema import Document
import re

class AdvancedPDFReader:
    """
    Advanced PDF Reader that extracts text using PyMuPDF (fitz).
    It captures page numbers and tries to organize text better than naive readers.
    """
    
    def __init__(self, include_page_numbers: bool = True):
        self.include_page_numbers = include_page_numbers
        
    def load_data(self, file_path: Path) -> List[Document]:
        """Load data from a PDF file."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        doc_id = file_path.name
        documents = []
        
        try:
            pdf_doc = fitz.open(str(file_path))
            
            for page_num in range(len(pdf_doc)):
                page = pdf_doc.load_page(page_num)
                # get_text("blocks") returns [(x0, y0, x1, y1, "lines in block", block_no, block_type), ...]
                blocks = page.get_text("blocks")
                
                # Sort blocks top-to-bottom, left-to-right
                blocks.sort(key=lambda b: (b[1], b[0]))
                
                text_content = []
                for b in blocks:
                    if b[6] == 0:  # block_type 0 is text
                        text = b[4].strip()
                        if text:
                            # Clean up text
                            text = re.sub(r'\s+', ' ', text)
                            text_content.append(text)
                            
                page_text = "\n\n".join(text_content)
                
                if page_text.strip():
                    # 1-indexed page matching Evaluation format
                    metadata = {
                        "doc_id": doc_id,
                        "page_number": page_num + 1,
                        "file_name": file_path.name
                    }
                    
                    doc = Document(
                        text=page_text,
                        metadata=metadata,
                        excluded_embed_metadata_keys=["page_number", "doc_id", "file_name"],
                        excluded_llm_metadata_keys=["file_name"]
                    )
                    documents.append(doc)
            
            pdf_doc.close()
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
        return documents

if __name__ == "__main__":
    # Test
    reader = AdvancedPDFReader()
    docs_dir = Path("./docs_corpus")
    pdf_files = list(docs_dir.glob("*.pdf"))
    if pdf_files:
        print(f"Testing with {pdf_files[0]}")
        docs = reader.load_data(pdf_files[0])
        print(f"Extracted {len(docs)} pages")
        if docs:
            print(f"Metadata: {docs[0].metadata}")
            print(f"Text sample:\n{docs[0].text[:200]}...")
