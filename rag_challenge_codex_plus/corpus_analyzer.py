import os
import json
import csv
import sys
from pathlib import Path
from typing import List, Dict, Any
import re

# Constants
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    Document
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from arlc import get_config

# Load config
CONFIG = get_config()

# Global Settings
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = OpenAILike(
    model="openai/gpt-4o-mini",
    api_key=os.environ.get("OPENROUTER_API_KEY") or CONFIG.get("openrouter_api_key"),
    api_base="https://openrouter.ai/api/v1",
    is_chat_model=True,
    max_tokens=512,
)

METADATA_PROMPT = """
You are an expert legal clerk. Analyze the following text from the first pages of a legal document and extract metadata in JSON format.

Text:
{text}

JSON Format:
{{
  "document_number": "Case Number or ID (e.g., CFI 057/2025). If Law, use Law number (e.g., Law No. 3 of 2018)",
  "date": "Document issue date in YYYY-MM-DD format",
  "title": "Short descriptive title of the document",
  "type": "Judgment / Law / Order / Enforcement Document / Other",
  "claimant": "Name of the claimant or plaintiff (if applicable)",
  "defendant": "Name of the defendant (if applicable)"
}}

If a field is not found, use "null".
"""

def extract_metadata(text: str) -> Dict[str, Any]:
    try:
        response = Settings.llm.complete(METADATA_PROMPT.format(text=text[:4000])).text.strip()
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        print(f"Error extracting metadata: {e}")
    return {}

def analyze_corpus():
    docs_dir = Path(CONFIG.docs_dir)
    pdf_files = list(docs_dir.glob("*.pdf"))
    
    metadata_list = []
    
    # Store indices in a dedicated folder
    index_storage_root = ROOT_DIR / "document_indices"
    index_storage_root.mkdir(exist_ok=True)

    print(f"Analyzing {len(pdf_files)} documents...")

    for i, pdf_file in enumerate(pdf_files):
        print(f"[{i+1}/{len(pdf_files)}] Processing {pdf_file.name}...")
        
        # 1. Read first few pages for metadata
        reader = SimpleDirectoryReader(input_files=[pdf_file])
        documents = reader.load_data()
        
        full_text = "\n\n".join([doc.text for doc in documents])
        metadata = extract_metadata(full_text)
        
        # Add internal metadata
        metadata['file_name'] = pdf_file.name
        metadata['db_name'] = pdf_file.stem
        
        metadata_list.append(metadata)
        
        # 2. Create and save index for this document
        storage_path = index_storage_root / pdf_file.stem
        if not storage_path.exists():
            storage_path.mkdir()
            index = VectorStoreIndex.from_documents(documents)
            index.storage_context.persist(persist_dir=str(storage_path))
            print(f"  - Index created and persisted to {storage_path.name}")
        else:
            print(f"  - Index already exists for {storage_path.name}")

    # 3. Save metadata to CSV
    csv_path = ROOT_DIR / "document_metadata.csv"
    keys = ["db_name", "file_name", "document_number", "date", "title", "type", "claimant", "defendant"]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metadata_list)
    
    # 4. Save metadata to JSON for programmatic access
    with open(ROOT_DIR / "document_metadata.json", 'w') as f:
        json.dump(metadata_list, f, indent=2)

    print(f"Analysis complete! Metadata saved to {csv_path}")

if __name__ == "__main__":
    if not os.environ.get("OPENROUTER_API_KEY") and not CONFIG.get("openrouter_api_key"):
        print("Error: OPENROUTER_API_KEY must be set.")
        sys.exit(1)
    analyze_corpus()
