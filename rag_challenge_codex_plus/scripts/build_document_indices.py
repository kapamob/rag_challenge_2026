from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import fitz  # pymupdf
from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.openai import OpenAIEmbedding


def _configure_embedding(model: str) -> None:
    load_dotenv(Path(".env"))
    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    api_base = (os.getenv("OPENROUTER_API_BASE") or "https://openrouter.ai/api/v1").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required.")

    normalized_model = model.split("/", 1)[1] if "/" in model else model
    Settings.embed_model = OpenAIEmbedding(model=normalized_model, api_key=api_key, api_base=api_base)


def _pdf_to_documents(pdf_path: Path, doc_id: str) -> list[Document]:
    pdf = fitz.open(pdf_path)
    out: list[Document] = []
    for i in range(pdf.page_count):
        text = (pdf[i].get_text("text") or "").strip()
        out.append(
            Document(
                text=text,
                metadata={
                    "doc_id": doc_id,
                    "file_name": pdf_path.name,
                    "page_number": i + 1,
                },
            )
        )
    pdf.close()
    return out


def build_indices(
    docs_list_csv: Path,
    docs_dir: Path,
    out_dir: Path,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    rebuild: bool = False,
) -> None:
    _configure_embedding(embedding_model)
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(docs_list_csv.open("r", encoding="utf-8", newline="")))
    for idx, row in enumerate(rows, 1):
        doc_id = (row.get("doc_id") or "").strip()
        if not doc_id:
            continue
        persist_dir = out_dir / doc_id
        if persist_dir.exists() and not rebuild:
            print(f"[{idx}/{len(rows)}] skip {doc_id} (exists)")
            continue
        pdf_path = docs_dir / f"{doc_id}.pdf"
        if not pdf_path.exists():
            print(f"[{idx}/{len(rows)}] missing pdf for {doc_id}")
            continue

        print(f"[{idx}/{len(rows)}] indexing {doc_id}")
        docs = _pdf_to_documents(pdf_path, doc_id)
        nodes = splitter.get_nodes_from_documents(docs)
        index = VectorStoreIndex(nodes=nodes, show_progress=False)
        persist_dir.mkdir(parents=True, exist_ok=True)
        storage_context = StorageContext.from_defaults()
        storage_context.docstore = index.storage_context.docstore
        storage_context.index_store = index.storage_context.index_store
        storage_context.vector_stores = index.storage_context.vector_stores
        storage_context.persist(persist_dir=str(persist_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-document vector indices.")
    parser.add_argument("--docs-list", default="artifacts/docs_list/v1_regex/docs_list.csv")
    parser.add_argument("--docs-dir", default="cache/warmup/docs_corpus")
    parser.add_argument("--out-dir", default="artifacts/document_indices/v1_regex_small")
    parser.add_argument("--embedding-model", default="openai/text-embedding-3-small")
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    build_indices(
        docs_list_csv=Path(args.docs_list).resolve(),
        docs_dir=Path(args.docs_dir).resolve(),
        out_dir=Path(args.out_dir).resolve(),
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
