import os
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import tiktoken
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.node_parser import SentenceSplitter

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from arlc import (  # noqa: E402
    EvaluationClient,
    RetrievalRef,
    SubmissionAnswer,
    SubmissionBuilder,
    Telemetry,
    TelemetryTimer,
    TimingMetrics,
    UsageMetrics,
    get_config,
    normalize_retrieved_pages,
)
from examples.llamaindex.advanced_pdf_reader import AdvancedPDFReader

CONFIG = get_config()
TOKENIZER = tiktoken.get_encoding("cl100k_base")

Settings.llm = OpenAILike(
    model="deepseek-chat",
    api_key=CONFIG.get_llm_api_key(),
    api_base=CONFIG.llm_api_base,
    is_chat_model=True,
    max_tokens=2048,
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Advanced chunking settings matching bge-small context window
Settings.chunk_size = 512
Settings.chunk_overlap = 64
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)


def download_resources(client: EvaluationClient) -> list[dict]:
    """Download questions and documents via the API"""
    print("Downloading questions...")
    questions = client.download_questions(target_path=CONFIG.questions_path)
    print("Downloading documents...")
    client.download_documents(CONFIG.docs_dir)
    print("Documents extracted")
    return questions


def extract_retrieval_refs(nodes: list) -> list[RetrievalRef]:
    """
    Build page references from LlamaIndex metadata using AdvancedPDFReader schema.
    doc_id must match mapping.json keys (hash without .pdf extension).
    page_numbers must be 1-indexed per evaluation spec.
    """
    raw_refs: list[RetrievalRef] = []
    for node in nodes:
        if not hasattr(node, "metadata"):
            continue
        metadata = node.metadata or {}
        
        doc_id = metadata.get("doc_id")
        if not doc_id:
            continue
            
        doc_id = Path(doc_id).stem if "." in doc_id else doc_id
        
        # AdvancedPDFReader sets 'page_number' as 1-indexed explicitly
        try:
            page_number = int(metadata.get("page_number", 0))
            if page_number < 1:
                continue
        except (TypeError, ValueError):
            continue
            
        raw_refs.append(RetrievalRef(doc_id=doc_id, page_numbers=[page_number]))
        
    return normalize_retrieved_pages(raw_refs)


def _parse_answer_by_type(raw: str, answer_type: str):
    """Parse string response according to answer_type."""
    text = (raw or "").strip()
    at = str(answer_type or "free_text").lower()
    
    if text.lower() in ["null", "none", "not found", "unanswerable"]:
        return None
        
    if at == "number":
        try:
            return float(text.replace(",", "."))
        except (TypeError, ValueError):
            return text
    if at == "boolean":
        lower = text.lower()
        if lower in ("true", "yes", "1"):
            return True
        if lower in ("false", "no", "0"):
            return False
        return text
    if at == "date":
        return text
    if at == "names":
        parts = [p for p in (x.strip() for x in text.replace(",", ";").split(";")) if p]
        return parts if parts else [text]
    if at == "null":
        return None
    return text


_TYPE_INSTRUCTIONS: dict[str, str] = {
    "number":    "Return only the numeric value (integer or decimal). No units, no explanation.",
    "boolean":   "Return only 'true' or 'false'. No explanation.",
    "name":      "Return only the exact name or entity as it appears in the documents. No explanation.",
    "names":     "Return a semicolon-separated list of names only. No explanation.",
    "date":      "Return the date in YYYY-MM-DD format only. No explanation.",
    "free_text": "Answer in full sentences with reasoning grounded strictly in the context.",
}


def build_prompt(context: str, question_text: str, answer_type: str = "free_text") -> str:
    """Build a prompt for the model, including answer-type formatting instructions."""
    instruction = _TYPE_INSTRUCTIONS.get(answer_type, _TYPE_INSTRUCTIONS["free_text"])
    return (
        f"You are a legal document assistant. Answer the following question based ONLY on the provided Context.\n"
        f"If the answer is not contained in the context, explicitly output 'null'.\n"
        f"{instruction}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question_text}\n\n"
        "Answer:"
    )


_EXCLUDE_DIRS = {"__pycache__", "docs_corpus", "storage", ".venv", "venv", "env"}
_EXCLUDE_FILES = {".env", "submission.json", "questions.json", "code_archive.zip"}


def ensure_code_archive(archive_path: Path) -> Path:
    """Archive the entire starter_kit directory, excluding generated/runtime artifacts."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_resolved = archive_path.resolve()
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as zip_file:
        for file_path in ROOT_DIR.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.resolve() == archive_resolved:
                continue
            parts = set(file_path.relative_to(ROOT_DIR).parts)
            if parts & _EXCLUDE_DIRS:
                continue
            if file_path.name in _EXCLUDE_FILES:
                continue
            zip_file.write(file_path, file_path.relative_to(ROOT_DIR))
    return archive_path


def main() -> None:
    """Run the advanced pipeline and build submission."""
    client = EvaluationClient.from_env()
    questions = download_resources(client)
    print(f"Loaded {len(questions)} questions")

    print("Loading documents using AdvancedPDFReader (PyMuPDF)...")
    docs_dir = Path(CONFIG.docs_dir)
    pdf_files = list(docs_dir.glob("**/*.pdf"))
    
    documents = []
    reader = AdvancedPDFReader(include_page_numbers=True)
    for pdf_file in pdf_files:
        documents.extend(reader.load_data(pdf_file))
        
    print(f"Loaded {len(documents)} document pages")

    print("Creating index with Advanced Chunking (size=1024, overlap=128)...")
    # LlamaIndex automatically uses Settings.node_parser and Settings.embed_model
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    
    # Advanced Retrieval: increased top_k from 3 to 5 for now to balance speed
    retriever = index.as_retriever(similarity_top_k=5)

    print("\nAnswering questions...")
    builder = SubmissionBuilder(
        architecture_summary="Advanced RAG: PyMuPDF parsing, Chunking (1024/128), Top-K=5, DeepSeek Chat, proper mapping",
    )
    for index_number, question_item in enumerate(questions, 1):
        question_text = question_item["question"]
        question_id = question_item["id"]
        print(f"[{index_number}/{len(questions)}] {question_id}")

        nodes = retriever.retrieve(question_text)
        context = "\n\n".join([node.text for node in nodes])
        prompt = build_prompt(context, question_text, question_item.get("answer_type", "free_text"))

        telemetry_timer = TelemetryTimer()
        telemetry_timer.mark_token() # start marking
        response = Settings.llm.complete(prompt)
        response_text = response.text
        telemetry_timer.mark_token() # end marking

        answer = _parse_answer_by_type(response_text, question_item.get("answer_type", "free_text"))
        
        # Proper handling of unanswerable cases for Grounding optimization
        retrieval_refs = extract_retrieval_refs(nodes)
        if answer is None:
            # If the LLM determined 'null' (not found), set retrieved chunk pages to empty
            # According to EVALUATION.md: "For unanswerable questions: set retrieved_chunk_pages to an empty array []... yield grounding 1.0"
            retrieval_refs = []
            
        timing = telemetry_timer.finish()
        usage = UsageMetrics(
            input_tokens=len(TOKENIZER.encode(prompt)),
            output_tokens=len(TOKENIZER.encode(response_text)),
        )
        telemetry = Telemetry(
            timing=TimingMetrics(
                ttft_ms=timing.ttft_ms,
                tpot_ms=timing.tpot_ms,
                total_time_ms=timing.total_time_ms,
            ),
            retrieval=retrieval_refs,
            usage=usage,
            model_name="deepseek-chat",
        )
        builder.add_answer(
            SubmissionAnswer(
                question_id=question_id,
                answer=answer,
                telemetry=telemetry,
            )
        )

    submission_path = builder.save(ROOT_DIR / "submission.json")
    code_archive_path = ensure_code_archive(ROOT_DIR / "code_archive.zip")
    print(f"\nSaved {submission_path}")
    print(f"Code archive: {code_archive_path}")


if __name__ == "__main__":
    main()
