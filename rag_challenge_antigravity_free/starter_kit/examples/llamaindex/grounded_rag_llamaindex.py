import json
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


def _parse_answer_by_type(raw: str, answer_type: str):
    """Parse string response according to answer_type."""
    text = (raw or "").strip()
    at = str(answer_type or "free_text").lower()
    
    if text.lower() in ["null", "none", "not found", "unanswerable", ""]:
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
        # Do not allow aliases as per Q&A
        parts = [p for p in (x.strip() for x in text.replace(",", ";").split(";")) if p]
        return parts if parts else [text]
    if at == "null":
        return None
    return text


_TYPE_INSTRUCTIONS: dict[str, str] = {
    "number":    "Return only the numeric value (integer or float). No units, no explanation.",
    "boolean":   "Return only 'true' or 'false'. No explanation.",
    "name":      "Return only the exact name or entity as it appears in the documents. No explanation. No aliases.",
    "names":     "Return a semicolon-separated list of exact names only. No explanation. No aliases.",
    "date":      "Return the date in YYYY-MM-DD format only. No explanation.",
    "free_text": "Answer in full sentences with reasoning grounded strictly in the context.",
}


def build_context_with_metadata(nodes: list) -> str:
    """Format nodes with explicit metadata for LLM attribution"""
    context_parts = []
    for i, node in enumerate(nodes):
        metadata = node.metadata or {}
        doc_id = metadata.get("doc_id", "unknown")
        doc_id = Path(doc_id).stem if "." in doc_id else doc_id
        page = metadata.get("page_number", "unknown")
        
        context_parts.append(f"--- Source {i+1} ---\n[doc_id: {doc_id}, page_numbers: [{page}]]\n{node.text}")
    return "\n\n".join(context_parts)


def build_prompt(context: str, question_text: str, answer_type: str = "free_text") -> str:
    """Build a prompt for the model, forcing JSON output for answer and sources attribution."""
    instruction = _TYPE_INSTRUCTIONS.get(answer_type, _TYPE_INSTRUCTIONS["free_text"])
    return (
        f"You are a strict legal document assistant. Answer the following question based ONLY on the provided Context.\n"
        f"You MUST return your entire response as a single valid JSON object with exactly two keys: 'answer' and 'sources'.\n\n"
        f"- 'answer': {instruction}\n"
        f"  If the answer cannot be found in the Context, explicitly set 'answer' to null.\n"
        f"- 'sources': A list of objects representing the exact sources used to formulate the answer. "
        f"Extract this from the [doc_id: ..., page_numbers: ...] headers in the Context. "
        f"Each object must have 'doc_id' (string) and 'page_numbers' (list of integers). "
        f"If the answer spans multiple pages or documents, list all of them. If the answer is null, set 'sources' to [].\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question_text}\n\n"
        "Return ONLY the JSON string. Do not use markdown blocks like ```json."
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
    """Run the grounded pipeline and build submission."""
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

    print("Creating index with Advanced Chunking (size=512, overlap=64)...")
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    retriever = index.as_retriever(similarity_top_k=7)

    print("\nAnswering questions...")
    builder = SubmissionBuilder(
        architecture_summary="Grounded RAG: LLM JSON Attribution, Streaming TTFT Fix, Chunking 512, Top-K=7",
    )
    for index_number, question_item in enumerate(questions, 1):
        question_text = question_item["question"]
        question_id = question_item["id"]
        print(f"[{index_number}/{len(questions)}] {question_id}")

        nodes = retriever.retrieve(question_text)
        context = build_context_with_metadata(nodes)
        prompt = build_prompt(context, question_text, question_item.get("answer_type", "free_text"))

        telemetry_timer = TelemetryTimer()
        
        # TTFT Fix: switch from complete() to stream_complete() to correctly log the first token
        response_text = ""
        try:
            for chunk in Settings.llm.stream_complete(prompt):
                telemetry_timer.mark_token()  # The first loop execution accurately captures TTFT
                response_text += chunk.delta
        except Exception as e:
            print(f"Error during streaming LLM generation: {e}")
            telemetry_timer.mark_token() # fallback
            
        telemetry_timer.mark_token() # End of stream marker
        timing = telemetry_timer.finish()

        # Parse JSON output for Answer + Attribution
        answer = None
        retrieval_refs = []
        
        try:
            clean_json = response_text.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json[7:-3].strip()
            elif clean_json.startswith("```"):
                clean_json = clean_json[3:-3].strip()
                
            data = json.loads(clean_json)
            raw_answer = data.get("answer")
            answer = _parse_answer_by_type(str(raw_answer) if raw_answer is not None else "null", question_item.get("answer_type", "free_text"))
            
            raw_sources = data.get("sources", [])
            for src in raw_sources:
                doc_id = src.get("doc_id")
                pages = src.get("page_numbers", [])
                if doc_id and pages:
                    retrieval_refs.append(RetrievalRef(doc_id=str(doc_id), page_numbers=pages))
                    
        except json.JSONDecodeError as e:
            print(f"Warning: JSON decode error on {question_id}: {e}\nRaw fallback parsing initiated.")
            answer = _parse_answer_by_type(response_text, question_item.get("answer_type", "free_text"))
            # If JSON parsing fails, fallback to sending everything we retrieved (pessimistic)
            for node in nodes:
                metadata = node.metadata or {}
                doc_id = metadata.get("doc_id")
                page = metadata.get("page_number", 0)
                if doc_id and page:
                    doc_id = Path(doc_id).stem if "." in doc_id else doc_id
                    retrieval_refs.append(RetrievalRef(doc_id=str(doc_id), page_numbers=[int(page)]))

        retrieval_refs = normalize_retrieved_pages(retrieval_refs)
        if answer is None:
            retrieval_refs = []  # Explicitly clear on null answers to maximize Grounding Metric

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
