import json
import os
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import tiktoken
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

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

# Available Models Configuration
MODELS = {
    "deepseek": {
        "class": OpenAILike,
        "params": {
            "model": "deepseek-chat",
            "api_key": CONFIG.get_llm_api_key(),
            "api_base": CONFIG.llm_api_base,
            "is_chat_model": True,
            "max_tokens": 2048,
        }
    },
    "gpt-4o-mini": {
        "class": OpenAILike,
        "params": {
            "model": "openai/gpt-4o-mini",
            "api_key": os.environ.get("OPENROUTER_API_KEY") or CONFIG.openrouter_api_key,
            "api_base": "https://openrouter.ai/api/v1",
            "is_chat_model": True,
            "max_tokens": 2048,
        }
    },
    "claude-3-haiku": {
        "class": OpenAILike,
        "params": {
            "model": "anthropic/claude-3-haiku",
            "api_key": os.environ.get("OPENROUTER_API_KEY") or CONFIG.openrouter_api_key,
            "api_base": "https://openrouter.ai/api/v1",
            "is_chat_model": True,
            "max_tokens": 2048,
        }
    }
}

# DEFAULT MODEL
SELECTED_MODEL = os.environ.get("SELECTED_MODEL", "deepseek")

def get_llm():
    cfg = MODELS.get(SELECTED_MODEL, MODELS["deepseek"])
    return cfg["class"](**cfg["params"])

SUBMISSION_FILENAME = os.environ.get("SUBMISSION_FILENAME", "submission.json")

Settings.llm = get_llm()
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.chunk_size = 512
Settings.chunk_overlap = 64
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)

def download_resources(client: EvaluationClient) -> list[dict]:
    print("Downloading questions...")
    questions = client.download_questions(target_path=CONFIG.questions_path)
    print("Downloading documents...")
    client.download_documents(CONFIG.docs_dir)
    return questions

def _parse_answer_by_type(raw: str, answer_type: str):
    text = str(raw or "").strip()
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
        if lower in ("true", "yes", "1"): return True
        if lower in ("false", "no", "0"): return False
        return text
    if at == "names":
        parts = [p for p in (x.strip() for x in text.replace(",", ";").split(";")) if p]
        return parts if parts else [text]
    return text

_TYPE_INSTRUCTIONS: dict[str, str] = {
    "number":    "Return only the numeric value (integer or float).",
    "boolean":   "Return only 'true' or 'false'.",
    "name":      "Return only the exact name/entity. No aliases.",
    "names":     "Return a semicolon-separated list of exact names. No aliases.",
    "date":      "Return the date in YYYY-MM-DD format only.",
    "free_text": "Answer in full sentences grounded strictly in the context.",
}

def build_context_with_metadata(nodes: list) -> str:
    context_parts = []
    for i, node in enumerate(nodes):
        metadata = node.metadata or {}
        doc_id = Path(metadata.get("doc_id", "unknown")).stem
        page = metadata.get("page_number", "unknown")
        context_parts.append(f"Source {i+1} [doc_id: {doc_id}, page: {page}]:\n{node.text}")
    return "\n\n".join(context_parts)

def build_prompt(context: str, question_text: str, answer_type: str = "free_text") -> str:
    instruction = _TYPE_INSTRUCTIONS.get(answer_type, _TYPE_INSTRUCTIONS["free_text"])
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question_text}\n\n"
        f"Requirement: {instruction}\n"
        "You MUST respond in JSON format with two fields: 'answer' and 'sources'.\n"
        "'sources' is a list of objects like {'doc_id': '...', 'page_numbers': [...]}.\n"
        "If no answer is found, set 'answer' to null and 'sources' to [].\n"
        "JSON Response:"
    )

def main() -> None:
    client = EvaluationClient.from_env()
    questions = download_resources(client)
    
    docs_dir = Path(CONFIG.docs_dir)
    pdf_files = list(docs_dir.glob("**/*.pdf"))
    
    print("Loading documents...")
    reader = AdvancedPDFReader(include_page_numbers=True)
    documents = []
    for pdf_file in pdf_files:
        documents.extend(reader.load_data(pdf_file))
        
    print(f"Indexing {len(documents)} pages with Hybrid Search...")
    nodes = Settings.node_parser.get_nodes_from_documents(documents)
    
    # Vector Index
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)
    vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
    
    # BM25 Retriever
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)
    
    # Fusion Retriever (Vector + BM25)
    retriever = QueryFusionRetriever(
        [vector_index.as_retriever(similarity_top_k=5), bm25_retriever],
        num_queries=1, # no multi-query yet for speed
        mode="reciprocal_rerank",
        use_async=False, # DeepSeek might not like high async
    )

    print(f"Answering questions using {SELECTED_MODEL} (Model Class: {Settings.llm.__class__.__name__})")
    builder = SubmissionBuilder(architecture_summary=f"Hybrid RAG (Vector+BM25), Model: {SELECTED_MODEL}, Attribution via JSON")
    
    for i, q_item in enumerate(questions, 1):
        q_text, q_id = q_item["question"], q_item["id"]
        print(f"[{i}/{len(questions)}] {q_id}")

        retrieved_nodes = retriever.retrieve(q_text)
        context = build_context_with_metadata(retrieved_nodes)
        prompt = build_prompt(context, q_text, q_item.get("answer_type", "free_text"))

        timer = TelemetryTimer()
        response_text = ""
        try:
            for chunk in Settings.llm.stream_complete(prompt):
                timer.mark_token()
                response_text += chunk.delta
        except Exception as e:
            print(f"LLM Error: {e}")
            timer.mark_token()
            
        timer.mark_token()
        timing = timer.finish()

        ans = None
        refs = []
        try:
            # More robust JSON cleaning
            clean_json = response_text.strip()
            if "```" in clean_json:
                clean_json = clean_json.split("```")[1]
                if clean_json.startswith("json"): clean_json = clean_json[4:]
            clean_json = clean_json.strip()
            
            # Find the first { and last } to avoid junk
            start = clean_json.find("{")
            end = clean_json.rfind("}")
            if start != -1 and end != -1:
                clean_json = clean_json[start:end+1]
            
            data = json.loads(clean_json)
            ans = _parse_answer_by_type(data.get("answer"), q_item.get("answer_type"))
            refs = [RetrievalRef(doc_id=s["doc_id"], page_numbers=s["page_numbers"]) for s in data.get("sources", [])]
        except Exception as e:
            print(f"JSON Parse Error for {q_id}: {e}")
            ans = _parse_answer_by_type(response_text, q_item.get("answer_type"))
            # Fallback refs from retrieval metadata
            refs = [RetrievalRef(doc_id=Path(n.metadata.get("doc_id", "")).stem, page_numbers=[n.metadata.get("page_number", 1)]) for n in retrieved_nodes[:2]]

        if ans is None: refs = []
        
        telemetry = Telemetry(
            timing=TimingMetrics(ttft_ms=timing.ttft_ms, tpot_ms=timing.tpot_ms, total_time_ms=timing.total_time_ms),
            retrieval=normalize_retrieved_pages(refs),
            usage=UsageMetrics(input_tokens=len(TOKENIZER.encode(prompt)), output_tokens=len(TOKENIZER.encode(response_text))),
            model_name=SELECTED_MODEL
        )
        builder.add_answer(SubmissionAnswer(question_id=q_id, answer=ans, telemetry=telemetry))

    builder.save(ROOT_DIR / SUBMISSION_FILENAME)
    print(f"\nSaved {SUBMISSION_FILENAME}")

if __name__ == "__main__":
    main()
