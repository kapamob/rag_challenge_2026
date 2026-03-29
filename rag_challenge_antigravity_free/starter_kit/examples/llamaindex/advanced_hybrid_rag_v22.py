import json
import os
import sys
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from datetime import datetime

import tiktoken
from llama_index.core import Settings, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import LLMRerank
from llama_index.readers.docling import DoclingReader

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

CONFIG = get_config()
TOKENIZER = tiktoken.get_encoding("cl100k_base")

# --- OpenRouter credentials (v22 needs OpenRouter, not DeepSeek) ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

from llama_index.core.embeddings import BaseEmbedding
from openai import OpenAI
from pydantic import PrivateAttr

class OpenRouterEmbedding(BaseEmbedding):
    """Custom embedding class for OpenRouter to support model prefixes like 'openai/'."""
    model_name: str
    _client: OpenAI = PrivateAttr()
    
    def __init__(self, model_name: str, api_key: str, api_base: str, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self._client = OpenAI(api_key=api_key, base_url=api_base)

    def _get_query_embedding(self, query: str):
        return self._client.embeddings.create(input=[query], model=self.model_name).data[0].embedding

    def _get_text_embedding(self, text: str):
        return self._client.embeddings.create(input=[text], model=self.model_name).data[0].embedding

    def _get_text_embeddings(self, texts: list[str]):
        # OpenRouter supports batching
        res = self._client.embeddings.create(input=texts, model=self.model_name).data
        return [item.embedding for item in res]

    async def _aget_query_embedding(self, query: str):
        return self._get_query_embedding(query)

# --- v22 SETTINGS ---
# Use text-embedding-3-large for better semantic resolution via OpenRouter
Settings.embed_model = OpenRouterEmbedding(
    model_name="openai/text-embedding-3-large",
    api_key=OPENROUTER_API_KEY,
    api_base=OPENROUTER_API_BASE
)

# Use gpt-4o-mini for efficient reasoning via OpenRouter
Settings.llm = OpenAILike(
    model="openai/gpt-4o-mini",
    api_key=OPENROUTER_API_KEY,
    api_base=OPENROUTER_API_BASE,
    is_chat_model=True,
    max_tokens=2048,
    temperature=0.0,
)

Settings.chunk_size = 1024
Settings.chunk_overlap = 128
Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=128)

def download_resources(client: EvaluationClient) -> list[dict]:
    """Download questions and documents via the API"""
    print("Downloading questions...")
    questions = client.download_questions(target_path=CONFIG.questions_path)
    print("Downloading documents...")
    client.download_documents(CONFIG.docs_dir)
    print("Documents extracted")
    return questions

def _parse_answer_by_type(raw: str, answer_type: str):
    """Parse string response according to answer_type. ISO 8601 for dates."""
    text = (raw or "").strip()
    at = str(answer_type or "free_text").lower()

    if text.lower() in ["null", "none", "not found", "unanswerable", ""]:
        return None

    if at == "number":
        try:
            # Clean non-numeric except . and ,
            cleaned = re.sub(r"[^\d.,-]", "", text)
            return float(cleaned.replace(",", "."))
        except (TypeError, ValueError):
            return text

    if at == "boolean":
        lower = text.lower()
        if any(x in lower for x in ("true", "yes", "1")): return True
        if any(x in lower for x in ("false", "no", "0")): return False
        return text

    if at == "date":
        # Strategy: strict conversion to YYYY-MM-DD
        formats = [
            "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y.%m.%d",
            "%d %B %Y", "%B %d, %Y", "%d %b %Y", "%b %d, %Y"
        ]
        # Clean text from potential garbage
        clean_date = re.sub(r"(st|nd|rd|th),?", "", text) # Remove ordinal suffixes
        for fmt in formats:
            try:
                dt = datetime.strptime(clean_date, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        # Fallback to whatever LLM gave if no format matched but maybe it's already OK
        match = re.search(r"(\d{4}-\d{2}-\d{2})", text)
        if match:
            return match.group(1)
        return text

    if at == "names":
        parts = [p for p in (x.strip() for x in text.replace(",", ";").split(";")) if p]
        return parts if parts else [text]

    if at == "null":
        return None

    return text

_TYPE_INSTRUCTIONS: dict[str, str] = {
    "number":    "Return only the numeric value. No units. No explanation.",
    "boolean":   "Return only 'true' or 'false'.",
    "name":      "Return the exact name from documents. No aliases.",
    "names":     "Return a semicolon-separated list of exact names.",
    "date":      "Return the date in ISO 8601 format (YYYY-MM-DD). Strictly.",
    "free_text": "Answer grounded in context. Max 280 chars.",
}

def build_context_with_metadata(nodes: list) -> str:
    context_parts = []
    for i, node in enumerate(nodes):
        metadata = node.metadata or {}
        # Docling usually provides dl_meta or standard metadata
        doc_id = metadata.get("file_name", metadata.get("doc_id", "unknown"))
        doc_id = Path(doc_id).stem if "." in doc_id else doc_id
        page = metadata.get("page_number", metadata.get("dl_meta", {}).get("page_number", "unknown"))
        
        context_parts.append(f"--- Source {i+1} ---\n[doc_id: {doc_id}, page_numbers: [{page}]]\n{node.text}")
    return "\n\n".join(context_parts)

def build_prompt(context: str, question_text: str, answer_type: str = "free_text") -> str:
    instruction = _TYPE_INSTRUCTIONS.get(answer_type, _TYPE_INSTRUCTIONS["free_text"])
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question_text}\n\n"
        f"You are a legal assistant. Answer based ONLY on the context.\n"
        f"Format: VALID JSON with keys 'answer' and 'sources'.\n"
        f"- 'answer': {instruction} (If not found, set to null)\n"
        f"- 'sources': List of {{'doc_id': str, 'page_numbers': [int]}}.\n"
        "Return ONLY the raw JSON string."
    )

def main():
    client = EvaluationClient.from_env()
    questions = download_resources(client)
    docs_dir = Path(CONFIG.docs_dir)
    storage_dir = Path("./storage_v22")

    if not storage_dir.exists():
        print("Using DoclingReader for advanced PDF parsing...")
        reader = DoclingReader()
        pdf_files = sorted(list(docs_dir.glob("*.pdf")))
        
        all_docs = []
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"[{i}/{len(pdf_files)}] Parsing {pdf_path.name}...")
            try:
                # Process one by one to avoid segfault/memory issues
                docs = reader.load_data(file_path=str(pdf_path))
                all_docs.extend(docs)
            except Exception as e:
                print(f"FAILED to parse {pdf_path.name}: {e}")

        print(f"Loaded {len(all_docs)} document sections")
        index = VectorStoreIndex.from_documents(all_docs, show_progress=True)
        storage_dir.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(storage_dir))
    else:
        print("Loading existing v22 index...")
        storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir))
        index = load_index_from_storage(storage_context)

    # Hybrid Search Configuration
    vector_retriever = index.as_retriever(similarity_top_k=10)
    bm25_retriever = BM25Retriever.from_defaults(index=index, similarity_top_k=5)
    
    retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=7,
        num_queries=1, # Single query for latency
        use_async=True,
    )
    
    reranker = LLMRerank(choice_batch_size=5, top_n=5)

    print("\nAnswering questions (v22: Docling + OpenAI 3-Large)...")
    builder = SubmissionBuilder(
        architecture_summary="v22: Docling Parsing, OpenAI text-embedding-3-large, Hybrid (Vector+BM25), ISO Date Fix.",
    )

    for i, question_item in enumerate(questions, 1):
        q_text = question_item["question"]
        q_id = question_item["id"]
        q_type = question_item.get("answer_type", "free_text")
        print(f"[{i}/{len(questions)}] {q_id}")

        timer = TelemetryTimer()
        nodes = retriever.retrieve(q_text)
        nodes = reranker.postprocess_nodes(nodes, query_str=q_text)
        
        context = build_context_with_metadata(nodes)
        prompt = build_prompt(context, q_text, q_type)

        response_text = ""
        try:
            for chunk in Settings.llm.stream_complete(prompt):
                timer.mark_token()
                response_text += chunk.delta
        except Exception as e:
            print(f"LLM Error: {e}")
            timer.mark_token()

        timing = timer.finish()
        
        answer = None
        retrieval_refs = []
        try:
            clean_json = response_text.strip()
            if "```" in clean_json:
                clean_json = re.sub(r"```json|```", "", clean_json).strip()
            
            data = json.loads(clean_json)
            raw_ans = data.get("answer")
            answer = _parse_answer_by_type(raw_ans, q_type)
            
            sources = data.get("sources", [])
            for s in sources:
                doc = s.get("doc_id")
                pgs = s.get("page_numbers", [])
                if doc and pgs:
                    retrieval_refs.append(RetrievalRef(doc_id=str(doc), page_numbers=pgs))
        except:
            answer = _parse_answer_by_type(response_text, q_type)
            for n in nodes:
                meta = n.metadata or {}
                doc = meta.get("file_name", meta.get("doc_id", "unknown"))
                doc = Path(doc).stem if "." in doc else doc
                pg = meta.get("page_number", meta.get("dl_meta", {}).get("page_number", 0))
                retrieval_refs.append(RetrievalRef(doc_id=str(doc), page_numbers=[int(pg) if str(pg).isdigit() else 1]))

        retrieval_refs = normalize_retrieved_pages(retrieval_refs)
        if answer is None: retrieval_refs = []

        usage = UsageMetrics(
            input_tokens=len(TOKENIZER.encode(prompt)),
            output_tokens=len(TOKENIZER.encode(response_text)),
        )
        telemetry = Telemetry(
            timing=TimingMetrics(ttft_ms=timing.ttft_ms, tpot_ms=timing.tpot_ms, total_time_ms=timing.total_time_ms),
            retrieval=retrieval_refs,
            usage=usage,
            model_name="openai/gpt-4o-mini",
        )
        builder.add_answer(SubmissionAnswer(question_id=q_id, answer=answer, telemetry=telemetry))

    final_sub_path = ROOT_DIR / f"submission_v22_docling.json"
    builder.save(final_sub_path)
    print(f"\nSaved v22 results to {final_sub_path}")

if __name__ == "__main__":
    main()
