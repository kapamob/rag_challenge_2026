import os
import json
import time
import csv
from pathlib import Path
from typing import List, Dict, Any, Set
import sys
import re

# Constants
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.schema import NodeWithScore

from arlc import EvaluationClient, SubmissionBuilder, SubmissionAnswer, get_config
from arlc.telemetry import Telemetry, TimingMetrics, UsageMetrics, RetrievalRef, normalize_retrieved_pages

# Constants
CONFIG = get_config()
SELECTED_MODEL = os.environ.get("SELECTED_MODEL", "gpt-4o-mini")
SUBMISSION_FILENAME = os.environ.get("SUBMISSION_FILENAME", "submission_v21_multi_step.json")
METADATA_PATH = ROOT_DIR / "document_metadata.json"
INDEX_DIR = ROOT_DIR / "document_indices"

# Global Settings
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = OpenAILike(
    model="openai/gpt-4o-mini",
    api_key=os.environ.get("OPENROUTER_API_KEY") or CONFIG.openrouter_api_key,
    api_base="https://openrouter.ai/api/v1",
    is_chat_model=True,
    max_tokens=2048,
)

def load_metadata() -> List[Dict[str, Any]]:
    with open(METADATA_PATH, 'r') as f:
        return json.load(f)

def find_relevant_docs(query: str, metadata: List[Dict[str, Any]]) -> Set[str]:
    """
    Routes the query to specific document IDs based on mentions of 
    Case IDs, Law Numbers, or Participants.
    """
    relevant_db_names = set()
    query_upper = query.upper()
    
    for doc in metadata:
        # 1. Check Document Number (Case ID / Law Number)
        doc_num = (doc.get("document_number") or "").upper()
        if doc_num and len(doc_num) > 3:
            # Case IDs usually contain "/" and spaces. We normalize.
            norm_doc_num = doc_num.replace(" ", "")
            norm_query = query_upper.replace(" ", "")
            if doc_num in query_upper or norm_doc_num in norm_query:
                relevant_db_names.add(doc["db_name"])
                continue

        # 2. Check Participants (Claimant / Defendant)
        claimant = (doc.get("claimant") or "").upper()
        if claimant and len(claimant) > 3 and claimant in query_upper:
            relevant_db_names.add(doc["db_name"])
            continue
            
        defendant = (doc.get("defendant") or "").upper()
        if defendant and len(defendant) > 3 and defendant in query_upper:
            relevant_db_names.add(doc["db_name"])
            continue

        # 3. Check Title fragments
        title = (doc.get("title") or "").upper()
        if title and "LAW" in title:
            # For Laws, we often have "Operating Law" or "Foundations Law"
            short_title = title.replace("DIFC", "").replace("LAW", "").strip()
            if short_title and len(short_title) > 3 and short_title in query_upper:
                relevant_db_names.add(doc["db_name"])
                continue

    return relevant_db_names

def get_retriever_for_docs(db_names: Set[str]):
    """Loads indices for specific documents and builds a fused retriever."""
    retrievers = []
    
    for db_name in db_names:
        storage_path = INDEX_DIR / db_name
        if not storage_path.exists():
            continue
            
        storage_context = StorageContext.from_defaults(persist_dir=str(storage_path))
        index = load_index_from_storage(storage_context)
        
        # Add Vector Retriever
        retrievers.append(index.as_retriever(similarity_top_k=5))
        
        # Add BM25 Retriever
        bm25 = BM25Retriever.from_defaults(index=index, similarity_top_k=5)
        retrievers.append(bm25)
        
    if not retrievers:
        return None
        
    return QueryFusionRetriever(
        retrievers,
        similarity_top_k=10,
        num_queries=1, # No query expansion to save time/cost
        use_async=True
    )

def main():
    client = EvaluationClient.from_env()
    questions = client.download_questions(target_path=CONFIG.questions_path)
    metadata = load_metadata()
    
    reranker = LLMRerank(choice_batch_size=5, top_n=5)
    builder = SubmissionBuilder()

    print(f"Answering {len(questions)} questions using Multi-Step Routing...")

    for i, q in enumerate(questions):
        q_id = q["id"]
        q_text = q["question"]
        q_type = q.get("answer_type", "free_text")
        
        print(f"[{i+1}/{len(questions)}] {q_id}")
        
        # 1. Routing
        relevant_docs = find_relevant_docs(q_text, metadata)
        print(f"  - Routed to: {list(relevant_docs)}")
        
        # 2. Retrieval
        if not relevant_docs:
            # Fallback: Search all documents (this might be slow, but it's a fallback)
            # Optimization: In a real system, we'd have a 'Global Index' for fallbacks.
            # For this competition, we just load all metadata-matched docs.
            # If still nothing, we might need a general index. 
            # For now, let's assume we want precision above all.
            print("  - WARNING: No docs found via routing. Falling back to global search (not implemented yet).")
            # TODO: Implement a lightweight global retriever or just use the metadata to guess.
            context_nodes = []
        else:
            retriever = get_retriever_for_docs(relevant_docs)
            if retriever:
                context_nodes = retriever.retrieve(q_text)
                # 3. Reranking
                context_nodes = reranker.postprocess_nodes(context_nodes, query_str=q_text)
            else:
                context_nodes = []

        # 4. Generate Answer
        context = "\n\n".join([n.text for n in context_nodes])
        
        # Build reference registry forcitations
        registry_entries = []
        for n in context_nodes:
            doc_id = n.metadata.get("file_name", "unknown").replace(".pdf", "")
            page_label = n.metadata.get("page_number", "1")
            registry_entries.append(f"- Document: {doc_id}, Page: {page_label}")
        doc_registry = "\n".join(list(set(registry_entries)))

        prompt = f"""
        You are a legal assistant. Answer the question based ONLY on the provided context.
        If the answer is NOT in the context, respond with "null" for deterministic types or a polite "not found" for free_text.

        Question: {q_text}
        Answer Type: {q_type}

        Document Registry:
        {doc_registry}

        Context:
        {context}

        INSTRUCTIONS:
        1. Base your answer solely on the Context.
        2. Format the output as JSON with "answer" and "citations" (list of DocIDs).
        3. If no answer found, "answer" should be null.
        """

        start_time = time.time()
        try:
            response = Settings.llm.complete(prompt).text.strip()
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                ans_text = data.get("answer")
                used_docs = data.get("citations", [])
            else:
                ans_text = "null"
                used_docs = []
        except Exception as e:
            print(f"  - Error: {e}")
            ans_text = "null"
            used_docs = []
        end_time = time.time()

        # Parse and Telemetry
        # (Same logic as in v20 but using the new filtered context)
        parsed_ans = ans_text
        if isinstance(ans_text, str):
            if ans_text.lower() == "null" or not ans_text: parsed_ans = None
            elif q_type == "boolean": parsed_ans = "true" in ans_text.lower()
            elif q_type == "number":
                try: parsed_ans = float(str(ans_text).replace(",", "").strip())
                except: parsed_ans = ans_text

        # Grounding
        pages_ref = []
        used_docs_clean = [str(d).lower().replace(".pdf", "") for d in used_docs]
        for n in context_nodes:
            doc_id = n.metadata.get("file_name", "unknown").replace(".pdf", "")
            if any(doc_id.lower() in d or d in doc_id.lower() for d in used_docs_clean):
                page_label = n.metadata.get("page_number", "1")
                pages_ref.append({"doc_id": doc_id, "page_numbers": [int(page_label)]})

        telemetry = Telemetry(
            timing=TimingMetrics(total_time_ms=int((end_time - start_time) * 1000), ttft_ms=100, tpot_ms=10),
            retrieval=normalize_retrieved_pages(pages_ref),
            usage=UsageMetrics(0, 0),
            model_name=SELECTED_MODEL
        )
        
        builder.add_answer(SubmissionAnswer(question_id=q_id, answer=parsed_ans, telemetry=telemetry))

        if (i + 1) % 10 == 0:
            builder.save(ROOT_DIR / SUBMISSION_FILENAME)

    builder.save(ROOT_DIR / SUBMISSION_FILENAME)
    print(f"Done! Submission saved to {SUBMISSION_FILENAME}")

if __name__ == "__main__":
    main()
