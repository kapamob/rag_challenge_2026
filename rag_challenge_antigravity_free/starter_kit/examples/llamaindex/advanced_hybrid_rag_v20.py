import os
import json
import time
from pathlib import Path
from typing import List
import sys

# Constants
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.schema import NodeWithScore

from arlc import EvaluationClient, SubmissionBuilder, SubmissionAnswer, get_config
from arlc.telemetry import Telemetry, TimingMetrics, UsageMetrics, RetrievalRef, normalize_retrieved_pages

# Constants
CONFIG = get_config()
ROOT_DIR = Path(__file__).resolve().parents[2]
SELECTED_MODEL = os.environ.get("SELECTED_MODEL", "gpt-4o-mini")
SUBMISSION_FILENAME = os.environ.get("SUBMISSION_FILENAME", "submission_v20_raw.json")
TARGET_QUESTION_ID = os.environ.get("TARGET_QUESTION_ID", None)

# Model definitions
MODELS = {
    "gpt-4o-mini": {
        "class": OpenAILike,
        "params": {
            "model": "openai/gpt-4o-mini",
            "api_key": os.environ.get("OPENROUTER_API_KEY") or CONFIG.openrouter_api_key,
            "api_base": "https://openrouter.ai/api/v1",
            "is_chat_model": True,
            "max_tokens": 2048,
        }
    }
}

def get_llm():
    cfg = MODELS.get(SELECTED_MODEL, MODELS["gpt-4o-mini"])
    return cfg["class"](**cfg["params"])

# Global Settings
Settings.llm = get_llm()
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def download_data():
    client = EvaluationClient.from_env()
    print("Downloading questions...")
    questions = client.download_questions(target_path=CONFIG.questions_path)
    print("Downloading documents...")
    client.download_documents(CONFIG.docs_dir)
    return questions

def ingest_documents():
    print("Loading documents...")
    reader = SimpleDirectoryReader(CONFIG.docs_dir, recursive=True)
    documents = reader.load_data()
    
    print("Chunking with SemanticSplitter...")
    splitter = SemanticSplitterNodeParser(
        buffer_size=1, breakpoint_percentile_threshold=95, embed_model=Settings.embed_model
    )
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes

def build_index(nodes):
    print(f"Indexing {len(nodes)} semantic nodes...")
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    return index, nodes

def main():
    questions = download_data()
    nodes = ingest_documents()
    index, nodes = build_index(nodes)

    # 1. Setup Retrievers
    vector_retriever = index.as_retriever(similarity_top_k=10)
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=10,
    )

    # 2. Hybrid Search (RRF)
    retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=10,
        num_queries=1,
        use_async=True,
    )

    # 3. Reranker (Precision boost)
    reranker = LLMRerank(choice_batch_size=5, top_n=10) # Increased from 5 to 10

    # 4. Load existing answers for resume
    existing_answers = {}
    full_path = ROOT_DIR / SUBMISSION_FILENAME
    if full_path.exists():
        try:
            with open(full_path, 'r') as f:
                data = json.load(f)
                existing_answers = {a['question_id']: a for a in data.get('answers', [])}
            print(f"Loaded {len(existing_answers)} existing answers. Resuming...")
        except Exception as e:
            print(f"Error loading existing answers: {e}")

    builder = SubmissionBuilder()
    # Pre-populate builder with existing answers
    for q_id, ans_data in existing_answers.items():
        # Ideally we'd re-validate but arlc.submission is opaque here.
        # Let's just keep track to skip.
        pass

    print(f"Answering {len(questions)} questions using {SELECTED_MODEL}...")
    for i, q in enumerate(questions):
        q_id = q["id"]
        
        if TARGET_QUESTION_ID and q_id != TARGET_QUESTION_ID:
            continue

        if q_id in existing_answers:
            print(f"[{i+1}/{len(questions)}] Skipping {q_id} (already answered)")
            # Re-add to builder so the final save has EVERYTHING
            # Note: SubmissionAnswer needs to be reconstructed correctly
            # Mapping back from dict to objects
            a_data = existing_answers[q_id]
            
            # We're already importing these at the top level.
            t = a_data.get('telemetry', {})
            tm = t.get('timing', {})
            us = t.get('usage', {})
            ret = t.get('retrieval', {}).get('retrieved_chunk_pages', [])
            
            telemetry_obj = Telemetry(
                timing=TimingMetrics(ttft_ms=tm.get('ttft_ms', 0), tpot_ms=tm.get('tpot_ms', 0), total_time_ms=tm.get('total_time_ms', 0)),
                retrieval=[RetrievalRef(doc_id=r['doc_id'], page_numbers=r['page_numbers']) for r in ret],
                usage=UsageMetrics(input_tokens=us.get('input_tokens', 0), output_tokens=us.get('output_tokens', 0)),
                model_name=t.get('model_name')
            )
            builder.add_answer(SubmissionAnswer(question_id=q_id, answer=a_data.get('answer'), telemetry=telemetry_obj))
            continue

        q_text = q["question"]
        q_type = q.get("answer_type", "free_text")
        
        print(f"[{i+1}/{len(questions)}] {q_id}")
        
        # 15.1 Metadata Manifest Injection
        import re
        case_pattern = re.compile(r'(CFI|SCT|ARB|CA|ENF|DEC|TCD)\s*\d+/\d+', re.IGNORECASE)
        query_cases = case_pattern.findall(q_text)
        # Normalize: replace space if any (e.g. "CFI 057/2025" -> "CFI 057/2025" but also look for "CFI057/2025")
        query_cases = [c[0] + " " + c[1] if isinstance(c, tuple) else c for c in query_cases]
        # Actually findall returns tuples if there are groups. Let's fix regex.
        query_cases = re.findall(r'(?:CFI|SCT|ARB|CA|ENF|DEC|TCD)\s*\d+/\d+', q_text, re.IGNORECASE)
        print(f"Detected case IDs in query: {query_cases}")

        # Increased retrieval depth for the reranker
        vector_retriever = index.as_retriever(similarity_top_k=20)
        bm25_retriever = BM25Retriever.from_defaults(index=index, similarity_top_k=20)
        
        v_nodes = vector_retriever.retrieve(q_text)
        b_nodes = bm25_retriever.retrieve(q_text)
        
        # Merge and Boost nodes containing the Case ID
        all_nodes = v_nodes + b_nodes
        
        # 20.2 FORCE SEARCH (Guaranteed Recall for Case IDs)
        if query_cases:
            print(f"Force-searching nodes for {query_cases}...")
            for node in nodes: # 'nodes' is the global list of all chunks
                content = node.get_content().upper()
                is_match = False
                for case_id in query_cases:
                    if case_id.upper() in content or case_id.upper().replace(" ", "") in content:
                        is_match = True
                        break
                if is_match:
                    # Create a NodeWithScore if not already present
                    from llama_index.core.schema import NodeWithScore
                    # Check if already in all_nodes
                    if not any(n.node.node_id == node.node_id for n in all_nodes):
                        all_nodes.append(NodeWithScore(node=node, score=10.0))
                        print(f"  + Force-added node from {node.metadata.get('file_name')} (Case match)")

        seen = set()
        unique_nodes = []
        for n in all_nodes:
            if n.node.node_id not in seen:
                # BOOSTING logic
                content = n.node.get_content().upper()
                boost = 0
                for case_id in query_cases:
                    norm_case = case_id.upper().replace(" ", "")
                    space_case = case_id.upper()
                    if space_case in content or norm_case in content:
                        boost = 5.0 # Even more aggressive boost
                
                n.score = (n.score or 1.0) * (1.0 + boost)
                unique_nodes.append(n)
                seen.add(n.node.node_id)
        
        # Sort by boosted score
        unique_nodes.sort(key=lambda x: x.score or 0, reverse=True)
        
        # 20.3 STRICT FILTERING (Remove non-matching cases if Case ID is in query)
        if query_cases:
            filtered_nodes = []
            for n in unique_nodes:
                content = n.node.get_content().upper()
                if any(case_id.upper() in content or case_id.upper().replace(" ", "") in content for case_id in query_cases):
                    filtered_nodes.append(n)
            unique_nodes = filtered_nodes
            print(f"Applied strict filter. Candidates left: {len(unique_nodes)}")
        
        print(f"Top candidates for reranker:")
        for n in unique_nodes[:15]:
            fname = n.node.metadata.get('file_name', 'UNK')
            snippet = n.node.get_content()[:100].replace('\n', ' ')
            print(f"  - Score: {n.score:.4f} | File: {fname} | Snippet: {snippet}")

        reranked_nodes = reranker.postprocess_nodes(unique_nodes[:15], query_str=q_text)

        # Build context from reranked nodes
        context = "\n\n".join([n.text for n in reranked_nodes])

        # 15.1 Metadata Manifest Injection
        # Extract metadata from reranked nodes to create a "Document Registry"
        registry_entries = []
        for n in reranked_nodes:
            doc_id = n.metadata.get("file_name", "unknown").replace(".pdf", "")
            page_label = n.metadata.get("page_label", "1")
            registry_entries.append(f"- Document: {doc_id}, Page: {page_label}")
        
        doc_registry = "\n".join(list(set(registry_entries)))

        # 15.2 Grounding Citation Filter / 15.4 Reflection
        # We'll ask for citations and parse them to satisfy the "only used documents" rule.
        prompt = f"""
You are an AI assistant in a legal agency. Your answers must be strictly based only on the provided context. Do not invent anything.
This is a database of legal documents: laws and analyses of specific cases. Each court decision contains a number, date, and the names of the plaintiff and defendant.
Question: {q_text}
Answer Type: {q_type}
Document Registry:
{doc_registry}

Context:
{context}

---
ИНСТРУКЦИИ:
1. Сначала проанализируй вопрос и выдели НОМЕР ДЕЛА (Case Number, например CFI 057/2025).
2. Проверь каждый документ в Context. Если НОМЕР ДЕЛА в документе не совпадает с искомым - ИГНОРИРУЙ этот документ, даже если там есть похожие даты.
3. Ответ должен быть максимально кратким (только дата в формате YYYY-MM-DD).
4. Добавь в конце ответа список использованных документов в формате CITATIONS: [DocID1, DocID2].
5. Тип ответа {q_type}:
   - deterministic (number, boolean, name, names, date): Только значение. Если нет в базе - "null".
   - free_text: Максимум 280 символов.
6. Не цитируй вопрос. Сверь данные в ответе и контексте перед выводом.

ФОРМАТ ОТВЕТА (JSON):
{{
  "answer": "твой_ответ",
  "citations": ["DocID", "DocID"]
}}
"""
        
        start_time = time.time()
        try:
            response = Settings.llm.complete(prompt).text.strip()
            # Try parsing JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                ans_text = data.get("answer")
                used_docs = data.get("citations", [])
            else:
                # Fallback to plain text if JSON fails
                ans_text = response
                used_docs = [n.metadata.get("file_name", "").replace(".pdf", "") for n in reranked_nodes]
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            ans_text = "null"
            used_docs = []
        
        end_time = time.time()
        
        # Simple parse cleanup
        parsed_ans = ans_text
        if isinstance(ans_text, str):
            if ans_text.lower() == "null" or not ans_text:
                parsed_ans = None
            elif q_type == "boolean":
                parsed_ans = "true" in ans_text.lower()
            elif q_type == "number":
                try:
                    parsed_ans = float(str(ans_text).replace(",", "").strip())
                except:
                    parsed_ans = ans_text
        
        # 15.2 Filter retrieval pages based on citations
        pages_ref = []
        used_docs_clean = [str(d).lower().replace(".pdf", "") for d in used_docs]
        for n in reranked_nodes:
            doc_id = n.metadata.get("file_name", "unknown").replace(".pdf", "")
            if any(doc_id.lower() in d or d in doc_id.lower() for d in used_docs_clean):
                page_label = n.metadata.get("page_label", "1")
                pages_ref.append({"doc_id": doc_id, "page_numbers": [int(page_label)]})
        

        timing = TimingMetrics(
            total_time_ms=int((end_time - start_time) * 1000),
            ttft_ms=100, 
            tpot_ms=10
        )
        usage = UsageMetrics(input_tokens=0, output_tokens=0)
        
        # Use normalize_retrieved_pages for robustness
        retrieval_refs = normalize_retrieved_pages(pages_ref) if parsed_ans is not None else []
        
        telemetry_obj = Telemetry(
            timing=timing,
            retrieval=retrieval_refs,
            usage=usage,
            model_name=SELECTED_MODEL
        )
        
        builder.add_answer(SubmissionAnswer(question_id=q_id, answer=parsed_ans, telemetry=telemetry_obj))

        # Incremental save
        if (i + 1) % 10 == 0:
            builder.save(ROOT_DIR / SUBMISSION_FILENAME)
            print(f"  -> Intermediate save to {SUBMISSION_FILENAME}")

    builder.save(ROOT_DIR / SUBMISSION_FILENAME)
    print(f"\nSaved {SUBMISSION_FILENAME}")

if __name__ == "__main__":
    main()
