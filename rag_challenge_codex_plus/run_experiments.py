from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz  # pymupdf
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parent
STARTER_DIR = ROOT_DIR / "starter_kit"
sys.path.append(str(STARTER_DIR))

from arlc import (  # noqa: E402
    EvaluationClient,
    RetrievalRef,
    SubmissionAnswer,
    SubmissionBuilder,
    Telemetry,
    TelemetryTimer,
    TimingMetrics,
    UsageMetrics,
    normalize_retrieved_pages,
)


load_dotenv(ROOT_DIR / ".env")

def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / 4))


@dataclass(frozen=True)
class ExperimentConfig:
    run_id: str
    retrieval_mode: str  # "hybrid" or "vector_only"
    question_limit: int
    vector_top_k: int
    lexical_top_k: int
    fused_top_k: int
    answer_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int


def _get_env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _normalize_model_for_llamaindex(model_name: str) -> str:
    # LlamaIndex OpenAI wrappers validate known OpenAI model names and reject provider prefixes.
    # OpenRouter-style ids like "openai/gpt-4o-mini" are converted to "gpt-4o-mini".
    if "/" in model_name:
        return model_name.split("/", 1)[1]
    return model_name


def _configure_models(config: ExperimentConfig) -> None:
    # User rule:
    # - OPENAI_API_* only for DeepSeek
    # - OPENROUTER_* for all other models and embeddings
    is_deepseek = "deepseek" in config.answer_model.lower()
    llm_model = _normalize_model_for_llamaindex(config.answer_model)
    embedding_model = _normalize_model_for_llamaindex(config.embedding_model)
    if is_deepseek:
        llm_key = _get_env("OPENAI_API_KEY")
        llm_base = _get_env("OPENAI_API_BASE")
        if not llm_key or not llm_base:
            raise RuntimeError("DeepSeek model selected, but OPENAI_API_KEY/OPENAI_API_BASE are not set.")
    else:
        llm_key = _get_env("OPENROUTER_API_KEY")
        llm_base = _get_env("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
        if not llm_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for non-DeepSeek LLM experiments.")

    embed_key = _get_env("OPENROUTER_API_KEY")
    embed_base = _get_env("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    if not embed_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for embedding experiments.")

    Settings.llm = OpenAI(
        model=llm_model,
        api_key=llm_key,
        api_base=llm_base,
        temperature=0.0,
    )
    Settings.embed_model = OpenAIEmbedding(
        model=embedding_model,
        api_key=embed_key,
        api_base=embed_base,
    )
    Settings.chunk_size = config.chunk_size
    Settings.chunk_overlap = config.chunk_overlap


def _download_once(cache_dir: Path) -> tuple[list[dict[str, Any]], Path]:
    questions_path = cache_dir / "questions.json"
    docs_dir = cache_dir / "docs_corpus"
    docs_zip = docs_dir / "documents.zip"

    eval_api_key = _get_env("EVAL_API_KEY")
    if not eval_api_key:
        raise RuntimeError("EVAL_API_KEY is missing in environment.")
    eval_base_url = _get_env("EVAL_BASE_URL", "https://platform.agentic-challenge.ai/api/v1")
    client = EvaluationClient(api_key=eval_api_key, base_url=eval_base_url)

    if questions_path.exists():
        questions = json.loads(questions_path.read_text(encoding="utf-8"))
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        questions = client.download_questions(target_path=questions_path)

    if not docs_zip.exists():
        docs_dir.mkdir(parents=True, exist_ok=True)
        client.download_documents(docs_dir)

    return questions, docs_dir


def _extract_participants(sample_text: str) -> tuple[str, str]:
    claimant = ""
    defendant = ""
    patterns = [
        (r"claimant[:\s]+([^\n;]+)", "claimant"),
        (r"defendant[:\s]+([^\n;]+)", "defendant"),
        (r"plaintiff[:\s]+([^\n;]+)", "claimant"),
    ]
    text = sample_text.lower()
    for pattern, kind in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        value = match.group(1).strip()
        if kind == "claimant" and not claimant:
            claimant = value
        if kind == "defendant" and not defendant:
            defendant = value
    return claimant, defendant


def _safe_pdf_date(raw_value: Any) -> str:
    if not raw_value:
        return ""
    value = str(raw_value).strip()
    value = value.replace("D:", "")
    for fmt in ("%Y%m%d%H%M%S", "%Y%m%d"):
        try:
            return datetime.strptime(value[: len(fmt.replace("%", ""))], fmt).date().isoformat()
        except ValueError:
            continue
    return ""


def _build_docs_and_pages(docs_dir: Path, output_dir: Path) -> tuple[list[Document], list[dict[str, Any]], list[dict[str, Any]]]:
    pdf_files = sorted(docs_dir.rglob("*.pdf"))
    all_documents: list[Document] = []
    page_records: list[dict[str, Any]] = []
    docs_list_rows: list[dict[str, Any]] = []

    samples_dir = output_dir / "parsed_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    for index, pdf_path in enumerate(pdf_files, 1):
        doc_id = pdf_path.stem
        pdf = fitz.open(pdf_path)
        meta = pdf.metadata or {}
        title = (meta.get("title") or "").strip()
        doc_date = _safe_pdf_date(meta.get("creationDate") or meta.get("modDate"))
        text_pages: list[str] = []
        non_empty = 0
        sample_texts: list[str] = []

        for page_idx in range(pdf.page_count):
            page = pdf[page_idx]
            text = (page.get_text("text") or "").strip()
            if text:
                non_empty += 1
            text_pages.append(text)
            if page_idx < 2 and text:
                sample_texts.append(text[:2000])
            all_documents.append(
                Document(
                    text=text,
                    metadata={
                        "doc_id": doc_id,
                        "file_name": pdf_path.name,
                        "title": title,
                        "page_number": page_idx + 1,
                    },
                )
            )
            page_records.append(
                {
                    "doc_id": doc_id,
                    "page_number": page_idx + 1,
                    "text": text,
                }
            )

        sample_text = "\n".join(sample_texts)
        claimant, defendant = _extract_participants(sample_text)
        docs_list_rows.append(
            {
                "doc_id": doc_id,
                "file_name": pdf_path.name,
                "title": title,
                "doc_number": "",
                "doc_date": doc_date,
                "doc_type": "",
                "claimant": claimant,
                "defendant": defendant,
                "total_pages": pdf.page_count,
                "non_empty_text_pages": non_empty,
                "text_coverage_ratio": round(non_empty / pdf.page_count, 4) if pdf.page_count else 0.0,
                "ocr_used": 0,
            }
        )
        pdf.close()

        if index <= 2:
            sample_path = samples_dir / f"{doc_id}_sample.md"
            sample_path.write_text("# Parsed sample\n\n" + sample_text[:8000], encoding="utf-8")

    with (output_dir / "docs_list.csv").open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(docs_list_rows[0].keys()) if docs_list_rows else [])
        if docs_list_rows:
            writer.writeheader()
            writer.writerows(docs_list_rows)
    return all_documents, page_records, docs_list_rows


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{2,}", (text or "").lower()))


def _lexical_retrieve(question: str, pages: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    q_tokens = _tokenize(question)
    scored: list[tuple[float, dict[str, Any]]] = []
    for page in pages:
        text = page["text"]
        if not text:
            continue
        p_tokens = _tokenize(text)
        overlap = len(q_tokens & p_tokens)
        if overlap <= 0:
            continue
        score = overlap / max(1, math.sqrt(len(p_tokens)))
        scored.append((score, page))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_k]]


def _rrf_fuse(vector_nodes: list[Any], lexical_pages: list[dict[str, Any]], fused_top_k: int) -> list[dict[str, Any]]:
    scores: dict[tuple[str, int], float] = {}
    payloads: dict[tuple[str, int], dict[str, Any]] = {}

    for rank, node in enumerate(vector_nodes, 1):
        md = node.metadata or {}
        key = (str(md.get("doc_id", "")), int(md.get("page_number", 0)))
        if not key[0] or key[1] <= 0:
            continue
        scores[key] = scores.get(key, 0.0) + (1.0 / (60 + rank))
        payloads[key] = {
            "doc_id": key[0],
            "page_number": key[1],
            "text": node.text,
            "source": "vector",
        }

    for rank, page in enumerate(lexical_pages, 1):
        key = (str(page.get("doc_id", "")), int(page.get("page_number", 0)))
        if not key[0] or key[1] <= 0:
            continue
        scores[key] = scores.get(key, 0.0) + (1.0 / (60 + rank))
        payloads.setdefault(
            key,
            {
                "doc_id": key[0],
                "page_number": key[1],
                "text": page.get("text", ""),
                "source": "lexical",
            },
        )

    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:fused_top_k]
    return [payloads[key] for key, _ in ordered]


_TYPE_INSTRUCTIONS = {
    "number": "Return only a JSON number. If absent, return null.",
    "boolean": "Return only true or false. If absent, return null.",
    "name": "Return only one exact name string. If absent, return null.",
    "names": "Return only a JSON array of name strings. If absent, return null.",
    "date": "Return only YYYY-MM-DD. If absent, return null.",
    "free_text": "Return concise factual answer from context only. Max 280 chars.",
}


def _build_prompt(question: str, answer_type: str, context_chunks: list[str]) -> str:
    instruction = _TYPE_INSTRUCTIONS.get(answer_type, _TYPE_INSTRUCTIONS["free_text"])
    context = "\n\n".join(context_chunks[:6]).strip()
    return (
        "You are a legal RAG assistant.\n"
        "Use only the provided context.\n"
        f"Instruction: {instruction}\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )


def _normalize_answer(raw: str, answer_type: str):
    text = (raw or "").strip()
    if not text:
        return None if answer_type != "free_text" else "There is no information on this question in the provided documents."

    if text.lower() == "null":
        return None
    if answer_type == "number":
        try:
            return float(text.replace(",", "."))
        except ValueError:
            return None
    if answer_type == "boolean":
        low = text.lower()
        if low in {"true", "yes", "1"}:
            return True
        if low in {"false", "no", "0"}:
            return False
        return None
    if answer_type == "names":
        if text.startswith("["):
            try:
                value = json.loads(text)
                if isinstance(value, list):
                    return [str(x).strip() for x in value if str(x).strip()]
            except json.JSONDecodeError:
                pass
        values = [part.strip() for part in re.split(r"[;,]", text) if part.strip()]
        return values or None
    if answer_type == "free_text":
        return text[:280]
    if answer_type == "date":
        match = re.search(r"\d{4}-\d{2}-\d{2}", text)
        return match.group(0) if match else None
    return text


def _build_retrieval_refs(evidence: list[dict[str, Any]]) -> list[RetrievalRef]:
    refs = [
        RetrievalRef(doc_id=item["doc_id"], page_numbers=[int(item["page_number"])])
        for item in evidence
        if item.get("doc_id") and item.get("page_number")
    ]
    return normalize_retrieved_pages(refs)


def _estimate_cost_usd(total_input_tokens: int, total_output_tokens: int) -> float:
    # Rough estimate for gpt-4o-mini-like pricing tier; keep conservative for budget tracking.
    input_per_m = 0.15
    output_per_m = 0.60
    return round((total_input_tokens / 1_000_000) * input_per_m + (total_output_tokens / 1_000_000) * output_per_m, 4)


def run_experiment(
    config: ExperimentConfig,
    questions: list[dict[str, Any]],
    results_root: Path,
    pages: list[dict[str, Any]],
    docs_rows: list[dict[str, Any]],
    index: VectorStoreIndex,
    include_docs_list_copy_from: Path | None = None,
) -> dict[str, Any]:
    run_dir = results_root / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    retriever = index.as_retriever(similarity_top_k=config.vector_top_k)
    if include_docs_list_copy_from and include_docs_list_copy_from.exists():
        (run_dir / "docs_list.csv").write_text(include_docs_list_copy_from.read_text(encoding="utf-8"), encoding="utf-8")

    sampled_questions = questions[: config.question_limit]
    builder = SubmissionBuilder(
        architecture_summary=f"{config.run_id}: PyMuPDF + {config.retrieval_mode} + {config.answer_model}",
    )

    submission_rows: list[dict[str, Any]] = []
    ttft_values: list[int] = []
    total_values: list[int] = []
    retrieval_pages_counts: list[int] = []
    null_count = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for q in sampled_questions:
        qid = q["id"]
        question = q["question"]
        answer_type = str(q.get("answer_type", "free_text")).lower()

        vector_nodes = retriever.retrieve(question)
        lexical_pages = _lexical_retrieve(question, pages, config.lexical_top_k) if config.retrieval_mode == "hybrid" else []
        evidence = _rrf_fuse(vector_nodes, lexical_pages, config.fused_top_k)

        if not evidence and answer_type != "free_text":
            answer = None
            response_text = "null"
            timing = TimingMetrics(ttft_ms=0, tpot_ms=0, total_time_ms=0)
            refs = []
            input_tokens = 0
            output_tokens = 0
        elif not evidence and answer_type == "free_text":
            answer = "There is no information on this question in the provided documents."
            response_text = answer
            timing = TimingMetrics(ttft_ms=0, tpot_ms=0, total_time_ms=0)
            refs = []
            input_tokens = 0
            output_tokens = _approx_tokens(response_text)
        else:
            context_chunks = [f"[doc:{e['doc_id']} page:{e['page_number']}]\n{e['text'][:2200]}" for e in evidence]
            prompt = _build_prompt(question, answer_type, context_chunks)
            telemetry_timer = TelemetryTimer()
            response_chunks: list[str] = []
            for chunk in Settings.llm.stream_complete(prompt):
                telemetry_timer.mark_token()
                response_chunks.append(chunk.delta)
            response_text = "".join(response_chunks).strip()
            answer = _normalize_answer(response_text, answer_type)
            timing = telemetry_timer.finish()
            refs = _build_retrieval_refs(evidence)
            input_tokens = _approx_tokens(prompt)
            output_tokens = _approx_tokens(response_text)

        if answer is None:
            null_count += 1

        telemetry = Telemetry(
            timing=TimingMetrics(ttft_ms=timing.ttft_ms, tpot_ms=timing.tpot_ms, total_time_ms=timing.total_time_ms),
            retrieval=refs,
            usage=UsageMetrics(input_tokens=input_tokens, output_tokens=output_tokens),
            model_name=config.answer_model,
        )
        builder.add_answer(SubmissionAnswer(question_id=qid, answer=answer, telemetry=telemetry))

        ttft_values.append(timing.ttft_ms)
        total_values.append(timing.total_time_ms)
        retrieval_pages_counts.append(sum(len(r.page_numbers) for r in refs))
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        submission_rows.append(
            {
                "question_id": qid,
                "answer_type": answer_type,
                "question": question,
                f"{config.run_id}.answer": json.dumps(answer, ensure_ascii=False),
                f"{config.run_id}.retrieval": json.dumps(
                    [{"doc_id": ref.doc_id, "page_numbers": ref.page_numbers} for ref in refs], ensure_ascii=False
                ),
            }
        )

    submission_path = builder.save(str(run_dir / "submission.json"))

    with (run_dir / "submission.csv").open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(submission_rows[0].keys()) if submission_rows else [])
        if submission_rows:
            writer.writeheader()
            writer.writerows(submission_rows)

    archive_path = run_dir / "code_archive.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in [ROOT_DIR / "run_experiments.py", ROOT_DIR / "plan_01.md", ROOT_DIR / "context.md"]:
            if path.exists():
                zf.write(path, arcname=path.name)

    metrics = {
        "run_id": config.run_id,
        "questions": len(sampled_questions),
        "docs": len(docs_rows),
        "retrieval_mode": config.retrieval_mode,
        "avg_ttft_ms": round(sum(ttft_values) / len(ttft_values), 2) if ttft_values else 0.0,
        "avg_total_time_ms": round(sum(total_values) / len(total_values), 2) if total_values else 0.0,
        "avg_retrieved_pages": round(sum(retrieval_pages_counts) / len(retrieval_pages_counts), 2)
        if retrieval_pages_counts
        else 0.0,
        "null_rate": round(null_count / len(sampled_questions), 4) if sampled_questions else 0.0,
        "input_tokens_total": total_input_tokens,
        "output_tokens_total": total_output_tokens,
        "estimated_cost_usd": _estimate_cost_usd(total_input_tokens, total_output_tokens),
        "submission_path": str(submission_path),
    }

    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "README.md").write_text(
        (
            f"# {config.run_id}\n\n"
            f"- parsing: PyMuPDF\n"
            f"- retrieval: {config.retrieval_mode}\n"
            f"- llm: {config.answer_model}\n"
            f"- embeddings: {config.embedding_model}\n"
            f"- chunk_size/chunk_overlap: {config.chunk_size}/{config.chunk_overlap}\n"
            f"- vector_top_k: {config.vector_top_k}\n"
            f"- lexical_top_k: {config.lexical_top_k}\n"
            f"- fused_top_k: {config.fused_top_k}\n"
            f"- question_limit: {config.question_limit}\n"
            f"- no submission sent to server\n"
        ),
        encoding="utf-8",
    )
    return metrics


def _save_global_results(results_root: Path, metrics_rows: list[dict[str, Any]]) -> None:
    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    file_path = results_dir / "experiments_summary.csv"
    with file_path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(metrics_rows[0].keys()) if metrics_rows else [])
        if metrics_rows:
            writer.writeheader()
            writer.writerows(metrics_rows)

    # Wide submission table
    wide_path = results_dir / "submission_wide.csv"
    merged: dict[str, dict[str, Any]] = {}
    for metrics in metrics_rows:
        run_dir = results_root / metrics["run_id"]
        submission_csv = run_dir / "submission.csv"
        if not submission_csv.exists():
            continue
        with submission_csv.open("r", encoding="utf-8", newline="") as file_handle:
            reader = csv.DictReader(file_handle)
            for row in reader:
                key = row["question_id"]
                merged.setdefault(
                    key,
                    {
                        "question_id": row["question_id"],
                        "answer_type": row["answer_type"],
                        "question": row["question"],
                    },
                )
                for col, value in row.items():
                    if col not in {"question_id", "answer_type", "question"}:
                        merged[key][col] = value

    if merged:
        all_fields = ["question_id", "answer_type", "question"]
        dynamic = sorted({k for row in merged.values() for k in row.keys()} - set(all_fields))
        with wide_path.open("w", encoding="utf-8", newline="") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=all_fields + dynamic)
            writer.writeheader()
            for row in merged.values():
                writer.writerow(row)


def main() -> None:
    cache_dir = ROOT_DIR / "cache" / "warmup"
    results_root = ROOT_DIR / "experiments"
    results_root.mkdir(parents=True, exist_ok=True)

    questions, docs_dir = _download_once(cache_dir)

    experiments = [
        ExperimentConfig(
            run_id="v1",
            retrieval_mode="hybrid",
            question_limit=25,
            vector_top_k=8,
            lexical_top_k=8,
            fused_top_k=6,
            answer_model="openai/gpt-4o-mini",
            embedding_model="openai/text-embedding-3-small",
            chunk_size=512,
            chunk_overlap=64,
        ),
        ExperimentConfig(
            run_id="v2",
            retrieval_mode="vector_only",
            question_limit=25,
            vector_top_k=8,
            lexical_top_k=0,
            fused_top_k=6,
            answer_model="openai/gpt-4o-mini",
            embedding_model="openai/text-embedding-3-small",
            chunk_size=512,
            chunk_overlap=64,
        ),
        ExperimentConfig(
            run_id="v3",
            retrieval_mode="hybrid",
            question_limit=25,
            vector_top_k=8,
            lexical_top_k=8,
            fused_top_k=6,
            answer_model="openai/gpt-4o-mini",
            embedding_model="openai/text-embedding-3-large",
            chunk_size=512,
            chunk_overlap=64,
        ),
    ]

    # Build parse artifacts once.
    prep_dir = results_root / "_prep"
    prep_dir.mkdir(parents=True, exist_ok=True)
    documents, pages, docs_rows = _build_docs_and_pages(docs_dir, prep_dir)

    # Build/reuse index per embedding model to keep embedding-module comparisons valid.
    index_by_embedding: dict[str, VectorStoreIndex] = {}
    unique_embedding_models = sorted({exp.embedding_model for exp in experiments})
    for emb_model in unique_embedding_models:
        tmp_config = ExperimentConfig(
            run_id="_index_build",
            retrieval_mode="vector_only",
            question_limit=1,
            vector_top_k=1,
            lexical_top_k=0,
            fused_top_k=1,
            answer_model=experiments[0].answer_model,
            embedding_model=emb_model,
            chunk_size=experiments[0].chunk_size,
            chunk_overlap=experiments[0].chunk_overlap,
        )
        _configure_models(tmp_config)
        print(f"Building index for embedding model: {emb_model}")
        index_by_embedding[emb_model] = VectorStoreIndex.from_documents(documents, show_progress=True)

    metrics_rows = []
    for exp in experiments:
        _configure_models(exp)
        print(f"Running {exp.run_id} ...")
        metrics = run_experiment(
            exp,
            questions,
            results_root,
            pages,
            docs_rows,
            index_by_embedding[exp.embedding_model],
            include_docs_list_copy_from=prep_dir / "docs_list.csv",
        )
        metrics_rows.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

    _save_global_results(results_root, metrics_rows)
    print("Done. No submission was sent.")


if __name__ == "__main__":
    main()
