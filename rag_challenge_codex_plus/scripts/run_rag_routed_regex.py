from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from openai import OpenAI as OpenAIClient

ROOT_DIR = Path(__file__).resolve().parents[1]
STARTER_DIR = ROOT_DIR / "starter_kit"
import sys

sys.path.append(str(STARTER_DIR))

from arlc import (  # noqa: E402
    RetrievalRef,
    SubmissionAnswer,
    SubmissionBuilder,
    Telemetry,
    TelemetryTimer,
    TimingMetrics,
    UsageMetrics,
    normalize_retrieved_pages,
)

ACTIVE_LLM = None


CASE_PATTERN = re.compile(r"\b([A-Z]{2,4})\s*0*([0-9]{1,4})\s*/\s*(20[0-9]{2})\b", re.IGNORECASE)
LAW_PATTERN = re.compile(r"\b(?:DIFC\s+)?Law\s+No\.?\s*([0-9]{1,3})\s*(?:of\s*(20[0-9]{2}))?\b", re.IGNORECASE)


class _StreamDelta:
    def __init__(self, delta: str) -> None:
        self.delta = delta


class _CompleteText:
    def __init__(self, text: str) -> None:
        self.text = text


class _DeepSeekLLMAdapter:
    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str,
        temperature: float = 0.0,
        reasoning_enabled: bool | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.reasoning_enabled = reasoning_enabled
        self.client = OpenAIClient(api_key=api_key, base_url=api_base)

    def _reasoning_kwargs(self) -> dict[str, Any]:
        if self.reasoning_enabled is None:
            return {}
        # OpenRouter reasoning control is passed via provider-specific body fields.
        return {"extra_body": {"reasoning": {"enabled": bool(self.reasoning_enabled)}}}

    def stream_complete(self, prompt: str):
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            stream=True,
            **self._reasoning_kwargs(),
        )
        for chunk in stream:
            delta = ""
            try:
                delta = chunk.choices[0].delta.content or ""
            except Exception:
                delta = ""
            if delta:
                yield _StreamDelta(delta)

    def complete(self, prompt: str):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            stream=False,
            **self._reasoning_kwargs(),
        )
        text = ""
        try:
            text = response.choices[0].message.content or ""
        except Exception:
            text = ""
        return _CompleteText(text)


def _approx_tokens(text: str) -> int:
    return max(1, int(len((text or "").strip()) / 4)) if (text or "").strip() else 0


def _collect_stream_text(prompt: str, max_attempts: int = 3, retry_sleep_seconds: float = 2.0) -> str:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        chunks: list[str] = []
        try:
            for ch in ACTIVE_LLM.stream_complete(prompt):
                chunks.append(ch.delta)
            return "".join(chunks).strip()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < max_attempts:
                time.sleep(retry_sleep_seconds * attempt)
                continue
            raise
    if last_error:
        raise last_error
    return ""


def _norm_model(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


def _configure(model: str, embedding_model: str, reasoning_enabled: bool | None = None) -> None:
    global ACTIVE_LLM
    load_dotenv(ROOT_DIR / ".env")
    model_lower = (model or "").lower()
    is_deepseek = "deepseek" in model_lower
    # Bare model names like "deepseek-chat" go to DeepSeek API (OPENAI_API_*).
    # Provider-qualified names like "deepseek/deepseek-v3.2" go through OpenRouter.
    is_openrouter_deepseek = is_deepseek and "/" in model
    if is_deepseek and not is_openrouter_deepseek:
        llm_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        llm_base = (os.getenv("OPENAI_API_BASE") or "").strip()
        if not llm_key or not llm_base:
            raise RuntimeError("OPENAI_API_KEY and OPENAI_API_BASE are required for deepseek-chat.")
    else:
        llm_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
        llm_base = (os.getenv("OPENROUTER_API_BASE") or "https://openrouter.ai/api/v1").strip()
        if not llm_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for non-deepseek models.")

    emb_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    emb_base = (os.getenv("OPENROUTER_API_BASE") or "https://openrouter.ai/api/v1").strip()
    if not emb_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for embeddings.")

    if is_deepseek:
        ACTIVE_LLM = _DeepSeekLLMAdapter(
            model=model,
            api_key=llm_key,
            api_base=llm_base,
            temperature=0.0,
            reasoning_enabled=reasoning_enabled if is_openrouter_deepseek else None,
        )
    else:
        ACTIVE_LLM = OpenAI(model=_norm_model(model), api_key=llm_key, api_base=llm_base, temperature=0.0)
        Settings.llm = ACTIVE_LLM
    Settings.embed_model = OpenAIEmbedding(model=_norm_model(embedding_model), api_key=emb_key, api_base=emb_base)


def _normalize_doc_number(value: str) -> str:
    value = re.sub(r"\s+", " ", (value or "").strip()).upper()
    m = CASE_PATTERN.search(value)
    if m:
        return f"{m.group(1).upper()} {int(m.group(2)):03d}/{m.group(3)}"
    m = LAW_PATTERN.search(value)
    if m:
        if m.group(2):
            return f"LAW NO. {int(m.group(1))} OF {m.group(2)}"
        return f"LAW NO. {int(m.group(1))}"
    return value


def _extract_doc_numbers(question: str) -> list[str]:
    out: list[str] = []
    for m in CASE_PATTERN.finditer(question):
        out.append(f"{m.group(1).upper()} {int(m.group(2)):03d}/{m.group(3)}")
    for m in LAW_PATTERN.finditer(question):
        if m.group(2):
            out.append(f"LAW NO. {int(m.group(1))} OF {m.group(2)}")
        else:
            out.append(f"LAW NO. {int(m.group(1))}")
    return sorted(set(out))


def _build_doc_maps(docs_list_path: Path) -> tuple[dict[str, dict[str, str]], dict[str, list[str]]]:
    rows = list(csv.DictReader(docs_list_path.open("r", encoding="utf-8", newline="")))
    doc_by_id = {r["doc_id"]: r for r in rows}
    number_to_doc_ids: dict[str, list[str]] = {}
    for r in rows:
        normalized = _normalize_doc_number(r.get("document_number", ""))
        if not normalized:
            continue
        number_to_doc_ids.setdefault(normalized, []).append(r["doc_id"])
    return doc_by_id, number_to_doc_ids


def _direct_docs_list_answer(question: str, answer_type: str, candidate_doc_rows: list[dict[str, str]]):
    q = question.lower()
    if not candidate_doc_rows:
        return None
    row = candidate_doc_rows[0]

    if "date of issue" in q and answer_type == "date":
        return row.get("date") or None
    if ("defendant" in q and answer_type == "name") or ("who is the defendant" in q):
        return row.get("defendant") or None
    if ("claimant" in q and answer_type == "name") or ("plaintiff" in q and answer_type == "name"):
        return row.get("claimant") or None
    if "official difc law number" in q and answer_type == "number":
        law = row.get("document_number", "")
        m = re.search(r"LAW NO\.\s*([0-9]{1,3})", law.upper())
        if m:
            return float(m.group(1))
    return None


def _direct_docs_list_answer_llm(question: str, answer_type: str, candidate_doc_rows: list[dict[str, str]]):
    if not candidate_doc_rows:
        return None, TimingMetrics(ttft_ms=0, tpot_ms=0, total_time_ms=0), 0, 0

    metadata_payload = []
    for row in candidate_doc_rows[:5]:
        metadata_payload.append(
            {
                "doc_id": row.get("doc_id", ""),
                "document_number": row.get("document_number", ""),
                "date": row.get("date", ""),
                "title": row.get("title", ""),
                "type": row.get("type", ""),
                "claimant": row.get("claimant", ""),
                "defendant": row.get("defendant", ""),
            }
        )

    prompt = (
        "You answer only from provided document metadata rows.\n"
        "If answer cannot be determined from metadata, return null.\n"
        f"answer_type: {answer_type}\n"
        f"question: {question}\n\n"
        f"metadata_rows:\n{json.dumps(metadata_payload, ensure_ascii=False, indent=2)}\n\n"
        "Return only the final answer value."
    )

    timer = TelemetryTimer()
    response = _collect_stream_text(prompt)
    if response:
        timer.mark_token()
    answer = _parse_answer(response, answer_type)
    timing = timer.finish()
    return answer, timing, _approx_tokens(prompt), _approx_tokens(response)


def _direct_docs_list_router_llm(
    question: str,
    answer_type: str,
    all_doc_rows: list[dict[str, str]],
):
    # LLM router contract:
    # 1) answer mode -> return answer from docs_list metadata only
    # 2) docs mode   -> return candidate doc_ids/document_numbers for retrieval branch
    if not all_doc_rows:
        return (
            {"decision": "docs", "answer": None, "doc_ids": [], "document_numbers": []},
            TimingMetrics(ttft_ms=0, tpot_ms=0, total_time_ms=0),
            0,
            0,
        )

    metadata_payload = []
    for row in all_doc_rows:
        metadata_payload.append(
            {
                "doc_id": row.get("doc_id", ""),
                "document_number": row.get("document_number", ""),
                "date": row.get("date", ""),
                "title": row.get("title", ""),
                "type": row.get("type", ""),
                "claimant": row.get("claimant", ""),
                "defendant": row.get("defendant", ""),
            }
        )

    prompt = (
        "You are a router for legal QA using only docs_list metadata.\n"
        "Strictly return JSON with this schema:\n"
        '{'
        '"decision":"answer"|"docs",'
        '"answer": <value or null>,'
        '"doc_ids": ["doc_id", ...],'
        '"document_numbers": ["normalized number", ...]'
        '}\n'
        "Rules:\n"
        "- If metadata is enough to answer with high confidence, set decision='answer'.\n"
        "- Otherwise set decision='docs' and return 1-6 likely documents.\n"
        "- doc_ids must be chosen from provided metadata list.\n"
        "- For decision='docs', answer must be null.\n\n"
        f"answer_type: {answer_type}\n"
        f"question: {question}\n\n"
        f"metadata_rows:\n{json.dumps(metadata_payload, ensure_ascii=False)}\n"
    )

    timer = TelemetryTimer()
    response = _collect_stream_text(prompt)
    if response:
        timer.mark_token()
    timing = timer.finish()

    payload = {"decision": "docs", "answer": None, "doc_ids": [], "document_numbers": []}
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                payload.update(parsed)
        except json.JSONDecodeError:
            pass
    return payload, timing, _approx_tokens(prompt), _approx_tokens(response)


def _parse_answer(raw: str, answer_type: str):
    text = (raw or "").strip()
    if not text:
        return None if answer_type != "free_text" else "There is no information on this question in the provided documents."
    if text.lower() == "null":
        return None
    if answer_type == "boolean":
        low = text.lower()
        if low in {"true", "yes", "1"}:
            return True
        if low in {"false", "no", "0"}:
            return False
        return None
    if answer_type == "number":
        try:
            return float(text.replace(",", "."))
        except ValueError:
            return None
    if answer_type == "date":
        m = re.search(r"\d{4}-\d{2}-\d{2}", text)
        return m.group(0) if m else None
    if answer_type == "names":
        vals = [x.strip() for x in re.split(r"[;,]", text) if x.strip()]
        return vals if vals else None
    if answer_type == "free_text":
        return text[:280]
    return text


def _retrieval_refs(evidence: list[dict[str, Any]]) -> list[RetrievalRef]:
    refs = [RetrievalRef(doc_id=e["doc_id"], page_numbers=[int(e["page_number"])]) for e in evidence]
    return normalize_retrieved_pages(refs)


def _answer_from_evidence_no_llm(question: str, answer_type: str, evidence: list[dict[str, Any]]):
    text = "\n".join((e.get("text") or "")[:2200] for e in evidence)
    if answer_type == "date":
        m = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
        return m.group(0) if m else None
    if answer_type == "number":
        m = re.search(r"\b\d+(?:\.\d+)?\b", text)
        return float(m.group(0)) if m else None
    if answer_type == "name":
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for ln in lines:
            if len(ln.split()) <= 8 and re.search(r"[A-Za-z]", ln):
                return ln[:120]
        return None
    if answer_type == "names":
        cands = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", text)
        uniq = []
        seen = set()
        for c in cands:
            s = c.strip()
            if s.lower() in seen:
                continue
            seen.add(s.lower())
            uniq.append(s)
            if len(uniq) >= 5:
                break
        return uniq if uniq else None
    if answer_type == "boolean":
        q = question.lower()
        t = text.lower()
        neg_hits = ["not permitted", "shall not", "is prohibited", "must not", "no person shall"]
        pos_hits = ["is permitted", "may", "is allowed", "can"]
        if any(h in t for h in neg_hits):
            return False
        if any(h in t for h in pos_hits):
            return True
        if "not" in q and any(h in t for h in pos_hits):
            return False
        return None
    if answer_type == "free_text":
        return text[:280] if text else None
    return None


def _llm_rerank_candidates(
    question: str,
    answer_type: str,
    candidates: list[dict[str, Any]],
    max_evidence_pages: int,
) -> tuple[list[dict[str, Any]], TimingMetrics, int, int]:
    if not candidates:
        return [], TimingMetrics(ttft_ms=0, tpot_ms=0, total_time_ms=0), 0, 0
    block_lines = []
    for idx, c in enumerate(candidates, 1):
        block_lines.append(
            f"[{idx}] doc_id={c['doc_id']} page={c['page_number']}\n"
            f"{(c.get('text') or '')[:1200]}"
        )
    prompt = (
        "You are a reranker for legal RAG evidence selection.\n"
        f"Question: {question}\n"
        f"Answer type: {answer_type}\n"
        f"Select up to {max_evidence_pages} most relevant candidates.\n"
        "Return strict JSON: {\"selected_ids\": [1,2,...]}.\n\n"
        "Candidates:\n"
        + "\n\n".join(block_lines)
    )

    timer = TelemetryTimer()
    response = _collect_stream_text(prompt)
    if response:
        timer.mark_token()
    timing = timer.finish()
    in_toks, out_toks = _approx_tokens(prompt), _approx_tokens(response)
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        return candidates[:max_evidence_pages], timing, in_toks, out_toks
    try:
        payload = json.loads(match.group(0))
        ids = payload.get("selected_ids", [])
        if not isinstance(ids, list):
            return candidates[:max_evidence_pages], timing, in_toks, out_toks
        selected = []
        seen = set()
        for raw in ids:
            try:
                idx = int(raw)
            except Exception:
                continue
            if idx < 1 or idx > len(candidates):
                continue
            if idx in seen:
                continue
            seen.add(idx)
            selected.append(candidates[idx - 1])
            if len(selected) >= max_evidence_pages:
                break
        return (selected if selected else candidates[:max_evidence_pages]), timing, in_toks, out_toks
    except json.JSONDecodeError:
        return candidates[:max_evidence_pages], timing, in_toks, out_toks


def run(
    questions_path: Path,
    docs_list_path: Path,
    indices_dir: Path,
    out_dir: Path,
    model: str,
    embedding_model: str,
    question_limit: int,
    top_k_per_doc: int,
    max_evidence_pages: int,
    run_label: str,
    direct_mode: str,
    retrieval_strategy: str,
    dynamic_bump_top_k: int,
    dynamic_min_candidates: int,
    max_rerank_candidates: int,
    global_fallback_on_direct_null: bool,
    disable_retrieval: bool,
    llm_only_for_free_text: bool,
    reasoning_enabled: bool | None,
) -> None:
    _configure(model, embedding_model, reasoning_enabled=reasoning_enabled)
    out_dir.mkdir(parents=True, exist_ok=True)

    questions = json.loads(questions_path.read_text(encoding="utf-8"))[:question_limit]
    doc_by_id, number_to_doc_ids = _build_doc_maps(docs_list_path)
    all_doc_rows = list(doc_by_id.values())

    # lazy index cache
    indices: dict[str, Any] = {}

    builder = SubmissionBuilder(
        architecture_summary=f"{run_label} regex-routed: docs_list-first ({direct_mode}) + per-document indices + constrained retrieval",
    )
    rows_out: list[dict[str, str]] = []

    def _save_progress() -> None:
        with (out_dir / "submission.progress.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()) if rows_out else [])
            if rows_out:
                writer.writeheader()
                writer.writerows(rows_out)
        builder.save(str(out_dir / "submission.progress.json"))

    for i, q in enumerate(questions, 1):
        qid = q["id"]
        question = q["question"]
        answer_type = str(q.get("answer_type", "free_text")).lower()
        print(f"[{i}/{len(questions)}] {qid}")
        agg_input_toks = 0
        agg_output_toks = 0
        agg_timing = TimingMetrics(ttft_ms=0, tpot_ms=0, total_time_ms=0)

        def _add_llm_usage(step_timing: TimingMetrics, in_toks: int, out_toks: int) -> None:
            nonlocal agg_input_toks, agg_output_toks, agg_timing
            agg_input_toks += int(in_toks or 0)
            agg_output_toks += int(out_toks or 0)
            agg_timing = TimingMetrics(
                ttft_ms=agg_timing.ttft_ms + int(step_timing.ttft_ms or 0),
                tpot_ms=agg_timing.tpot_ms + int(step_timing.tpot_ms or 0),
                total_time_ms=agg_timing.total_time_ms + int(step_timing.total_time_ms or 0),
            )

        doc_numbers = _extract_doc_numbers(question)
        target_doc_ids: list[str] = []
        for num in doc_numbers:
            target_doc_ids.extend(number_to_doc_ids.get(_normalize_doc_number(num), []))
        target_doc_ids = sorted(set(target_doc_ids))
        target_rows = [doc_by_id[d] for d in target_doc_ids if d in doc_by_id]

        # Route 1: direct from docs_list
        force_global_retrieval = False
        if direct_mode == "llm":
            direct, direct_timing, direct_in_toks, direct_out_toks = _direct_docs_list_answer_llm(
                question, answer_type, target_rows
            )
            _add_llm_usage(direct_timing, direct_in_toks, direct_out_toks)
            if direct is None and global_fallback_on_direct_null:
                # User rule: if direct-LLM returns null, search over all document DBs.
                force_global_retrieval = True
        elif direct_mode == "llm_router":
            routed_payload, direct_timing, direct_in_toks, direct_out_toks = _direct_docs_list_router_llm(
                question=question,
                answer_type=answer_type,
                all_doc_rows=all_doc_rows,
            )
            _add_llm_usage(direct_timing, direct_in_toks, direct_out_toks)
            decision = str(routed_payload.get("decision", "docs")).strip().lower()
            if decision == "answer":
                direct = _parse_answer(str(routed_payload.get("answer", "") or ""), answer_type)
                if direct is None and global_fallback_on_direct_null:
                    # User rule: if direct-LLM returns null, search over all document DBs.
                    force_global_retrieval = True
            else:
                direct = None
                llm_doc_ids = []
                for raw in routed_payload.get("doc_ids", []) if isinstance(routed_payload.get("doc_ids", []), list) else []:
                    sid = str(raw).strip()
                    if sid and sid in doc_by_id:
                        llm_doc_ids.append(sid)
                llm_doc_numbers = []
                for raw in (
                    routed_payload.get("document_numbers", [])
                    if isinstance(routed_payload.get("document_numbers", []), list)
                    else []
                ):
                    s = _normalize_doc_number(str(raw))
                    if s:
                        llm_doc_numbers.append(s)
                for num in llm_doc_numbers:
                    llm_doc_ids.extend(number_to_doc_ids.get(num, []))
                llm_doc_ids = sorted(set(llm_doc_ids))
                if llm_doc_ids:
                    target_doc_ids = llm_doc_ids
                    target_rows = [doc_by_id[d] for d in target_doc_ids if d in doc_by_id]
                    doc_numbers = sorted(set(llm_doc_numbers)) if llm_doc_numbers else doc_numbers
        else:
            direct = _direct_docs_list_answer(question, answer_type, target_rows)
            direct_timing = TimingMetrics(ttft_ms=0, tpot_ms=0, total_time_ms=0)
            direct_in_toks, direct_out_toks = 0, 0
        if direct is not None:
            answer = direct
            refs = []
            timing = direct_timing
            route_source = f"docs_list_{direct_mode}"
        else:
            if disable_retrieval:
                answer = None if answer_type != "free_text" else "There is no information on this question in the provided documents."
                refs = []
                timing = TimingMetrics(ttft_ms=0, tpot_ms=0, total_time_ms=0)
                route_source = "fallback_retrieval_disabled"
                telemetry = Telemetry(
                    timing=TimingMetrics(
                        ttft_ms=agg_timing.ttft_ms,
                        tpot_ms=agg_timing.tpot_ms,
                        total_time_ms=agg_timing.total_time_ms,
                    ),
                    retrieval=refs,
                    usage=UsageMetrics(input_tokens=agg_input_toks, output_tokens=agg_output_toks),
                    model_name=model,
                )
                builder.add_answer(SubmissionAnswer(question_id=qid, answer=answer, telemetry=telemetry))
                rows_out.append(
                    {
                        "question_id": qid,
                        "answer_type": answer_type,
                        "question": question,
                        f"{run_label}.answer": json.dumps(answer, ensure_ascii=False),
                        f"{run_label}.retrieval": json.dumps([], ensure_ascii=False),
                        "routed_doc_ids": ";".join(target_doc_ids),
                        "routed_doc_numbers": ";".join(doc_numbers),
                        "route_source": route_source,
                    }
                )
                continue
            # Route 2: constrained retrieval from target documents only.
            evidence = []
            if force_global_retrieval:
                target_doc_ids = sorted(doc_by_id.keys())
                target_rows = [doc_by_id[d] for d in target_doc_ids if d in doc_by_id]
            if not target_doc_ids:
                # fallback: use no answer to avoid high-noise global retrieval
                answer = None if answer_type != "free_text" else "There is no information on this question in the provided documents."
                refs = []
                timing = TimingMetrics(ttft_ms=0, tpot_ms=0, total_time_ms=0)
                route_source = "fallback_no_target_docs"
            else:
                for doc_id in target_doc_ids:
                    persist = indices_dir / doc_id
                    if not persist.exists():
                        continue
                    if doc_id not in indices:
                        storage = StorageContext.from_defaults(persist_dir=str(persist))
                        indices[doc_id] = load_index_from_storage(storage)
                    retriever = indices[doc_id].as_retriever(similarity_top_k=top_k_per_doc)
                    nodes = retriever.retrieve(question)
                    for n in nodes:
                        md = n.metadata or {}
                        evidence.append(
                            {
                                "doc_id": md.get("doc_id", doc_id),
                                "page_number": int(md.get("page_number", 0) or 0),
                                "text": n.text,
                                "score": float(getattr(n, "score", 0.0) or 0.0),
                            }
                        )
                    if retrieval_strategy == "dynamic" and len(nodes) < dynamic_min_candidates:
                        bump_k = max(dynamic_bump_top_k, top_k_per_doc)
                        bump_nodes = indices[doc_id].as_retriever(similarity_top_k=bump_k).retrieve(question)
                        for n in bump_nodes:
                            md = n.metadata or {}
                            evidence.append(
                                {
                                    "doc_id": md.get("doc_id", doc_id),
                                    "page_number": int(md.get("page_number", 0) or 0),
                                    "text": n.text,
                                    "score": float(getattr(n, "score", 0.0) or 0.0),
                                }
                            )
                evidence = [e for e in evidence if e["page_number"] > 0]
                # de-duplicate by (doc_id, page_number) keeping best score.
                best: dict[tuple[str, int], dict[str, Any]] = {}
                for e in evidence:
                    key = (e["doc_id"], int(e["page_number"]))
                    if key not in best or e["score"] > best[key]["score"]:
                        best[key] = e
                evidence = list(best.values())
                evidence.sort(key=lambda x: x["score"], reverse=True)
                if retrieval_strategy == "llm_rerank":
                    pool = evidence[: max(1, max_rerank_candidates)]
                    evidence, rerank_timing, rerank_in_toks, rerank_out_toks = _llm_rerank_candidates(
                        question=question,
                        answer_type=answer_type,
                        candidates=pool,
                        max_evidence_pages=max_evidence_pages,
                    )
                    _add_llm_usage(rerank_timing, rerank_in_toks, rerank_out_toks)
                else:
                    evidence = evidence[:max_evidence_pages]

                if not evidence:
                    answer = None if answer_type != "free_text" else "There is no information on this question in the provided documents."
                    refs = []
                    timing = TimingMetrics(ttft_ms=0, tpot_ms=0, total_time_ms=0)
                    route_source = "fallback_no_evidence"
                else:
                    context = "\n\n".join(
                        f"[doc:{e['doc_id']} page:{e['page_number']}]\n{e['text'][:2200]}" for e in evidence
                    )
                    refs = _retrieval_refs(evidence)
                    if llm_only_for_free_text and answer_type != "free_text":
                        answer = _answer_from_evidence_no_llm(question, answer_type, evidence)
                        timing = TimingMetrics(ttft_ms=0, tpot_ms=0, total_time_ms=0)
                        route_source = "retrieval_rules"
                    else:
                        prompt = (
                            "Answer using only the context.\n"
                            f"Question: {question}\n"
                            f"Answer type: {answer_type}\n"
                            "If missing, return null for deterministic types.\n\n"
                            f"Context:\n{context}\n\nAnswer:"
                        )
                        timer = TelemetryTimer()
                        response = _collect_stream_text(prompt)
                        if response:
                            timer.mark_token()
                        answer = _parse_answer(response, answer_type)
                        timing = timer.finish()
                        _add_llm_usage(timing, _approx_tokens(prompt), _approx_tokens(response))
                        route_source = "retrieval"

        telemetry = Telemetry(
            timing=TimingMetrics(
                ttft_ms=agg_timing.ttft_ms,
                tpot_ms=agg_timing.tpot_ms,
                total_time_ms=agg_timing.total_time_ms,
            ),
            retrieval=refs,
            usage=UsageMetrics(input_tokens=agg_input_toks, output_tokens=agg_output_toks),
            model_name=model,
        )
        builder.add_answer(SubmissionAnswer(question_id=qid, answer=answer, telemetry=telemetry))

        rows_out.append(
            {
                "question_id": qid,
                "answer_type": answer_type,
                "question": question,
                f"{run_label}.answer": json.dumps(answer, ensure_ascii=False),
                f"{run_label}.retrieval": json.dumps(
                    [{"doc_id": r.doc_id, "page_numbers": r.page_numbers} for r in refs], ensure_ascii=False
                ),
                "routed_doc_ids": ";".join(target_doc_ids),
                "routed_doc_numbers": ";".join(doc_numbers),
                "route_source": route_source,
            }
        )
        _save_progress()

    submission_path = builder.save(str(out_dir / "submission.json"))
    with (out_dir / "submission.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()) if rows_out else [])
        if rows_out:
            writer.writeheader()
            writer.writerows(rows_out)

    with zipfile.ZipFile(out_dir / "code_archive.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for p in [
            ROOT_DIR / "scripts" / "build_docs_list.py",
            ROOT_DIR / "scripts" / "build_document_indices.py",
            ROOT_DIR / "scripts" / "run_rag_routed_regex.py",
        ]:
            zf.write(p, arcname=p.relative_to(ROOT_DIR))
    print(f"saved: {submission_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run routed RAG using regex docs_list and per-doc indices.")
    parser.add_argument("--questions", default="cache/warmup/questions.json")
    parser.add_argument("--docs-list", default="artifacts/docs_list/v1_regex/docs_list.csv")
    parser.add_argument("--indices-dir", default="artifacts/document_indices/v1_regex_small")
    parser.add_argument("--out-dir", default="experiments/v4")
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--embedding-model", default="openai/text-embedding-3-small")
    parser.add_argument("--question-limit", type=int, default=25)
    parser.add_argument("--top-k-per-doc", type=int, default=3)
    parser.add_argument("--max-evidence-pages", type=int, default=3)
    parser.add_argument("--run-label", default="v4")
    parser.add_argument("--direct-mode", choices=["rules", "llm", "llm_router"], default="rules")
    parser.add_argument("--retrieval-strategy", choices=["baseline", "dynamic", "llm_rerank"], default="baseline")
    parser.add_argument("--dynamic-bump-top-k", type=int, default=8)
    parser.add_argument("--dynamic-min-candidates", type=int, default=6)
    parser.add_argument("--max-rerank-candidates", type=int, default=24)
    parser.add_argument("--no-global-fallback-on-direct-null", action="store_true")
    parser.add_argument("--disable-retrieval", action="store_true")
    parser.add_argument("--llm-only-for-free-text", action="store_true")
    parser.add_argument("--reasoning-enabled", action="store_true")
    parser.add_argument("--reasoning-disabled", action="store_true")
    args = parser.parse_args()

    if args.reasoning_enabled and args.reasoning_disabled:
        parser.error("Use only one of --reasoning-enabled / --reasoning-disabled.")
    reasoning_enabled: bool | None = None
    if args.reasoning_enabled:
        reasoning_enabled = True
    elif args.reasoning_disabled:
        reasoning_enabled = False
    elif str(args.model).lower().startswith("deepseek/"):
        # Default OpenRouter DeepSeek to non-reasoning mode to mimic deepseek-chat.
        reasoning_enabled = False

    run(
        questions_path=Path(args.questions).resolve(),
        docs_list_path=Path(args.docs_list).resolve(),
        indices_dir=Path(args.indices_dir).resolve(),
        out_dir=Path(args.out_dir).resolve(),
        model=args.model,
        embedding_model=args.embedding_model,
        question_limit=args.question_limit,
        top_k_per_doc=args.top_k_per_doc,
        max_evidence_pages=args.max_evidence_pages,
        run_label=args.run_label,
        direct_mode=args.direct_mode,
        retrieval_strategy=args.retrieval_strategy,
        dynamic_bump_top_k=args.dynamic_bump_top_k,
        dynamic_min_candidates=args.dynamic_min_candidates,
        max_rerank_candidates=args.max_rerank_candidates,
        global_fallback_on_direct_null=not args.no_global_fallback_on_direct_null,
        disable_retrieval=args.disable_retrieval,
        llm_only_for_free_text=args.llm_only_for_free_text,
        reasoning_enabled=reasoning_enabled,
    )


if __name__ == "__main__":
    main()
