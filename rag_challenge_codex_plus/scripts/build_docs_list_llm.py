from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from build_docs_list import (  # type: ignore
    _classify_type,
    _confidence,
    _extract_party,
    _extract_text_pages,
    _pick_date,
    _pick_document_number,
    _pick_title,
)


@dataclass
class DocRowLLM:
    db_name: str
    doc_id: str
    file_name: str
    title: str
    document_number: str
    document_number_candidates: str
    date: str
    type: str
    claimant: str
    defendant: str
    total_pages: int
    text_coverage_ratio: float
    confidence: str
    confidence_score: int
    extraction_mode: str


SYSTEM_PROMPT = (
    "You extract structured metadata from legal documents. "
    "Return strict JSON only with keys: "
    "document_number, date, title, type, claimant, defendant."
)


def _normalize_ws(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _extract_json_block(text: str) -> dict[str, Any]:
    if not text:
        return {}
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def _call_llm_extract(client: OpenAI, model: str, text: str) -> dict[str, Any]:
    user_prompt = (
        "Extract metadata from this legal document text.\n\n"
        "Rules:\n"
        "- document_number must be like 'CFI 057/2025' or 'Law No. 3 of 2018' when available.\n"
        "- date should be YYYY-MM-DD if found.\n"
        "- type should be one of: Judgment, Law, Order, Enforcement Document, Other.\n"
        "- Use empty string when not found.\n"
        "- Output JSON only.\n\n"
        f"TEXT:\n{text[:12000]}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    content = (response.choices[0].message.content or "").strip()
    return _extract_json_block(content)


def build_docs_list_llm(
    docs_dir: Path,
    out_dir: Path,
    model: str = "openai/gpt-4o-mini",
) -> tuple[Path, Path, Path]:
    load_dotenv(Path(".env"))
    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    api_base = (os.getenv("OPENROUTER_API_BASE") or "https://openrouter.ai/api/v1").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for LLM docs_list extraction.")

    client = OpenAI(api_key=api_key, base_url=api_base)
    pdf_files = sorted(docs_dir.glob("*.pdf"))
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[DocRowLLM] = []

    for pdf in pdf_files:
        pages, non_empty = _extract_text_pages(pdf)
        header_text = "\n".join(pages[:4])
        total_pages = len(pages)
        coverage = round(non_empty / total_pages, 4) if total_pages else 0.0

        llm = _call_llm_extract(client, model, header_text)

        llm_doc_number = _normalize_ws(str(llm.get("document_number", "")))
        regex_doc_number, regex_candidates = _pick_document_number(header_text)
        # Guarantee non-empty using deterministic fallback.
        document_number = llm_doc_number or regex_doc_number

        llm_date = _normalize_ws(str(llm.get("date", "")))
        llm_title = _normalize_ws(str(llm.get("title", "")))
        llm_type = _normalize_ws(str(llm.get("type", "")))
        llm_claimant = _normalize_ws(str(llm.get("claimant", "")))
        llm_defendant = _normalize_ws(str(llm.get("defendant", "")))

        date = llm_date or _pick_date(header_text)
        title = llm_title or _pick_title(pages[0] if pages else "")
        doc_type = llm_type if llm_type in {"Judgment", "Law", "Order", "Enforcement Document", "Other"} else _classify_type(header_text)
        claimant = llm_claimant or _extract_party(
            [
                r"Claimant(?:s)?\s*[:\-]\s*([^\n]{2,180})",
                r"Plaintiff(?:s)?\s*[:\-]\s*([^\n]{2,180})",
                r"Applicant(?:s)?\s*[:\-]\s*([^\n]{2,180})",
            ],
            header_text,
        )
        defendant = llm_defendant or _extract_party(
            [
                r"Defendant(?:s)?\s*[:\-]\s*([^\n]{2,180})",
                r"Respondent(?:s)?\s*[:\-]\s*([^\n]{2,180})",
            ],
            header_text,
        )
        confidence, confidence_score = _confidence(document_number, title, date, claimant, defendant, doc_type)

        mode = "llm"
        if not llm_doc_number and regex_doc_number:
            mode = "llm+regex_fallback"

        rows.append(
            DocRowLLM(
                db_name=pdf.stem,
                doc_id=pdf.stem,
                file_name=pdf.name,
                title=title,
                document_number=document_number,
                document_number_candidates="; ".join(regex_candidates),
                date=date,
                type=doc_type,
                claimant=claimant,
                defendant=defendant,
                total_pages=total_pages,
                text_coverage_ratio=coverage,
                confidence=confidence,
                confidence_score=confidence_score,
                extraction_mode=mode,
            )
        )

    csv_path = out_dir / "docs_list.csv"
    json_path = out_dir / "docs_list.json"
    low_conf_path = out_dir / "docs_list_low_confidence.csv"

    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
            writer.writeheader()
            writer.writerows(asdict(r) for r in rows)

        with json_path.open("w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in rows], f, ensure_ascii=False, indent=2)

        low_rows = [r for r in rows if r.confidence == "low" or not r.document_number]
        with low_conf_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
            writer.writeheader()
            writer.writerows(asdict(r) for r in low_rows)
    else:
        csv_path.write_text("", encoding="utf-8")
        json_path.write_text("[]", encoding="utf-8")
        low_conf_path.write_text("", encoding="utf-8")

    return csv_path, json_path, low_conf_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build docs_list with LLM extraction.")
    parser.add_argument("--docs-dir", default="cache/warmup/docs_corpus")
    parser.add_argument("--out-dir", default="artifacts/docs_list/v2_llm")
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    csv_path, json_path, low_path = build_docs_list_llm(docs_dir, out_dir, model=args.model)
    print(f"docs_list csv: {csv_path}")
    print(f"docs_list json: {json_path}")
    print(f"low confidence: {low_path}")


if __name__ == "__main__":
    main()

