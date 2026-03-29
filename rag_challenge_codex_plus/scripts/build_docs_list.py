from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import fitz  # pymupdf


CASE_PATTERN = re.compile(r"\b([A-Z]{2,4})\s*0*([0-9]{1,4})\s*/\s*(20[0-9]{2})\b", re.IGNORECASE)
PREFERRED_CASE_PREFIXES = ("CFI", "CA", "SCT", "ARB", "ENF", "DEC", "TCD")
LAW_PATTERN = re.compile(
    r"\b(?:DIFC\s+)?Law\s+No\.?\s*([0-9]{1,3})\s*(?:of\s*([0-9]{4}))?\b",
    re.IGNORECASE,
)
DATE_ISO_PATTERN = re.compile(r"\b(20[0-9]{2})-(0[1-9]|1[0-2])-([0-2][0-9]|3[01])\b")
DATE_DMY_SLASH = re.compile(r"\b([0-2][0-9]|3[01])[./-](0[1-9]|1[0-2])[./-](20[0-9]{2})\b")
DATE_MONTH_WORD = re.compile(
    r"\b([0-2]?[0-9]|3[01])\s+"
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(20[0-9]{2})\b",
    re.IGNORECASE,
)


@dataclass
class DocRow:
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


def _normalize_ws(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _extract_text_pages(pdf_path: Path) -> tuple[list[str], int]:
    doc = fitz.open(pdf_path)
    pages: list[str] = []
    non_empty = 0
    for i in range(doc.page_count):
        text = _normalize_ws(doc[i].get_text("text"))
        if text:
            non_empty += 1
        pages.append(text)
    total = doc.page_count
    doc.close()
    return pages, non_empty if total else 0


def _collect_case_numbers(text: str) -> list[str]:
    preferred: list[str] = []
    other: list[str] = []
    for m in CASE_PATTERN.finditer(text):
        prefix = m.group(1).upper()
        num = int(m.group(2))
        year = m.group(3)
        value = f"{prefix} {num:03d}/{year}"
        if prefix in PREFERRED_CASE_PREFIXES:
            preferred.append(value)
        else:
            other.append(value)
    preferred = sorted(set(preferred))
    other = sorted(set(other))
    return preferred + other


def _collect_law_numbers(text: str) -> list[str]:
    out = []
    for m in LAW_PATTERN.finditer(text):
        number = int(m.group(1))
        year = m.group(2)
        if year:
            out.append(f"Law No. {number} of {year}")
        else:
            out.append(f"Law No. {number}")
    return sorted(set(out))


def _pick_document_number(text: str) -> tuple[str, list[str]]:
    case_numbers = _collect_case_numbers(text)
    law_numbers = _collect_law_numbers(text)
    candidates = case_numbers + law_numbers
    if case_numbers:
        return case_numbers[0], candidates
    if law_numbers:
        return law_numbers[0], candidates
    return "", candidates


def _parse_dates(text: str) -> list[str]:
    out: set[str] = set()
    for m in DATE_ISO_PATTERN.finditer(text):
        out.add(m.group(0))
    for m in DATE_DMY_SLASH.finditer(text):
        d, mo, y = m.group(1), m.group(2), m.group(3)
        out.add(f"{y}-{int(mo):02d}-{int(d):02d}")
    for m in DATE_MONTH_WORD.finditer(text):
        d, month_name, y = m.group(1), m.group(2), m.group(3)
        try:
            dt = datetime.strptime(f"{int(d)} {month_name} {y}", "%d %B %Y")
            out.add(dt.date().isoformat())
        except ValueError:
            continue
    return sorted(out)


def _pick_date(text: str) -> str:
    dates = _parse_dates(text)
    if not dates:
        return ""
    # Prefer latest date in legal docs header sections for issue date heuristics.
    return dates[-1]


def _pick_title(page0_text: str) -> str:
    lines = [ln.strip() for ln in re.split(r"[\\n\\.]", page0_text) if ln.strip()]
    bad_prefixes = ("page ", "date", "issued", "before ", "in the matter")
    for line in lines[:40]:
        low = line.lower()
        if len(line) < 8:
            continue
        if any(low.startswith(prefix) for prefix in bad_prefixes):
            continue
        if CASE_PATTERN.search(line):
            continue
        return line[:200]
    return ""


def _extract_party(patterns: Iterable[str], text: str) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _normalize_ws(match.group(1))[:180]
    return ""


def _classify_type(text: str) -> str:
    low = text.lower()
    if "judgment" in low:
        return "Judgment"
    if "order" in low:
        return "Order"
    if "enforcement" in low:
        return "Enforcement Document"
    if "law no." in low or "difc law" in low:
        return "Law"
    return "Other"


def _confidence(document_number: str, title: str, date: str, claimant: str, defendant: str, doc_type: str) -> tuple[str, int]:
    score = 0
    if document_number:
        score += 4
    if title:
        score += 2
    if date:
        score += 2
    if claimant or defendant:
        score += 1
    if doc_type and doc_type != "Other":
        score += 1
    if score >= 8:
        return "high", score
    if score >= 5:
        return "medium", score
    return "low", score


def build_docs_list(docs_dir: Path, out_dir: Path) -> tuple[Path, Path, Path]:
    pdf_files = sorted(docs_dir.glob("*.pdf"))
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[DocRow] = []

    for pdf in pdf_files:
        pages, non_empty = _extract_text_pages(pdf)
        joined = "\n".join(pages[:4])
        title = _pick_title(pages[0] if pages else "")
        doc_number, candidates = _pick_document_number(joined)
        date = _pick_date(joined)
        doc_type = _classify_type(joined)

        claimant = _extract_party(
            [
                r"Claimant(?:s)?\s*[:\-]\s*([^\n]{2,180})",
                r"Plaintiff(?:s)?\s*[:\-]\s*([^\n]{2,180})",
                r"Applicant(?:s)?\s*[:\-]\s*([^\n]{2,180})",
            ],
            joined,
        )
        defendant = _extract_party(
            [
                r"Defendant(?:s)?\s*[:\-]\s*([^\n]{2,180})",
                r"Respondent(?:s)?\s*[:\-]\s*([^\n]{2,180})",
            ],
            joined,
        )

        confidence, confidence_score = _confidence(doc_number, title, date, claimant, defendant, doc_type)
        total_pages = len(pages)
        coverage = round(non_empty / total_pages, 4) if total_pages else 0.0

        rows.append(
            DocRow(
                db_name=pdf.stem,
                doc_id=pdf.stem,
                file_name=pdf.name,
                title=title,
                document_number=doc_number,
                document_number_candidates="; ".join(candidates),
                date=date,
                type=doc_type,
                claimant=claimant,
                defendant=defendant,
                total_pages=total_pages,
                text_coverage_ratio=coverage,
                confidence=confidence,
                confidence_score=confidence_score,
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
    parser = argparse.ArgumentParser(description="Build docs_list metadata for challenge corpus.")
    parser.add_argument("--docs-dir", default="cache/warmup/docs_corpus", help="Directory containing source PDFs.")
    parser.add_argument("--out-dir", default="artifacts/docs_list", help="Output directory for docs_list artifacts.")
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    csv_path, json_path, low_path = build_docs_list(docs_dir, out_dir)

    print(f"docs_list csv: {csv_path}")
    print(f"docs_list json: {json_path}")
    print(f"low confidence: {low_path}")


if __name__ == "__main__":
    main()
