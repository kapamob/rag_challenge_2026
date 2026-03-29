from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_by_doc_id(path: Path) -> dict[str, dict[str, str]]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    return {r["doc_id"]: r for r in rows}


def compare(v1_path: Path, v2_path: Path, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    v1 = _read_by_doc_id(v1_path)
    v2 = _read_by_doc_id(v2_path)
    common_ids = sorted(set(v1) & set(v2))

    diffs = []
    fields = ["document_number", "date", "type", "title", "claimant", "defendant"]
    for doc_id in common_ids:
        r1 = v1[doc_id]
        r2 = v2[doc_id]
        row = {
            "doc_id": doc_id,
            "file_name": r1.get("file_name", r2.get("file_name", "")),
        }
        any_diff = False
        for f in fields:
            a = (r1.get(f, "") or "").strip()
            b = (r2.get(f, "") or "").strip()
            row[f"v1.{f}"] = a
            row[f"v2.{f}"] = b
            row[f"diff.{f}"] = int(a != b)
            any_diff = any_diff or (a != b)
        row["has_any_diff"] = int(any_diff)
        diffs.append(row)

    diff_csv = out_dir / "docs_list_diff.csv"
    with diff_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(diffs[0].keys()) if diffs else ["doc_id"])
        writer.writeheader()
        if diffs:
            writer.writerows(diffs)

    # summary
    v1_fill = sum(1 for d in v1.values() if (d.get("document_number", "") or "").strip())
    v2_fill = sum(1 for d in v2.values() if (d.get("document_number", "") or "").strip())
    any_diff_count = sum(1 for d in diffs if d["has_any_diff"] == 1)
    per_field_diff = {f: sum(1 for d in diffs if d[f"diff.{f}"] == 1) for f in fields}

    summary_md = out_dir / "docs_list_comparison.md"
    with summary_md.open("w", encoding="utf-8") as f:
        f.write("# Docs List Comparison\n\n")
        f.write(f"- docs in v1: {len(v1)}\n")
        f.write(f"- docs in v2: {len(v2)}\n")
        f.write(f"- common docs: {len(common_ids)}\n")
        f.write(f"- document_number filled v1: {v1_fill}\n")
        f.write(f"- document_number filled v2: {v2_fill}\n")
        f.write(f"- docs with any field difference: {any_diff_count}\n\n")
        f.write("## Per-field diffs\n")
        for k, v in per_field_diff.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n")
        f.write(f"Diff table: `{diff_csv}`\n")

    return diff_csv, summary_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare docs_list variants.")
    parser.add_argument("--v1", default="artifacts/docs_list/v1_regex/docs_list.csv")
    parser.add_argument("--v2", default="artifacts/docs_list/v2_llm/docs_list.csv")
    parser.add_argument("--out-dir", default="artifacts/docs_list/compare")
    args = parser.parse_args()

    diff_csv, summary_md = compare(Path(args.v1).resolve(), Path(args.v2).resolve(), Path(args.out_dir).resolve())
    print(f"diff csv: {diff_csv}")
    print(f"summary md: {summary_md}")


if __name__ == "__main__":
    main()

