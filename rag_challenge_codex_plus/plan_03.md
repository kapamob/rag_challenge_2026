# Plan 03: Docs-List-First + Per-Document Vector DB

Date: 2026-03-19  
Status: Proposed (awaiting approval)

## 1) Current architecture (as-is)

Based on `run_experiments.py`:

1. Data loading
- questions/documents are downloaded once and cached (`cache/warmup`).

2. Parsing
- all PDFs parsed with PyMuPDF page-by-page.
- lightweight metadata extraction only:
  - `title` from PDF metadata,
  - `doc_date` from PDF metadata,
  - claimant/defendant by weak regex over sample text.
- `doc_number` and `doc_type` are mostly empty -> `docs_list` quality is low.

3. Indexing
- a single global set of page documents is created.
- index built with selected embedding model (currently per embedding, but still global corpus index).

4. Retrieval
- vector retrieval over global index (+ optional lexical fusion).
- no strict document-level routing by case number/doc number.

5. Answering
- LLM prompt uses top retrieved pages.
- telemetry uses all selected pages; this causes many extra page refs.

## 2) Why this conflicts with your target design

You requested:
- first try answer from `docs_list`,
- if not found, extract document numbers from question,
- retrieve only from vector DBs of those documents,
- each document must have its own dedicated vector DB.

Current system fails these points because:
- `docs_list` is incomplete/low-trust,
- retrieval is global (not constrained to selected document DBs),
- source pages are systematically over-selected (excess refs).

## 3) Target architecture (to-be)

Use `corpus_analyzer.py` ideas as baseline, but strengthen with validation.

### Layer A: Corpus Analyzer / Docs Registry

For each PDF:
- parse first N pages + optional full-text hints,
- extract structured metadata:
  - `doc_id`, `file_name`, `db_name`
  - `document_number` (e.g., `CFI 057/2025`, `Law No. 3 of 2018`)
  - `date`, `title`, `type`
  - `claimant`, `defendant`
- save `docs_list.csv` + `docs_list.json`.

Quality controls for docs_list:
- regex-based extraction for case numbers + law numbers,
- LLM extraction only as fallback/augmentation,
- normalization rules (`CFI 057/2025` == `CFI 57/2025`),
- confidence flags per field (`high/medium/low`),
- manual audit file for low-confidence rows.

### Layer B: Per-document vector databases

For each `doc_id`:
- build own index in `document_indices/<doc_id>/...`
- include chunk metadata: `doc_id`, `page_number`, `chunk_id`.
- cache by parser+embedding config hash.

### Layer C: Question Router (Docs-list-first)

Route per question:
1. `docs_list_lookup` route:
   - if answer can be resolved directly from metadata table (date/title/type/party/case match), return without RAG.
2. `doc_number_extraction` route:
   - extract mentioned doc numbers/case numbers from question,
   - map to candidate `doc_id` via normalized `document_number`.
3. constrained retrieval route:
   - if candidate docs found -> query only those document DBs,
   - else fallback to global shortlist (strict capped fallback).

### Layer D: Evidence selection for telemetry

- after retrieval, keep only minimal evidence pages that support final answer.
- enforce top evidence cap (e.g., max 1-2 docs, max 1-3 pages each by answer_type).
- reduce excess source refs.

## 4) Execution plan (next concrete steps)

1. Rebuild docs analyzer
- create `scripts/build_docs_list.py` using `corpus_analyzer.py` as reference.
- output:
  - `artifacts/docs_list/docs_list.csv`
  - `artifacts/docs_list/docs_list.json`
  - `artifacts/docs_list/docs_list_low_confidence.csv`.

2. Build per-doc indices
- create `scripts/build_document_indices.py`.
- persist each doc index separately.

3. Implement router
- create `scripts/run_rag_routed.py` with routes:
  - metadata route,
  - doc-number-constrained route,
  - fallback route.

4. Re-run baseline comparison
- new runs:
  - `v4` = routed + small embedding
  - `v5` = routed + large embedding
- evaluate vs truth on same questions.

5. Error-focused report
- compare `v3` vs `v4/v5` on:
  - answer correctness,
  - doc_id correctness,
  - excess/insufficient sources.

## 5) Success criteria for this refactor

- `docs_list` coverage: `document_number` filled for >90% documents.
- doc_id error count reduced materially vs `v3`.
- excess sources reduced from 25/25 to <=10/25 on the current sample.
- answer correctness improved over `v3` (10/25 baseline).

