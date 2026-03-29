# Plan 01: RAG Challenge (No Submission Yet)

Date: 2026-03-19  
Status: Active  
Constraint: do not send submission to server.

## 1) Locked baseline choices

- Parsing baseline: `PyMuPDF` (+ OCR fallback only for pages with near-empty text).
- Retrieval baseline: `Hybrid` (BM25 + vector).
- LLM baseline: `gpt-4o-mini`.
- `free_text` hard limit: `<= 280 chars`.
- Budget for first cycle: `$5`.

## 2) What `$5` budget affects

- Number of answered questions in paid runs:
  - with reranking (extra LLM calls) we may need to run only a subset first (e.g., 20-40 questions),
  - without reranking we can run more questions.
- Model choices:
  - `gpt-4o-mini` remains default,
  - deep/expensive rerank or larger embedding models must be sampled, not full-corpus first.
- Experiment design:
  - first pass = ablation on small fixed subset,
  - full run only for top candidates.

## 3) Pipeline architecture to implement

1. `download_once` layer:
- downloads questions/docs only if absent,
- stores a local manifest hash for reproducibility.

2. `parse` module:
- PyMuPDF extractor,
- optional OCR fallback.

3. `docs_list` builder:
- file-level metadata table:
  - `doc_id`, `file_name`, `title`, `doc_number`, `doc_date`, `doc_type`,
  - `claimant`, `defendant`,
  - parser diagnostics (text coverage, OCR used, pages).

4. `chunk_ingest` module:
- chunking strategies (fixed 512/64 and semantic),
- metadata per chunk: `chunk_id`, `doc_id`, `page_number`, parent/page pointers.

5. `index` module:
- SimpleVectorStore / FAISS variants,
- optional BM25 index.

6. `retrieve` module:
- hybrid retrieval,
- optional LLM reranking,
- evidence pruning to used pages only.

7. `answer` module:
- route by `answer_type`,
- strict typed output normalization,
- `is_unanswerable` gate:
  - deterministic -> `null`,
  - free_text -> no-info sentence + empty retrieval.

8. `telemetry` module:
- submission-compatible metrics (`ttft_ms`, `tpot_ms`, `total_time_ms`),
- internal extra metric `ttft_e2e_ms` (all LLM calls) for local analysis.

9. `runner` module:
- constructor-style config for each run (`v1`, `v2`, ...),
- artifacts per version folder.

## 4) Experiments and per-module metrics

## 4.1 Parsing module comparison (PyMuPDF vs Docling)
- Inputs fixed: same question subset, same ingestion/retrieval/LLM.
- Metrics:
  - `parse_text_coverage` (non-empty text pages / total pages),
  - `ocr_page_ratio`,
  - downstream `Grounding_F2.5_local` and `EM_det_local`.

## 4.2 Chunking module comparison (fixed vs semantic)
- Metrics:
  - retrieval `Recall@k_pages`,
  - retrieval `Precision@k_pages`,
  - downstream `Grounding_F2.5_local`.

## 4.3 Vector DB comparison (SimpleVectorStore vs FAISS)
- Metrics:
  - query latency p50/p95,
  - retrieval quality (`Recall@k`, `MRR`),
  - downstream score impact (`Grounding_F2.5_local`).

## 4.4 Retrieval strategy comparison (Hybrid vs Hybrid+LLM-reranking)
- Metrics:
  - retrieval quality (`Recall@k`, `nDCG@k`, `Grounding_F2.5_local`),
  - extra cost per 100 questions,
  - TTFT impact.

## 4.5 LLM comparison (gpt-4o-mini vs deepseek-chat)
- Metrics:
  - deterministic `EM_det_local`,
  - free-text judge proxy (local LLM-judge rubric),
  - hallucination proxy rate,
  - cost and TTFT.

## 4.6 Embedding comparison
- Metrics:
  - `Recall@k`, `MRR`, `nDCG`,
  - index size and ingestion time,
  - cost per 100 questions.

## 5) Files and outputs per run

For each `experiments/vN/`:
- `README.md` (exact config and modules),
- `submission.json` (local only),
- `code_archive.zip`,
- `docs_list.csv`,
- `submission.csv` (`question_id`, `answer_type`, `question`, `vN.answer`, `vN.retrieval`),
- debug files (`parsed_sample_*.md/csv`).

Global:
- `results/submission_wide.csv` (columns across versions),
- `results/total_score.csv` (after you provide ground truth).

## 6) Execution order (budget-aware)

1. Implement modular framework + caching.
2. Build `v1` baseline and run on small fixed subset (budget-safe).
3. Run targeted ablations one module at a time:
- parser,
- retrieval,
- rerank,
- LLM.
4. Collect per-module metrics table and choose best combination.
5. Prepare best candidate package for future submission (without sending now).

