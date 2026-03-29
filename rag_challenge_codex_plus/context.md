# Context (Working Memory)

Date initialized: 2026-03-19

## Goal

Build a modular RAG experimentation framework for challenge project `rag_challenge` using `starter_kit` and LlamaIndex, compare multiple methods, pick best combination, but **do not submit** yet.

## Hard constraints from user

- Use LlamaIndex.
- Do not send to submission now.
- Cache/reuse heavy artifacts (downloaded docs/questions, indexes, embeddings).
- Keep each experiment in separate folder `v1`, `v2`, `v3`, ...
- Maintain per-run artifacts: `README.md`, `code_archive.zip`, `submission.json`, `docs_list.csv`, extra debug files.
- Maintain summary tables for cross-run comparison.
- `free_text` must follow challenge constraints (`<= 280 chars`).
- Keys:
  - DeepSeek key separate.
  - Others via OpenRouter.
- Warn user if API usage may be expensive.
- Budget for first cycle: `$5`.

## Baseline confirmed by user

1. Baseline stack: `PyMuPDF + Hybrid + gpt-4o-mini` -> YES.
2. Budget: `$5`.
3. Strict free_text length -> YES.

## Competition-specific critical notes (from starter kit)

- Grounding (`retrieved_chunk_pages`) is a multiplier in final score.
- Telemetry required and must be valid.
- `doc_id` must match corpus file id.
- `page_numbers` must be physical PDF pages, 1-based.
- For unanswerable deterministic types: use `null`.
- For unanswerable free_text: answer with no-info sentence and empty retrieval refs.

## Plan file policy

- Current active plan file: `plan_01.md`.
- On any substantial plan change after experiment results, create next file:
  - `plan_02.md`, `plan_03.md`, etc.

## Chat log policy

- Keep chat history in `chat.md`.
- Update `chat.md` after each user message.

## Next execution step

Implement project skeleton for experiment runner + caching + baseline `v1` pipeline, then run first budget-aware sample evaluation.

## Executed experiments (2026-03-19)

- Implemented runner: `run_experiments.py`
- Download/cache path: `cache/warmup`
- Created runs:
  - `experiments/v1` = `PyMuPDF + hybrid + gpt-4o-mini`
  - `experiments/v2` = `PyMuPDF + vector_only + gpt-4o-mini`
- No submission sent.

Results on 25-question sample:
- `v1`: avg_ttft=1105.44 ms, avg_total=1356.8 ms, null_rate=0.32, est_cost=0.0073 (LLM answer stage only)
- `v2`: avg_ttft=1124.56 ms, avg_total=1365.68 ms, null_rate=0.36, est_cost=0.0082 (LLM answer stage only)

Current reading:
- For this sample, hybrid outperformed vector-only on latency, null-rate, and estimated generation cost.

## Update (OpenRouter model IDs + v3)

User provided canonical IDs:
- LLM: `openai/gpt-4o-mini`
- embeddings: `openai/text-embedding-3-small`, `openai/text-embedding-3-large`

Code updated:
- OpenRouter-style model IDs are normalized for LlamaIndex compatibility.
- Index now built per embedding model (fair embedding ablation).

New run:
- `v3` = `PyMuPDF + hybrid + openai/gpt-4o-mini + openai/text-embedding-3-large`

Current sample metrics (25 questions):
- `v1` (small emb, hybrid): ttft 1044.12 ms, null_rate 0.32
- `v2` (small emb, vector_only): ttft 999.48 ms, null_rate 0.32
- `v3` (large emb, hybrid): ttft 1092.8 ms, null_rate 0.28

## Docs List Build (2026-03-19)

- Added script: `scripts/build_docs_list.py`
- Generated artifacts:
  - `artifacts/docs_list/docs_list.csv`
  - `artifacts/docs_list/docs_list.json`
  - `artifacts/docs_list/docs_list_low_confidence.csv`

Current stats:
- documents processed: 30
- `document_number` filled: 24/30
- confidence: high=15, medium=15, low=0
- low-confidence review file currently contains 6 docs (mostly missing `document_number`).
