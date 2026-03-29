# Experiments Report

Документ фиксирует все эксперименты в проекте `rag_challenge`: локальные оффлайн-сравнения и online-результаты сабмитов на сервер конкурса.

## Scope

- Локальные серии: `v1` ... `v14` (контрольный набор 25 вопросов, сверка через `results/submission_true_answers_32.csv`).
- Online-серии: `v8_100`, `v10_100`, `v14_100` (100 вопросов, официальный скоринг на сервере).
- Базовый стек: LlamaIndex + per-document indices + docs_list routing.

## Local Benchmark (25 Questions)

| Run | Концепция | Correct | Rate | Answer Errors | DocID Errors | Excess | Insufficient | Equal | Avg Total ms | Avg TTFT ms | P95 ms | Input Tokens | Output Tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 | Базовый pipeline (run_experiments): PyMuPDF parse, global hybrid retrieval (BM25+vector), embeddings small, LLM gpt-4o-mini. | 7 | 0.28 | 18 | 21 | 25 | 0 | 0 | 1215.5 | 1044.1 | 1597 | 47553 | 381 |
| v2 | Как v1, но vector-only retrieval (без BM25). | 6 | 0.24 | 19 | 19 | 25 | 0 | 0 | 1203.8 | 999.5 | 1745 | 52923 | 379 |
| v3 | Как v1, но embeddings large. | 10 | 0.4 | 15 | 21 | 25 | 0 | 0 | 1270.2 | 1092.8 | 1864 | 46052 | 329 |
| v4 | Переход на routed regex docs_list + per-document indices; direct rules + constrained retrieval по найденным doc ids. | 8 | 0.32 | 17 | 16 | 15 | 10 | 0 | 699.9 | 643.3 | 1285 | 18455 | 206 |
| v5 | Как v4, но direct step через LLM (docs_list -> ответ). | 9 | 0.36 | 16 | 17 | 9 | 15 | 1 | 1040.5 | 951.5 | 1801 | 13178 | 195 |
| v6 | Как v5, retrieval strategy = llm_rerank (вариант 2, wider recall + rerank). | 11 | 0.44 | 14 | 17 | 3 | 15 | 7 | 780.2 | 699.2 | 1449 | 8619 | 211 |
| v7 | Как v5, retrieval strategy = dynamic (вариант 1). | 10 | 0.4 | 15 | 17 | 9 | 15 | 1 | 823.2 | 743.3 | 1841 | 20658 | 213 |
| v8 | Новый llm_router (answer|docs) по docs_list; retrieval только по doc-кандидатам от роутера. | 15 | 0.6 | 10 | 10 | 10 | 3 | 12 | 1078.3 | 772.0 | 2242 | 28961 | 338 |
| v9 | Как v8 + правило: если direct/router вернул null, fallback retrieval по всем документам. | 16 | 0.64 | 9 | 10 | 11 | 3 | 11 | 2712.9 | 2238.6 | 3692 | 31417 | 412 |
| v10 | Как v9, но embeddings large (отдельные индексы large). | 17 | 0.68 | 8 | 7 | 12 | 2 | 11 | 8630.2 | 4283.1 | 13150 | 172597 | 2911 |
| v11 | Fast-режим: small, baseline retrieval, top_k=2, max_pages=1, no global fallback. | 11 | 0.44 | 14 | 14 | 1 | 13 | 11 | 6244.8 | 2838.4 | 9193 | 57724 | 2580 |
| v12 | Ultrafast: direct rules + retrieval disabled. | 1 | 0.04 | 24 | 23 | 0 | 23 | 2 | 0.0 | 0.0 | 0 | 0 | 0 |
| v13 | Компромисс speed: direct rules, top_k=1/max_pages=1, LLM только для free_text. | 3 | 0.12 | 22 | 22 | 2 | 21 | 2 | 204.7 | 194.4 | 1558 | 961 | 14 |
| v14 | На базе v10: embeddings small, chunking 300/50, LLM deepseek-chat (через OPENAI_API_BASE), retrieval llm_rerank. | 18 | 0.72 | 7 | 11 | 9 | 5 | 11 | 7161.4 | 3938.5 | 9817 | 83792 | 1647 |

## Key Observations (Local)

- Лучшее локальное качество: `v14` (`18/25`, rate `0.72`).
- Следом: `v10` (`17/25`) и `v9` (`16/25`).
- Самые быстрые: `v13` и `v12`, но с сильной деградацией качества.
- Основной trade-off: рост качества достигается через дополнительные LLM-шаги (router/rerank/final), что увеличивает latency и токены.

## Online Submissions (100 Questions, Server Metrics)

| Run | UUID | Total Score | Deterministic | Assistant | Grounding | Telemetry | TTFT ms |
|---|---|---:|---:|---:|---:|---:|---:|
| v8_100 | `2bc32d0c-83c0-4e8f-a6c2-dc8fb0afef3f` | 0.307005 | 0.714286 | 0.426667 | 0.555829 | 0.988 | 6253 |
| v10_100 | `07bf1935-4f64-415e-82ce-f7d03fb1c6a8` | 0.362997 | 0.771429 | 0.520000 | 0.575995 | 0.987 | 4889 |
| v14_100 | `8048d084-3da3-40b8-b831-9827053582f2` | 0.420288 | 0.742857 | 0.426667 | 0.689665 | 0.986 | 3890 |

## Online Conclusion

- Лучший online результат на текущий момент: `v14_100` (`total_score = 0.420288`).
- `v14_100` улучшил `total_score` относительно `v10_100` и `v8_100` за счет более сильного `grounding` и более низкого `ttft_ms`.

## Artifact Paths

- Local summaries: `results/total_score.csv`, `results/error_analysis_summary.csv`, `results/submission_wide.csv`
- Per-run artifacts: `experiments/v*/submission.json`, `experiments/v*/submission.csv`, `experiments/v*/code_archive.zip`
- v14 chunked indices: `artifacts/document_indices/v14_small_c300_o50`
