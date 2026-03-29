# Plan 02: Next Steps After v1/v2/v3 + Ground Truth

Date: 2026-03-19  
Status: Active  
Constraint: still **no submission to server**.

## 1) Current baseline status

- Runs completed: `v1`, `v2`, `v3` (25 questions each).
- Local answer correctness (matched set):
  - `v1`: 7/25
  - `v2`: 6/25
  - `v3`: 10/25
- `submission_wide.csv` already enriched with true columns from `submission_true_answers_32.csv`.

## 2) Immediate next step (highest value)

Run all active configs on **all 32 labeled questions** to get fair `X/32`:
- `v1` (`small + hybrid`)
- `v2` (`small + vector_only`)
- `v3` (`large + hybrid`)

Output:
- update `results/total_score.csv` with `fully_correct_count_32`.

## 3) Add retrieval-grounding evaluation (critical for challenge score)

Implement local evaluator for retrieval quality using `true.retrieval`:
- `Grounding_F2.5` (same beta as challenge)
- `Precision_pages`
- `Recall_pages`
- `Hit@k_pages` (k from predicted set size)

Output:
- `results/retrieval_metrics_by_run.csv`
- append key retrieval metrics to `results/experiments_summary.csv`.

## 4) Module-by-module ablation matrix

Keep one variable changed per run:

1. **Embeddings module**
- `openai/text-embedding-3-small` vs `openai/text-embedding-3-large`
- fixed retrieval `hybrid`, fixed LLM `openai/gpt-4o-mini`

2. **Retrieval module**
- `vector_only` vs `hybrid`
- fixed embedding best from previous step, fixed LLM

3. **LLM module**
- `openai/gpt-4o-mini` vs `deepseek-chat` (when DeepSeek keys are available)
- fixed embedding/retrieval best from previous steps

4. **Parsing module**
- `PyMuPDF` vs `Docling` (same downstream settings)
- compare with same 32 questions.

For each module comparison report 3 metrics:
- deterministic correctness (`fully_correct_count`)
- grounding (`F2.5`)
- latency (`avg_ttft_ms`)

## 5) Cost tracking improvement

Current estimate covers answer generation only.  
Need add separate embedding/indexing cost accounting:
- `embedding_tokens_total` (or characters/pages proxy if provider tokens unavailable)
- `embedding_cost_estimate_usd`
- `total_cost_estimate_usd = answer_cost + embedding_cost`

Output:
- `results/cost_breakdown.csv`.

## 6) Decision rule for best combo

Select winner config by priority:
1. highest `Grounding_F2.5`
2. then highest deterministic correctness
3. then lower TTFT
4. then lower total cost

Output:
- `results/recommendation.md` with chosen config and justification.

## 7) Practical execution order

1. Re-run `v1/v2/v3` on 32 questions.
2. Compute grounding metrics from true retrieval.
3. Choose better of `small/large` embedding.
4. Run retrieval ablation.
5. Run LLM ablation (add DeepSeek once env is ready).
6. Run parser ablation with Docling.
7. Produce final recommendation file.

