# Advanced RAG v22 (Docling + OpenAI Embeddings)

This phase aims to improve parsing quality using Docling and retrieval performance using OpenAI's `text-embedding-3-large` model. It also addresses formatting issues discovered in v21.

## Proposed Changes

### [RAG Pipeline]

#### [NEW] [advanced_hybrid_rag_v22.py](starter_kit/examples/llamaindex/advanced_hybrid_rag_v22.py)
- Integrate `DoclingReader` for more robust PDF-to-Markdown/Text conversion.
- Switch `Settings.embed_model` to `OpenAIEmbedding(model="text-embedding-3-large")`.
- Implement rigorous ISO 8601 (`YYYY-MM-DD`) formatting for date fields in [_parse_answer_by_type](starter_kit/examples/llamaindex/grounded_rag_llamaindex.py#61-90).
- Maintain Hybrid Search (Vector + BM25) and LLM Reranking from v20.
- Re-evaluate the query routing logic from v21 (optionally integrate or stick to advanced hybrid if simpler is better for this evaluation). *Decision: We will stick to the robust Hybrid approach from v20 but with better parsing and embeddings.*

## Verification Plan

### Automated Tests
- Run `advanced_hybrid_rag_v22.py` on the 100-question dataset.
- Generate `submission_comparison_v11.csv` to compare results.
- Verify date formats in the generated [submission.json](starter_kit/submission.json).

## Verification Plan

### Manual Verification
- Review [document_metadata.csv](starter_kit/document_metadata.csv) for accuracy.
- Verify that [multi_step_rag_v21.py](starter_kit/examples/llamaindex/multi_step_rag_v21.py) correctly filters documents for specific case-related questions.
