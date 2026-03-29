# v6

- docs_list module: `v1_regex`
- direct route (step 3): `llm` over docs_list metadata
- retrieval strategy: `llm_rerank`
  - per-doc recall: top_k_per_doc=8
  - rerank pool: up to 24 candidate pages
  - final evidence pages: 4
- indexing: separate vector index per document (`artifacts/document_indices/v1_regex_small`)
- llm: `openai/gpt-4o-mini`
- embeddings: `openai/text-embedding-3-small`
- question_limit: 25
- no submission sent to server

