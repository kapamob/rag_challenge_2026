# v7

- docs_list module: `v1_regex`
- direct route (step 3): `llm` over docs_list metadata
- retrieval strategy: `dynamic` recall-bump
  - initial top_k_per_doc=3
  - if candidates are sparse, bump to top_k=8
  - final evidence pages: 5
- indexing: separate vector index per document (`artifacts/document_indices/v1_regex_small`)
- llm: `openai/gpt-4o-mini`
- embeddings: `openai/text-embedding-3-small`
- question_limit: 25
- no submission sent to server

