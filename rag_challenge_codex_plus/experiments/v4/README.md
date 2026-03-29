# v4

- docs_list module: `v1_regex`
- routing: docs-list-first + document-number extraction from question
- indexing: separate vector index per document (`artifacts/document_indices/v1_regex_small`)
- retrieval scope: only document indices referenced by extracted document numbers
- llm: `openai/gpt-4o-mini`
- embeddings: `openai/text-embedding-3-small`
- top_k_per_doc: 3
- max_evidence_pages: 3
- question_limit: 25
- no submission sent to server

