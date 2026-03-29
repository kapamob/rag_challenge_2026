# RAG Challenge 2026

I tried to describe the entire process and save it in md files: chat*.md and others. Start with them.

Two independent solutions for the [Agentic RAG Challenge](https://agentic-challenge.ai/) — a competition to build the best Retrieval-Augmented Generation pipeline for answering questions over legal PDF documents.

Both solutions were developed with AI coding assistants and explore different approaches to PDF parsing, hybrid retrieval, and answer generation.

## Solutions

| Solution | AI Assistant | Approach | Final Version |
|----------|-------------|----------|---------------|
| [rag_challenge_codex_plus](./rag_challenge_codex_plus/) | Codex | PyMuPDF + Hybrid RRF + systematic experiments (v1–v16) | v16 |
| [rag_challenge_antigravity_free](./rag_challenge_antigravity_free/) | Antigravity (free) | Docling + Hybrid + LLM Reranking (v0–v22) | v22 |

## Shared Tech Stack

- **Framework**: LlamaIndex
- **Embeddings**: OpenAI text-embedding-3-large (via OpenRouter)
- **LLM**: GPT-4o-mini (via OpenRouter)
- **Retrieval**: Hybrid (BM25 + Vector search)

See each solution's README for detailed setup and usage instructions.


AKT team.
