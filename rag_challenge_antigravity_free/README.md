# RAG Challenge — Antigravity Free

A RAG pipeline built for the [Agentic RAG Challenge](https://platform.agentic-challenge.ai/) using **Antigravity** (Google AI assistant, free tier). The solution evolved through 22 versions, from naive RAG to an advanced hybrid pipeline with Docling parsing and LLM reranking.

## Approach

- **PDF Parsing**: Docling (v22) / PyMuPDF (earlier versions) — advanced document structure extraction
- **Retrieval**: Hybrid search (BM25 + Vector) via `QueryFusionRetriever` + LLM Reranking
- **Embeddings**: OpenAI `text-embedding-3-large` (via OpenRouter)
- **LLM**: `gpt-4o-mini` (via OpenRouter) with streaming for TTFT measurement
- **Answer normalization**: Type-aware parsing with strict ISO 8601 date formatting

## Project Structure

```
rag_challenge_antigravity_free/
├── README.md                       # This file
├── .gitignore
│
├── bkp_files/                      # Submission history & backups
│   ├── submission_*.json               # Submissions from different versions
│   ├── submission_comparison_v*.csv    # Answer comparisons across versions
│   ├── walkthrough_*.md                # Development notes
│   └── code_archive/                  # Archived code snapshots
│       └── code_archive_v*.zip
│
└── starter_kit/                    # Challenge starter kit + custom solution
    ├── .env.example                    # Environment variables template
    ├── README.md                       # Challenge documentation
    ├── API.md                          # API reference
    ├── EVALUATION.md                   # Scoring methodology
    ├── openapi.yaml                    # API schema
    ├── requirements.txt                # Python dependencies
    │
    ├── arlc/                           # Challenge client library
    │   ├── client.py                       # API client
    │   ├── config.py                       # Configuration
    │   ├── submission.py                   # Submission builder & data models
    │   └── telemetry.py                    # Timing & usage metrics
    │
    ├── examples/                       # RAG implementations (evolution history)
    │   ├── submit.py                       # Submission example
    │   ├── telemetry_example.py            # Telemetry example
    │   ├── langchain/                      # LangChain baseline
    │   │   └── naive_rag_langchain.py
    │   └── llamaindex/                     # LlamaIndex implementations
    │       ├── naive_rag_llamaindex.py         # v0: Simple RAG baseline
    │       ├── advanced_rag_llamaindex.py      # v1: PyMuPDF + better chunking
    │       ├── advanced_pdf_reader.py          # Custom PDF reader (PyMuPDF)
    │       ├── grounded_rag_llamaindex.py      # Grounded RAG with source tracking
    │       ├── hybrid_rag_comparison.py        # Hybrid retrieval comparison
    │       ├── advanced_hybrid_rag.py          # Hybrid (Vector + BM25) RAG
    │       ├── advanced_hybrid_rag_v20.py      # v20: Improved hybrid + date fixes
    │       ├── multi_step_rag_v21.py           # v21: Multi-step with document routing
    │       ├── advanced_hybrid_rag_v22.py      # v22: Docling + embedding-3-large (final)
    │       ├── questions.json                  # Cached questions
    │       └── requirements_llamaindex.txt     # LlamaIndex dependencies
    │
    ├── scripts/                        # Utility scripts
    │   ├── corpus_analyzer.py              # LLM-based document metadata extraction
    │   ├── compare_pdf_extractors.py       # Compare PyMuPDF vs Docling parsing
    │   ├── validate_submission.py          # Submission format validator
    │   ├── fix_submission_formats.py       # LLM-based format fixer (dates → ISO 8601)
    │   ├── prepare_submission_v21.py       # Submission preparation script
    │   └── final_submission_v21.py         # Final submission builder
    │
    ├── history/                        # Development history & plans
    │   ├── chat.md                         # Full AI chat log (development process)
    │   ├── framework_comparison.md         # LangChain vs LlamaIndex comparison
    │   ├── implementation_plan_v1.md       # Initial plan
    │   ├── implementation_plan_v2.md       # Revised plan
    │   ├── implementation_plan_v22.md      # Final plan (Docling + large embeddings)
    │   ├── implementation_plan_old.md      # Earlier plan iterations
    │   ├── task_old.md                     # Task checklists (earlier)
    │   ├── task_v22.md                     # Task checklist (v22)
    │   ├── walkthrough_phase5.md           # Phase 5 walkthrough (streaming fix)
    │   └── walkthrough_v21.md              # v21 walkthrough (multi-step routing)
    │
    └── leaderboard/                    # Competition leaderboard snapshots
        ├── leaderboard_202260313-0615.csv
        └── leaderboard_202260313-0615.pdf
```

## Setup

```bash
# Clone the repo
git clone https://github.com/kapamob/rag_challenge_antigravity_free.git
cd rag_challenge_antigravity_free/starter_kit

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r examples/llamaindex/requirements_llamaindex.txt

# Configure environment
cp .env.example .env
# Edit .env and fill in your API keys
```

## Running (v22 — final version)

```bash
cd starter_kit/examples/llamaindex
python advanced_hybrid_rag_v22.py
```

This will:
1. Download questions and documents from the challenge API
2. Parse PDFs with Docling (or load cached index)
3. Run hybrid retrieval (Vector + BM25) with LLM reranking
4. Generate answers with strict type formatting
5. Save submission JSON

## Version History

| Version | Key Changes |
|---------|-------------|
| v0 | Naive RAG (LlamaIndex baseline) |
| v1–v5 | PyMuPDF parsing, better chunking |
| v10–v15 | Hybrid retrieval (BM25 + Vector), answer normalization |
| v20 | Improved date formatting, strict ISO 8601 |
| v21 | Multi-step RAG with document routing + corpus metadata |
| **v22** | **Docling parsing + text-embedding-3-large + LLM reranking** |
