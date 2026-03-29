# RAG Challenge — Codex Plus

A systematic RAG pipeline built for the [Agentic RAG Challenge](https://platform.agentic-challenge.ai/). This solution uses **Codex** (Google AI assistant) to design and iterate through 16 experiment versions, exploring different retrieval strategies, embedding models, and LLM configurations.

## Approach

- **PDF Parsing**: PyMuPDF (fitz) — page-level text extraction with metadata
- **Retrieval**: Hybrid search (BM25 lexical + vector similarity) fused via Reciprocal Rank Fusion (RRF)
- **Embeddings**: OpenAI `text-embedding-3-small` / `text-embedding-3-large` (via OpenRouter)
- **LLM**: `gpt-4o-mini` (via OpenRouter) with streaming for TTFT measurement
- **Answer normalization**: Strict type-aware parsing (number, boolean, date → ISO 8601, names, free_text)

## Project Structure

```
rag_challenge_codex_plus/
├── README.md                   # This file
├── .gitignore
├── .env.example                # → see starter_kit/.env.example
│
├── run_experiments.py          # Main pipeline: parse → index → retrieve → answer
│
├── plan_01.md                  # Development plan v1 (architecture & experiment design)
├── plan_02.md                  # Development plan v2
├── plan_03.md                  # Development plan v3
│
├── scripts/                    # Utility scripts
│   ├── build_docs_list.py          # Extract document-level metadata from PDFs
│   ├── build_docs_list_llm.py      # LLM-assisted metadata extraction
│   ├── build_document_indices.py   # Build vector indices
│   ├── compare_docs_list_variants.py # Compare metadata extraction approaches
│   └── run_rag_routed_regex.py     # Regex-routed RAG variant
│
├── experiments/                # Results from 16 experiment versions
│   ├── _prep/                      # Shared parsing artifacts (docs_list, samples)
│   ├── v1/ … v16_merged/          # Per-version: submission.json, submission.csv,
│   │                                #   code_archive.zip, metrics.json, README.md
│   └── ...
│
├── results/                    # Cross-experiment analysis
│   ├── experiments_summary.csv     # Metrics table across all versions
│   ├── submission_wide.csv         # Answers comparison across versions
│   ├── total_score.csv             # Score summary
│   ├── error_analysis_v*.csv       # Per-version error breakdowns
│   └── error_questions_by_run.md   # Questions that failed across runs
│
└── starter_kit/                # Official challenge starter kit
    ├── .env.example                # Environment variables template
    ├── README.md                   # Challenge documentation
    ├── API.md                      # API reference
    ├── EVALUATION.md               # Scoring methodology
    ├── openapi.yaml                # API schema
    ├── requirements.txt            # Python dependencies
    ├── submission.json             # Example submission format
    ├── arlc/                       # Challenge client library
    │   ├── client.py                   # API client (download questions/docs, submit)
    │   ├── config.py                   # Configuration management
    │   ├── submission.py               # Submission builder & data models
    │   └── telemetry.py                # Timing & usage metrics
    └── examples/                   # Reference implementations
        ├── submit.py                   # Submission example
        ├── telemetry_example.py        # Telemetry usage example
        ├── langchain/                  # LangChain baseline
        └── llamaindex/                 # LlamaIndex baseline
```

## Setup

```bash
# Clone the repo
git clone https://github.com/kapamob/rag_challenge_codex_plus.git
cd rag_challenge_codex_plus

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r starter_kit/requirements.txt
pip install pymupdf llama-index llama-index-llms-openai llama-index-embeddings-openai python-dotenv

# Configure environment
cp starter_kit/.env.example .env
# Edit .env and fill in your API keys
```

## Running

```bash
python run_experiments.py
```

This will:
1. Download questions and documents from the challenge API
2. Parse all PDFs with PyMuPDF
3. Build vector indices
4. Run configured experiments (v1, v2, v3 by default)
5. Save results to `experiments/` and `results/`

## Key Design Decisions

- **No submission sent to server** — all experiments run locally for iterative improvement
- **Budget-aware**: experiments start with small subsets (25 questions) before scaling to 100
- **Modular architecture**: each component (parsing, chunking, retrieval, answering) can be swapped independently
- **Comprehensive tracking**: every run produces metrics, CSVs, and code archives for reproducibility
