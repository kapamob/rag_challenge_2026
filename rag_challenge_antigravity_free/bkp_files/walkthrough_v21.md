# Walkthrough: Phase 11 (Multi-Step Routing RAG)

I've implemented a new multi-step RAG pipeline aimed at improving retrieval precision by pre-analyzing the corpus and routing queries to specific document-level indices.

## Key Accomplishments

### 1. Corpus Analysis & Metadata Extraction
- Created [corpus_analyzer.py](starter_kit/scripts/corpus_analyzer.py) which used LLM to extract metadata for all 30 documents.
- Generated [document_metadata.csv](starter_kit/document_metadata.csv) containing:
    - **document_number** (Case IDs and Law Numbers)
    - **participants** (Claimant / Defendant)
    - **title** and **type**

### 2. Document-Specific Indexing
- Instead of a single cross-document index, I created **30 isolated indices** stored in `document_indices/`. 
- This ensures that retrieval is confined to the specific legal context relevant to the question.

### 3. Routing Logic
- Implemented [multi_step_rag_v21.py](starter_kit/examples/llamaindex/multi_step_rag_v21.py) with a routing layer:
    - **ID Matching**: Automatically detects Case IDs (e.g., SCT 295/2025) and Law numbers.
    - **Participant Matching**: Detects Claimant/Defendant names from the query.
    - **Lazy Loading**: Only the indices for matching documents are loaded and searched.

## Results & Validation

- **Completion**: All 100 questions processed.
- **Precision**: Routing successfully isolated relevant documents for case-specific questions, potentially reducing "noise" and grounding errors.
- **Identified Gaps**: Questions 62 and 96 failed to route correctly (no document mentioned in query). A global fallback or fuzzy metadata search is recommended for these cases.

### Comparison
You can review the results in the new comparison file: [submission_comparison_v10.csv](starter_kit/submission_comparison_v10.csv).

## Next Steps
- [ ] Investigate routing for general questions (Q62, Q96).
- [ ] Submit `submission_v21_multi_step.json` to the leaderboard to evaluate Grounding score improvements.
