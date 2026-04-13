# Conversational RAG Pipeline Report

## Objective

Build an end-to-end conversational Retrieval-Augmented Generation (RAG) system that supports Arabic and English documents, streams responses, and provides source grounding.

## Implemented Solution

The system is implemented with a Streamlit interface and a modular Python backend:

- **UI Layer**: `app/ui/streamlit_app.py`
  - Upload and process up to 3 files (`PDF`, `DOCX`)
  - Chat-style interaction
  - Per-turn answer display with source rendering
  - Conversation history in the same page

- **Document Processing**:
  - `DocumentLoader` extracts page-level text from PDF and DOCX.
  - `TextChunker` splits text into manageable chunks with overlap.

- **Embedding Layer**:
  - OpenAI embeddings (`text-embedding-3-small`)
  - Token-aware batching to avoid max-token-per-request failures

- **Vector Search Layer**:
  - FAISS in-memory vector index
  - Nearest-neighbor retrieval by semantic similarity
  - Similarity score filtering (`MIN_RELEVANCE_SCORE`) before answer generation

- **Generation Layer**:
  - OpenAI chat model (`gpt-4o-mini`) with streaming output
  - Strict grounding rules in the prompt
  - Explicit fallback when context is insufficient

## Key Design Decisions

1. **FAISS as vector store**
   - Chosen for speed and simplicity in local assignment scope.
2. **Page-aware metadata**
   - Each chunk preserves source filename and page to support citations.
3. **Grounding controls**
   - Relevance threshold and fallback response reduce hallucinations.
4. **Safe source display**
   - Sources are hidden when response is fallback/no-answer.

## Observed Results

- Multi-document processing works for Arabic/English inputs.
- System streams responses and shows references for grounded answers.
- Out-of-scope questions return a no-information response.
- Token limit issues during embedding were mitigated by token-aware batching.

## Limitations

- FAISS index is in-memory only (no persistent vector DB on disk).
- Relevance threshold is static and may require tuning by dataset.
- OCR-heavy scanned PDFs may still need dedicated OCR preprocessing.

## Deployment

Dockerized deployment is provided via:

- `Dockerfile`
- `.dockerignore`
- `INSTRUCTIONS.md` run steps

The app runs on port `8501` and uses `.env` for API key injection.
