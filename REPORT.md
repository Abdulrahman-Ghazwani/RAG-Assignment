# Conversational RAG — Short Report

## Goal

End-to-end RAG: upload documents (PDF/DOCX), ask questions in **Arabic or English**, get **streaming** answers grounded in the files, with optional **source** lines.

## What we built

| Part | Implementation |
|------|----------------|
| UI | **Streamlit** — upload (max 3), process button, chat, sources when grounded |
| Load | **PyMuPDF** (PDF per page), **python-docx** (DOCX as one block) |
| Chunk | **RecursiveCharacterTextSplitter** — size/overlap from `config.py` |
| Embed | **OpenAI** `text-embedding-3-small` (batched, long texts truncated) |
| Retrieve | **ChromaDB** (HTTP client, L2 space), **top‑K** + **min similarity** |
| Generate | **OpenAI** `gpt-4o-mini`, **temperature 0**, **stream** |

## Design (minimal)

- **ChromaDB**: runs as a **Docker** service (`docker-compose.yml`); vectors persist in a named volume.
- **Score filter** before LLM: weak matches → fixed “no information” reply (no sources).
- **Prompt**: answer only from context; same language as question; explicit fallback sentence.

## Quick experiment (example)

1. Upload 1–2 short PDFs you know well.  
2. Ask a fact that appears verbatim → streamed answer + sources.  
3. Ask something absent from the files → fallback message, no sources.

## Limits

- Chroma data persists in Docker volume; re-processing documents replaces the collection.  
- DOCX has no real page map (cited as page 1).  
- Scanned PDFs need OCR outside this project.

## Run

See `INSTRUCTIONS.md` and `Dockerfile` for local and Docker steps.
