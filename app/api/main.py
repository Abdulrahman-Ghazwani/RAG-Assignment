"""
FastAPI application: HTTP API for the RAG (Retrieval-Augmented Generation) backend.

How to read this file (beginner-friendly):
- **FastAPI** maps URL paths to Python functions. Each `@app.get` / `@app.post` is one endpoint.
- **`X-Session-Id`** header: a UUID sent by the browser so each user has separate data (documents, index, chat).
- **`_sessions`**: in-memory dictionary holding per-session state. It resets if the API process restarts.
- **Chroma**: vector database (separate Docker container). Embeddings are stored there; session *metadata* is in `_sessions`.

Typical flow:
1. Client uploads files → `POST /api/process` → text is chunked, embedded, stored in Chroma.
2. Client asks a question → `POST /api/chat` → retrieve similar chunks, stream answer (SSE).

See `INSTRUCTIONS.md` in the repo root for how to run the whole stack with Docker.
"""

import hashlib
import json
import os
import re
import tempfile
import uuid
from collections import defaultdict
from threading import Lock

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import NO_ANSWER_MESSAGE, load_openai_api_key
from app.services.chunker import TextChunker
from app.services.document_loader import DocumentLoader
from app.services.rag_pipeline import RAGPipeline

# ---------------------------------------------------------------------------
# App instance & CORS (Cross-Origin Resource Sharing)
# Browsers block random sites from calling your API unless allowed. For local dev,
# we allow the Angular dev server / UI origin. In Docker, nginx often proxies /api
# so same-origin rules apply; CORS still helps for direct API calls.
# ---------------------------------------------------------------------------
app = FastAPI(title="RAG Assignment API")

_origins = os.getenv("CORS_ORIGINS", "http://localhost:4200").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Limits: max files per upload request, and max distinct indexed docs per session.
MAX_FILES_PER_REQUEST = 3
MAX_DISTINCT_DOCUMENTS_PER_SESSION = 3

# All per-session state lives here (RAM). Key = session UUID string.
_sessions: dict[str, dict] = {}
_lock = Lock()  # Simple lock so concurrent requests do not corrupt `_sessions`.


# ---------------------------------------------------------------------------
# Small helpers (validation, naming, session dict)
# ---------------------------------------------------------------------------
def _normalize_session_id(raw: str | None) -> str:
    if not raw or not raw.strip():
        raise HTTPException(status_code=400, detail="Missing X-Session-Id header.")
    s = raw.strip()
    try:
        uuid.UUID(s)
    except ValueError:
        raise HTTPException(status_code=400, detail="X-Session-Id must be a valid UUID.")
    return s


def _chroma_collection_name(session_id: str) -> str:
    """Alphanumeric-only name for Chroma (avoids collisions between users)."""
    cleaned = re.sub(r"[^0-9a-fA-F]", "", session_id)
    return f"rag_{cleaned}"[:512]


# Pydantic models = JSON body shape + automatic validation for POST requests.
class ChatBody(BaseModel):
    question: str = Field(..., min_length=1, max_length=8000)


class RemoveCorpusBody(BaseModel):
    sha256: str | None = None
    filename: str | None = Field(None, max_length=1024)


def _session_get(session_id: str) -> dict:
    with _lock:
        if session_id not in _sessions:
            _sessions[session_id] = {
                "indexed": False,
                "pipeline": None,
                "processed_hashes": set(),
                "indexed_documents": [],
                "chat_history": [],
            }
        return _sessions[session_id]


def _format_sources(retrieved: list, full_answer: str, has_grounding: bool) -> list[str]:
    """Build short 'filename (page …)' lines for the UI when the answer used retrieved chunks."""
    norm_a = " ".join(full_answer.split()).strip().lower()
    norm_n = " ".join(NO_ANSWER_MESSAGE.split()).strip().lower()
    show_src = has_grounding and retrieved and norm_a != norm_n
    if not show_src:
        return []
    by_file: dict[str, set] = defaultdict(set)
    for c in retrieved:
        by_file[c["source"]].add(c.get("page") or "?")
    lines = []
    for src in sorted(by_file):
        pages = sorted(by_file[src], key=lambda x: (x == "?", x))[:2]
        lines.append(f"{src} (page {', '.join(str(p) for p in pages)})")
    return lines


# =============================================================================
# HTTP routes (each decorator registers one URL + HTTP method)
# =============================================================================


@app.get("/api/health")
def health():
    """Liveness check for Docker / load balancers."""
    return {"status": "ok"}


@app.post("/api/process")
async def process_documents(
    x_session_id: str | None = Header(None, alias="X-Session-Id"),
    files: list[UploadFile] | None = File(None),
):
    """
    Accept PDF/DOCX uploads: extract text, chunk, embed into Chroma, update session.
    If the session already had an index, new documents are **appended** when possible.
    """
    files = files or []
    session_id = _normalize_session_id(x_session_id)
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_FILES_PER_REQUEST} files per upload.",
        )

    if not load_openai_api_key():
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured.")

    state = _session_get(session_id)
    processed_hashes: set[str] = state.get("processed_hashes", set())

    # Read bytes, SHA-256 per file; dedupe within this request (same content = one index pass)
    rows: list[tuple[str, bytes, str]] = []
    for f in files:
        ext = os.path.splitext(f.filename or "")[1].lower()
        if ext not in (".pdf", ".docx"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {f.filename}. Use PDF or DOCX.",
            )
        raw = await f.read()
        digest = hashlib.sha256(raw).hexdigest()
        rows.append((f.filename or "document", raw, digest))

    seen_in_request: set[str] = set()
    skipped_duplicate_in_request: list[dict] = []
    unique_rows: list[tuple[str, bytes, str]] = []
    for name, raw, digest in rows:
        if digest in seen_in_request:
            skipped_duplicate_in_request.append(
                {"filename": name, "sha256": digest, "reason": "duplicate_content_in_request"},
            )
            continue
        seen_in_request.add(digest)
        unique_rows.append((name, raw, digest))

    skipped_already_indexed: list[dict] = []
    to_process: list[tuple[str, bytes, str]] = []
    for name, raw, digest in unique_rows:
        if digest in processed_hashes:
            skipped_already_indexed.append({"filename": name, "sha256": digest})
        else:
            to_process.append((name, raw, digest))

    if not to_process:
        parts: list[str] = []
        if skipped_already_indexed:
            parts.append(
                "These files are already in your index; nothing to re-index."
            )
        if skipped_duplicate_in_request:
            parts.append(
                f"Skipped {len(skipped_duplicate_in_request)} duplicate(s) in this upload."
            )
        message = " ".join(parts) if parts else "Nothing new to index."
        return {
            "ok": True,
            "message": message,
            "indexed": [],
            "skipped_already_indexed": skipped_already_indexed,
            "skipped_duplicate_in_request": skipped_duplicate_in_request,
        }

    if len(processed_hashes) + len(to_process) > MAX_DISTINCT_DOCUMENTS_PER_SESSION:
        raise HTTPException(
            status_code=400,
            detail=(
                f"This session already has {MAX_DISTINCT_DOCUMENTS_PER_SESSION} indexed documents. "
                "Remove one from the server index (or clear the entire index) before adding more."
            ),
        )

    all_chunks: list[dict] = []
    indexed_meta: list[dict] = []
    for name, raw, digest in to_process:
        ext = os.path.splitext(name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(raw)
            path = tmp.name
        try:
            pages = DocumentLoader.load_file(path)
            all_chunks.extend(TextChunker.split_pages(pages, name))
        finally:
            if os.path.exists(path):
                os.remove(path)
        indexed_meta.append(
            {"filename": name, "size": len(raw), "sha256": digest},
        )

    if not all_chunks:
        raise HTTPException(status_code=400, detail="No text could be extracted from the new files.")

    collection = _chroma_collection_name(session_id)
    existing_pipeline = state.get("pipeline")
    indexed_before = bool(state.get("indexed")) and existing_pipeline is not None
    # Reuse the same RAGPipeline + Chroma collection when we are only adding new files;
    # otherwise build a fresh pipeline (first index, or pipeline object missing store).
    can_append = (
        indexed_before
        and getattr(existing_pipeline, "vector_store", None) is not None
    )

    if can_append:
        existing_pipeline.append_chunks(all_chunks)
        pipeline = existing_pipeline
    else:
        pipeline = RAGPipeline()
        pipeline.build_index(all_chunks, collection_name=collection)

    new_hashes = {r[2] for r in to_process}
    with _lock:
        st = _sessions[session_id]
        st["indexed"] = True
        st["pipeline"] = pipeline
        st["processed_hashes"] = processed_hashes | new_hashes
        st["chat_history"] = []
        # Merge new file metadata with existing list keyed by sha256 (dedupe by hash).
        by_hash = {d["sha256"]: d for d in st.get("indexed_documents", [])}
        for m in indexed_meta:
            by_hash[m["sha256"]] = {
                "filename": m["filename"],
                "sha256": m["sha256"],
                "size": m["size"],
            }
        st["indexed_documents"] = sorted(
            by_hash.values(),
            key=lambda x: str(x["filename"]).lower(),
        )

    msg_parts = [f"Indexed {len(to_process)} new document(s)."]
    if skipped_already_indexed:
        msg_parts.append(f"Skipped {len(skipped_already_indexed)} already indexed.")
    if skipped_duplicate_in_request:
        msg_parts.append(f"Skipped {len(skipped_duplicate_in_request)} duplicate(s) in this upload.")

    return {
        "ok": True,
        "message": " ".join(msg_parts),
        "indexed": indexed_meta,
        "skipped_already_indexed": skipped_already_indexed,
        "skipped_duplicate_in_request": skipped_duplicate_in_request,
    }


def _corpus_documents_for_session(state: dict) -> list[dict]:
    """Only session metadata — never infer from Chroma alone (avoids ghost rows vs processed_hashes)."""
    return list(state.get("indexed_documents") or [])


@app.get("/api/corpus")
def get_corpus(
    x_session_id: str | None = Header(None, alias="X-Session-Id"),
):
    """Return whether this session has an index and the list of indexed document metadata."""
    session_id = _normalize_session_id(x_session_id)
    state = _session_get(session_id)
    return {
        "indexed": bool(state.get("indexed")),
        "documents": _corpus_documents_for_session(state),
        "distinct_count": len(state.get("processed_hashes", set())),
    }


def _find_indexed_entry(state: dict, sha256: str | None, filename: str | None) -> dict | None:
    docs = list(state.get("indexed_documents") or [])
    if sha256:
        for d in docs:
            if d.get("sha256") == sha256:
                return d
    fn = (filename or "").strip()
    if fn:
        for d in docs:
            if str(d.get("filename", "")) == fn:
                return d
    return None


@app.post("/api/corpus/remove")
def remove_corpus_document(
    body: RemoveCorpusBody,
    x_session_id: str | None = Header(None, alias="X-Session-Id"),
):
    """Remove one document from Chroma and session (frees a slot)."""
    session_id = _normalize_session_id(x_session_id)
    sha_in = (body.sha256 or "").strip() or None
    fn_in = (body.filename or "").strip() or None
    if not sha_in and not fn_in:
        raise HTTPException(status_code=400, detail="Provide sha256 or filename.")

    state = _session_get(session_id)
    entry = _find_indexed_entry(state, sha_in, fn_in)
    fname = str(entry["filename"]) if entry else fn_in
    digest = entry.get("sha256") if entry else sha_in
    if not fname:
        raise HTTPException(status_code=404, detail="Document not found in this session.")

    pipeline = state.get("pipeline")
    vs = getattr(pipeline, "vector_store", None) if pipeline else None
    if vs is None:
        raise HTTPException(status_code=400, detail="Nothing is indexed for this session.")

    try:
        removed_n = vs.delete_by_source(fname)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not delete vectors: {exc}") from exc
    if removed_n == 0 and entry is None:
        raise HTTPException(status_code=404, detail="No vectors found for that file.")

    with _lock:
        st = _sessions.get(session_id)
        if not st:
            raise HTTPException(status_code=404, detail="Session not found.")
        ph: set = st.setdefault("processed_hashes", set())
        if digest:
            ph.discard(digest)
        new_docs = [d for d in st.get("indexed_documents", []) if str(d.get("filename", "")) != fname]
        st["indexed_documents"] = sorted(new_docs, key=lambda x: str(x["filename"]).lower())
        if not st["indexed_documents"]:
            st["indexed"] = False
            st["pipeline"] = None
            st["processed_hashes"] = set()
        st["chat_history"] = []

    return {"ok": True, "removed": fname}


@app.post("/api/corpus/clear")
def clear_entire_corpus(
    x_session_id: str | None = Header(None, alias="X-Session-Id"),
):
    """Delete all vectors for this session and reset index state (frees all slots)."""
    session_id = _normalize_session_id(x_session_id)
    state = _session_get(session_id)
    pipeline = state.get("pipeline")
    vs = getattr(pipeline, "vector_store", None) if pipeline else None
    if vs is not None:
        try:
            vs.wipe_entire_collection()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not clear vector store: {exc}") from exc

    with _lock:
        st = _sessions.get(session_id)
        if not st:
            raise HTTPException(status_code=404, detail="Session not found.")
        st["indexed"] = False
        st["pipeline"] = None
        st["processed_hashes"] = set()
        st["indexed_documents"] = []
        st["chat_history"] = []

    return {"ok": True, "message": "Index cleared for this session."}


def _append_chat_turn(session_id: str, question: str, answer: str, sources: list[str]) -> None:
    with _lock:
        st = _sessions.get(session_id)
        if not st:
            return
        hist = st.setdefault("chat_history", [])
        hist.append(
            {
                "question": question,
                "answer": answer,
                "sources": sources,
            },
        )


@app.get("/api/chat/history")
def get_chat_history(
    x_session_id: str | None = Header(None, alias="X-Session-Id"),
):
    """Return stored Q&A turns for this session (in-memory until API restarts)."""
    session_id = _normalize_session_id(x_session_id)
    state = _session_get(session_id)
    turns = state.get("chat_history", [])
    return {"turns": turns}


@app.post("/api/chat")
def chat(
    body: ChatBody,
    x_session_id: str | None = Header(None, alias="X-Session-Id"),
):
    """
    Stream the assistant reply using **Server-Sent Events** (SSE): each line is `data: {...}`.
    The UI reads tokens as they arrive until a final event with `done: true` and optional `sources`.
    """
    session_id = _normalize_session_id(x_session_id)
    state = _session_get(session_id)
    if not state.get("indexed") or state.get("pipeline") is None:
        raise HTTPException(
            status_code=400,
            detail="Please upload and process documents first.",
        )

    pipeline: RAGPipeline = state["pipeline"]
    question = body.question.strip()

    def sse():
        stream, retrieved, has_grounding, fallback = pipeline.answer_stream(question)
        if not has_grounding:
            text = fallback or NO_ANSWER_MESSAGE
            yield f"data: {json.dumps({'token': text})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': []})}\n\n"
            _append_chat_turn(session_id, question, text, [])
            return

        assert stream is not None
        full = ""
        for ch in stream:
            delta = ch.choices[0].delta.content
            if delta:
                full += delta
                yield f"data: {json.dumps({'token': delta})}\n\n"

        if not full.strip():
            full = NO_ANSWER_MESSAGE
            yield f"data: {json.dumps({'token': full})}\n\n"

        sources = _format_sources(retrieved, full, True)
        yield f"data: {json.dumps({'done': True, 'sources': sources})}\n\n"
        _append_chat_turn(session_id, question, full, sources)

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
