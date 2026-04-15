import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
import tempfile
from collections import defaultdict

import streamlit as st

from app.config import NO_ANSWER_MESSAGE, load_openai_api_key
from app.services.chunker import TextChunker
from app.services.document_loader import DocumentLoader
from app.services.rag_pipeline import RAGPipeline

st.set_page_config(page_title="Document RAG", layout="wide")

st.markdown(
    """
    <style>
    /* Sticky upload panel: stays at top while scrolling chat */
    section.main div[data-testid="stVerticalBlockBorderWrapper"]:has(.stFileUploader) {
        position: sticky;
        top: 3.25rem;
        z-index: 900;
        background: color-mix(in srgb, var(--background-color, #0e1117) 94%, transparent);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding-bottom: 1rem;
        margin-bottom: 0.75rem;
        border-bottom: 1px solid rgba(250, 250, 250, 0.08);
    }
    /* Full-viewport overlay; inner layout: icon centered above label with clear gap */
    div[data-testid="stSpinner"] {
        position: fixed !important;
        inset: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        max-width: none !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        background: rgba(14, 17, 23, 0.78) !important;
        backdrop-filter: blur(6px) !important;
        -webkit-backdrop-filter: blur(6px) !important;
        z-index: 10000 !important;
    }
    div[data-testid="stSpinner"] > div {
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        gap: 1.25rem !important;
        row-gap: 1.25rem !important;
        width: 100% !important;
        max-width: min(90vw, 28rem) !important;
        padding: 0 1rem !important;
        box-sizing: border-box !important;
    }
    div[data-testid="stSpinner"] > div > div:last-child {
        justify-content: center !important;
        align-items: center !important;
        text-align: center !important;
        width: 100% !important;
    }
    div[data-testid="stSpinner"] > div > div:last-child p {
        margin: 0 !important;
        text-align: center !important;
        line-height: 1.5 !important;
    }
    div[data-testid="stSpinner"] svg,
    div[data-testid="stSpinner"] i {
        width: 3rem !important;
        height: 3rem !important;
        flex-shrink: 0 !important;
    }
    div[data-testid="stSpinner"] i {
        border-top-color: rgba(255, 107, 53, 0.35) !important;
        border-right-color: rgba(255, 107, 53, 0.35) !important;
        border-bottom-color: rgba(255, 107, 53, 0.35) !important;
        border-left-color: #ff6b35 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Conversational RAG Pipeline")
st.caption("Upload up to 3 PDF or DOCX files — ask in Arabic or English.")

_key_sig = load_openai_api_key() or ""
_prev_key = st.session_state.get("_openai_key_sig")
if _prev_key is not None and _prev_key != _key_sig:
    for _k in ("pipeline", "indexed", "qa_history"):
        st.session_state.pop(_k, None)
st.session_state._openai_key_sig = _key_sig

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

with st.container():
    uploaded = st.file_uploader(
        "Upload up to 3 files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )
    if uploaded and len(uploaded) > 3:
        st.error("Maximum allowed uploads is 3 files.")

    process_clicked = st.button("Process Documents")
    if uploaded and process_clicked:
        if len(uploaded) > 3:
            st.stop()

        with st.spinner("Processing documents…"):
            all_chunks = []
            for f in uploaded:
                ext = os.path.splitext(f.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(f.read())
                    path = tmp.name
                try:
                    pages = DocumentLoader.load_file(path)
                    all_chunks.extend(TextChunker.split_pages(pages, f.name))
                finally:
                    if os.path.exists(path):
                        os.remove(path)

            if not all_chunks:
                st.error("No text could be extracted from the uploaded files.")
            else:
                st.session_state.pipeline = RAGPipeline()
                st.session_state.pipeline.build_index(all_chunks)
                st.session_state.indexed = True
                st.success("Documents processed successfully.")

for item in st.session_state.qa_history:
    with st.chat_message("user"):
        st.write(item["question"])
    with st.chat_message("assistant"):
        st.write(item["answer"])
        if item["sources"]:
            st.markdown("**Sources**")
            for s in item["sources"]:
                st.write(f"- {s}")

question = st.chat_input("Ask a question about the uploaded documents...")

if question:
    history_item = {"question": question, "answer": "", "sources": []}
    with st.chat_message("user"):
        st.write(question)

    if not st.session_state.indexed:
        with st.chat_message("assistant"):
            msg = "Please upload and process documents first."
            st.warning(msg)
            history_item["answer"] = msg
    else:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full = ""
            with st.spinner("Searching documents & generating answer…"):
                stream, retrieved, has_grounding, fallback = (
                    st.session_state.pipeline.answer_stream(question)
                )
            if not has_grounding:
                full = fallback or NO_ANSWER_MESSAGE
                placeholder.markdown(full)
            else:
                for ch in stream:
                    d = ch.choices[0].delta.content
                    if d:
                        full += d
                        placeholder.markdown(full)
            history_item["answer"] = full

        norm_a = " ".join(full.split()).strip().lower()
        norm_n = " ".join(NO_ANSWER_MESSAGE.split()).strip().lower()
        show_src = has_grounding and retrieved and norm_a != norm_n
        if show_src:
            by_file = defaultdict(set)
            for c in retrieved:
                by_file[c["source"]].add(c.get("page") or "?")
            st.markdown("**Sources**")
            for src in sorted(by_file):
                pages = sorted(by_file[src], key=lambda x: (x == "?", x))[:2]
                line = f"{src} (page {', '.join(str(p) for p in pages)})"
                st.write(f"- {line}")
                history_item["sources"].append(line)

    if not history_item["answer"]:
        history_item["answer"] = NO_ANSWER_MESSAGE
    st.session_state.qa_history.append(history_item)
