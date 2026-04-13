import os
import tempfile
import streamlit as st
from collections import defaultdict


from app.services.document_loader import DocumentLoader
from app.services.chunker import TextChunker
from app.services.rag_pipeline import RAGPipeline
from app.config import NO_ANSWER_MESSAGE


st.set_page_config(page_title="Conversational RAG Pipeline", layout="wide")
st.title("Conversational RAG Pipeline")
st.write("Upload PDF or DOCX files in Arabic or English, then ask questions.")


if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()

if "indexed" not in st.session_state:
    st.session_state.indexed = False

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []


for item in st.session_state.qa_history:
    with st.chat_message("user"):
        st.write(item["question"])
    with st.chat_message("assistant"):
        st.write(item["answer"])
        if item["sources"]:
            st.markdown("### Sources")
            for source in item["sources"]:
                st.write(f"- {source}")


uploaded_files = st.file_uploader(
    "Upload up to 3 files",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

if uploaded_files and len(uploaded_files) > 3:
    st.error("Maximum allowed uploads is 3 files.")


if uploaded_files and st.button("Process Documents"):
    if len(uploaded_files) > 3:
        st.stop()

    all_chunks = []

    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        try:
            pages = DocumentLoader.load_file(temp_path)
            chunks = TextChunker.split_pages(pages, uploaded_file.name)
            all_chunks.extend(chunks)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    if not all_chunks:
        st.error("No text could be extracted from the uploaded files.")
    else:
        st.session_state.pipeline = RAGPipeline()
        st.session_state.pipeline.build_index(all_chunks)
        st.session_state.indexed = True
        st.success("Documents processed successfully.")


question = st.chat_input("Ask a question about the uploaded documents...")

if question:
    history_item = {
        "question": question,
        "answer": "",
        "sources": []
    }

    with st.chat_message("user"):
        st.write(question)

    if not st.session_state.indexed:
        with st.chat_message("assistant"):
            message = "Please upload and process documents first."
            st.warning(message)
            history_item["answer"] = message
    else:
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            stream, retrieved_chunks, has_grounding, fallback_message = (
                st.session_state.pipeline.answer_stream(question)
            )

            if not has_grounding:
                full_response = fallback_message or NO_ANSWER_MESSAGE
                response_placeholder.markdown(full_response)
            else:
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        full_response += delta
                        response_placeholder.markdown(full_response)
            history_item["answer"] = full_response

        normalized_response = " ".join(full_response.split()).strip().lower()
        normalized_no_answer = " ".join(NO_ANSWER_MESSAGE.split()).strip().lower()
        should_show_sources = (
            has_grounding
            and bool(retrieved_chunks)
            and normalized_response != normalized_no_answer
        )

        if should_show_sources:
            sources_map = defaultdict(set)

            for chunk in retrieved_chunks:
                source = chunk["source"]
                page = chunk.get("page")
                if page:
                    sources_map[source].add(page)
                else:
                    sources_map[source].add("?")

            st.markdown("### Sources")
            for src in sorted(sources_map.keys()):
                pages = sorted(sources_map[src], key=lambda x: (x == "?", x))
                pages = pages[:2]
                pages_str = ", ".join(str(p) for p in pages)
                source_entry = f"{src} (page {pages_str})"
                st.write(f"- {source_entry}")
                history_item["sources"].append(source_entry)

    if not history_item["answer"]:
        history_item["answer"] = NO_ANSWER_MESSAGE

    st.session_state.qa_history.append(history_item)