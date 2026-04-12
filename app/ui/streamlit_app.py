import os
import tempfile
import streamlit as st

from app.services.document_loader import DocumentLoader
from app.services.chunker import TextChunker
from app.services.rag_pipeline import RAGPipeline


st.set_page_config(page_title="Conversational RAG", layout="wide")
st.title("Conversational RAG Pipeline")
st.write("Upload PDF or DOCX files in Arabic or English, then ask questions.")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()

if "indexed" not in st.session_state:
    st.session_state.indexed = False

uploaded_files = st.file_uploader(
    "Upload up to 3 files",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

if uploaded_files and st.button("Process Documents"):
    all_chunks = []

    for uploaded_file in uploaded_files[:3]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        text = DocumentLoader.load_pdf(temp_path)
        chunks = TextChunker.split_text(text, uploaded_file.name)
        all_chunks.extend(chunks)

        os.remove(temp_path)

    st.session_state.pipeline.build_index(all_chunks)
    st.session_state.indexed = True
    st.success("Documents processed successfully.")

question = st.chat_input("Ask a question about the uploaded documents...")

if question:
    if not st.session_state.indexed:
        st.warning("Please upload and process documents first.")
    else:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            stream, retrieved_chunks = st.session_state.pipeline.answer_stream(question)

            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_response += delta
                    response_placeholder.markdown(full_response)

            st.markdown("### Sources")
            used_sources = {
                f"{chunk['source']} (chunk {chunk['chunk_id']})"
                for chunk in retrieved_chunks
            }
            for src in used_sources:
                st.write(f"- {src}")