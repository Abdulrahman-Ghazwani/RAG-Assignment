from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_OVERLAP, CHUNK_SIZE


class TextChunker:
    @staticmethod
    def split_pages(pages: list, source_name: str):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        all_chunks = []
        for page in pages:
            for chunk in splitter.split_text(page["content"]):
                if chunk.strip():
                    all_chunks.append({
                        "content": chunk,
                        "source": source_name,
                        "page": page["page"],
                    })
        return all_chunks
