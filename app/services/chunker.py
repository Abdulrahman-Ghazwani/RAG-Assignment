from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextChunker:
    @staticmethod
    def split_text(text: str, source_name: str):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=120
        )

        chunks = splitter.split_text(text)

        return [
            {
                "content": chunk,
                "source": source_name,
                "chunk_id": idx
            }
            for idx, chunk in enumerate(chunks)
            if isinstance(chunk, str) and chunk.strip()
        ]