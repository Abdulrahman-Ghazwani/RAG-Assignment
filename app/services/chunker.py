from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextChunker:
    @staticmethod
    def split_pages(pages: list, source_name: str):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=120
        )

        all_chunks = []

        for page in pages:
            chunks = splitter.split_text(page["content"])

            for chunk in chunks:
                if chunk.strip():
                    all_chunks.append({
                        "content": chunk,
                        "source": source_name,
                        "page": page["page"],
                    })

        return all_chunks