from openai import OpenAI
from app.config import OPENAI_API_KEY, CHAT_MODEL, TOP_K
from app.services.embedder import Embedder
from app.services.vector_store import FaissVectorStore


class RAGPipeline:
    def __init__(self):
        self.embedder = Embedder()
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.vector_store = None

    def build_index(self, chunks: list[dict]):
        valid_chunks = [
            chunk for chunk in chunks
            if isinstance(chunk.get("content"), str) and chunk["content"].strip()
        ]

        texts = [chunk["content"].strip() for chunk in valid_chunks]

        if not texts:
            raise ValueError("No valid chunks found to build the index.")

        embeddings = self.embedder.embed_texts(texts)

        dim = len(embeddings[0])
        self.vector_store = FaissVectorStore(dim)
        self.vector_store.add(embeddings, valid_chunks)

    def retrieve(self, question: str):
        if self.vector_store is None:
            raise ValueError("Vector store is not built yet. Process documents first.")

        query_embedding = self.embedder.embed_query(question)
        return self.vector_store.search(query_embedding, top_k=TOP_K)

    def build_prompt(self, question: str, retrieved_chunks: list[dict]) -> str:
        context = "\n\n".join(
            [
                f"[Source: {chunk['source']} | Chunk: {chunk['chunk_id']}]\n{chunk['content']}"
                for chunk in retrieved_chunks
            ]
        )

        return f"""
You are a document QA assistant.

Answer only from the provided context.
If the answer is not clearly supported by the context, say:
"I could not find a supported answer in the retrieved context."

Rules:
- Keep the answer concise: 2 to 4 sentences.
- Do not mention chunk numbers.
- Do not say "uploaded documents do not contain enough information" unless the retrieved context truly lacks the answer.
- If multiple context pieces contribute to the answer, combine them.
- Prefer exact factual grounding over general explanations.
- End with a short citation label like: [Sources: 1, 2]

Context:
{context}

Question:
{question}
"""

    def answer_stream(self, question: str):
        retrieved_chunks = self.retrieve(question)
        prompt = self.build_prompt(question, retrieved_chunks)

        stream = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a document-grounded RAG assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            stream=True
        )

        return stream, retrieved_chunks