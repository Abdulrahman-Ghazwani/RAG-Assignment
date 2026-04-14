from openai import OpenAI

from app.config import (
    CHAT_MODEL,
    MIN_RELEVANCE_SCORE,
    NO_ANSWER_MESSAGE,
    OPENAI_API_KEY,
    TOP_K,
)
from app.services.embedder import Embedder
from app.services.vector_store import FaissVectorStore


class RAGPipeline:
    def __init__(self):
        self.embedder = Embedder()
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.vector_store = None

    def build_index(self, chunks: list[dict]):
        valid = [
            c for c in chunks
            if isinstance(c.get("content"), str) and c["content"].strip()
        ]
        texts = [c["content"].strip() for c in valid]
        if not texts:
            raise ValueError("No valid chunks found to build the index.")

        embeddings = self.embedder.embed_texts(texts)
        self.vector_store = FaissVectorStore(len(embeddings[0]))
        self.vector_store.add(embeddings, valid)

    def retrieve(self, question: str):
        if self.vector_store is None:
            raise ValueError("Vector store is not built yet. Process documents first.")
        q_emb = self.embedder.embed_query(question)
        hits = self.vector_store.search(q_emb, top_k=TOP_K)
        return [h for h in hits if h.get("score", -1.0) >= MIN_RELEVANCE_SCORE]

    def build_prompt(self, question: str, retrieved_chunks: list[dict]) -> str:
        ctx = "\n\n".join(
            f"[Page {c.get('page', '?')}]\n{c['content']}" for c in retrieved_chunks
        )
        return (
            "Answer only using the context. Use the same language as the question.\n"
            "If the answer is not clearly in the context, respond with exactly:\n"
            f"{NO_ANSWER_MESSAGE}\n"
            "No external knowledge, no page numbers in the answer, no sources list in the answer.\n\n"
            f"Context:\n{ctx}\n\nQuestion:\n{question}"
        )

    def answer_stream(self, question: str):
        retrieved_chunks = self.retrieve(question)
        if not retrieved_chunks:
            return None, [], False, NO_ANSWER_MESSAGE

        prompt = self.build_prompt(question, retrieved_chunks)
        stream = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You answer only from provided context."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            stream=True,
        )
        return stream, retrieved_chunks, True, None
