from openai import OpenAI
from app.config import (
    OPENAI_API_KEY,
    CHAT_MODEL,
    TOP_K,
    MIN_RELEVANCE_SCORE,
    NO_ANSWER_MESSAGE,
)
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
        retrieved = self.vector_store.search(query_embedding, top_k=TOP_K)
        return [
            chunk for chunk in retrieved
            if chunk.get("score", -1.0) >= MIN_RELEVANCE_SCORE
        ]

    def build_prompt(self, question: str, retrieved_chunks: list[dict]) -> str:
        context = "\n\n".join(
            [
                f"[Page {chunk.get('page', '?')}]\n{chunk['content']}"
                for chunk in retrieved_chunks
            ]
        )

        return f"""
    You are a helpful assistant answering questions only from the provided context.

    Rules:
    - Answer ONLY using the retrieved context.
    - Use extractive style: copy exact phrases/sentences from the context when answering.
    - Do not paraphrase facts and do not introduce new words, entities, or claims.
    - If the answer is not clearly in the context, say exactly:
      The uploaded documents do not contain enough information to answer this question.
    - If you cannot support the answer with at least one direct quote from context, return the same fallback sentence.
    - Respond in the same language as the question.
    - Keep the answer clear, concise, and accurate.
    - Do NOT guess or add external knowledge.
    - Do NOT mention page numbers in the answer text.
    - Do NOT include a sources list in the answer.
    - The interface will display sources separately.

    Context:
    {context}

    Question:
    {question}
    """

    def answer_stream(self, question: str):
        retrieved_chunks = self.retrieve(question)
        if not retrieved_chunks:
            return None, [], False, NO_ANSWER_MESSAGE

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
            temperature=0,
            stream=True
        )

        return stream, retrieved_chunks, True, None