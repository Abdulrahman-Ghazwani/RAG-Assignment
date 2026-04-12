from openai import OpenAI
from app.config import OPENAI_API_KEY, EMBEDDING_MODEL


class Embedder:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        cleaned_texts = []

        for text in texts:
            if text is None:
                continue

            if not isinstance(text, str):
                text = str(text)

            text = text.strip()

            if text:
                cleaned_texts.append(text)

        if not cleaned_texts:
            raise ValueError("No valid texts found for embedding.")

        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=cleaned_texts
        )

        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        if not isinstance(text, str):
            text = str(text)

        text = text.strip()

        if not text:
            raise ValueError("Query text is empty.")

        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[text]
        )

        return response.data[0].embedding