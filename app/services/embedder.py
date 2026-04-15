from openai import OpenAI
import tiktoken

from app.config import EMBEDDING_MODEL


class Embedder:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        try:
            self.encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def _truncate(self, text: str) -> str:
        ids = self.encoding.encode(text)
        if len(ids) > 8000:
            return self.encoding.decode(ids[:8000])
        return text

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        cleaned = []
        for t in texts:
            if t is None:
                continue
            if not isinstance(t, str):
                t = str(t)
            t = t.strip()
            if t:
                cleaned.append(self._truncate(t))
        if not cleaned:
            raise ValueError("No valid texts found for embedding.")

        out: list[list[float]] = []
        batch_size = 16
        for i in range(0, len(cleaned), batch_size):
            batch = cleaned[i : i + batch_size]
            r = self.client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            out.extend(item.embedding for item in r.data)
        return out

    def embed_query(self, text: str) -> list[float]:
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        if not text:
            raise ValueError("Query text is empty.")
        r = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[self._truncate(text)],
        )
        return r.data[0].embedding
