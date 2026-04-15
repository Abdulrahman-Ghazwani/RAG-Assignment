from concurrent.futures import ThreadPoolExecutor

import tiktoken
from openai import OpenAI

from app.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_PARALLEL,
    EMBEDDING_MODEL,
)


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

    def _embed_one_batch(self, batch: list[str]) -> list[list[float]]:
        r = self.client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        return [item.embedding for item in r.data]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        cleaned: list[str] = []
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

        bs = max(1, EMBEDDING_BATCH_SIZE)
        batches: list[list[str]] = [cleaned[i : i + bs] for i in range(0, len(cleaned), bs)]

        if len(batches) == 1:
            return self._embed_one_batch(batches[0])

        workers = max(1, min(EMBEDDING_MAX_PARALLEL, len(batches)))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(self._embed_one_batch, batches))

        out: list[list[float]] = []
        for p in parts:
            out.extend(p)
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
