from openai import OpenAI
import tiktoken
from app.config import OPENAI_API_KEY, EMBEDDING_MODEL


class Embedder:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        try:
            self.encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

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

        # Keep each embeddings request below OpenAI request-token limit.
        max_request_tokens = 200000
        max_single_text_tokens = 8000
        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_tokens = 0

        for text in cleaned_texts:
            token_ids = self.encoding.encode(text)
            if len(token_ids) > max_single_text_tokens:
                token_ids = token_ids[:max_single_text_tokens]
                text = self.encoding.decode(token_ids)

            text_tokens = len(token_ids)
            if current_batch and current_tokens + text_tokens > max_request_tokens:
                batches.append(current_batch)
                current_batch = [text]
                current_tokens = text_tokens
            else:
                current_batch.append(text)
                current_tokens += text_tokens

        if current_batch:
            batches.append(current_batch)

        all_embeddings = []
        for batch in batches:
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch
            )
            all_embeddings.extend(item.embedding for item in response.data)

        return all_embeddings

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