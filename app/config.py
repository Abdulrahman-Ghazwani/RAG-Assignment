import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _reload_env() -> None:
    path = _PROJECT_ROOT / ".env"
    try:
        load_dotenv(path, override=True, encoding="utf-8-sig")
    except TypeError:
        load_dotenv(path, override=True)


def _normalize_openai_api_key(raw: str | None) -> str | None:
    if raw is None:
        return None
    key = raw.replace("\ufeff", "").replace("\r", "")
    key = "".join(key.split())
    if len(key) >= 2 and key[0] == key[-1] and key[0] in "\"'":
        key = key[1:-1].strip()
    return key or None


def load_openai_api_key() -> str | None:
    _reload_env()
    return _normalize_openai_api_key(os.getenv("OPENAI_API_KEY"))


_reload_env()

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_documents")

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# Embeddings: larger batches + limited parallelism = fewer round trips (same model & quality).
# OpenAI allows many inputs per request; tune if you hit rate limits.
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "128"))
EMBEDDING_MAX_PARALLEL = int(os.getenv("EMBEDDING_MAX_PARALLEL", "4"))

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
TOP_K = 4
MIN_RELEVANCE_SCORE = 0.30
NO_ANSWER_MESSAGE = (
    "The uploaded documents do not contain enough information to answer this question."
)
