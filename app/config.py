import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

CHUNK_SIZE = 700
CHINK_OVERLAP = 120
TOP_K = 4
MIN_RELEVANCE_SCORE = 0.30
NO_ANSWER_MESSAGE = (
    "The uploaded documents do not contain enough information to answer this question."
)