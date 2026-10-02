"""Configuration — everything is an environment variable with a sane default.

OPENAI_API_KEY      OpenAI key. If absent, the app runs in LOCAL mode:
                    local embeddings (all-MiniLM-L6-v2 via ChromaDB) and an
                    extractive answer built from the retrieved passages.
EMBEDDING_PROVIDER  "openai" | "local"  (default: openai when a key exists)
EMBEDDING_MODEL     default text-embedding-3-small
CHAT_MODEL          default gpt-4o-mini
CHROMA_DIR          where the vector database lives (default ./chroma_db)
COLLECTION          collection name (default documents)
CHUNK_SIZE          characters per chunk (default 1000)
CHUNK_OVERLAP       overlap between chunks (default 150)
TOP_K               passages retrieved per question (default 5)
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai" if OPENAI_API_KEY else "local").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "./chroma_db"))
COLLECTION = os.getenv("COLLECTION", "documents")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "5"))

# Collections are tied to an embedding model: a different model needs a
# different collection, otherwise vector dimensions would not match.
COLLECTION_FULL = f"{COLLECTION}_{EMBEDDING_PROVIDER}_{EMBEDDING_MODEL if EMBEDDING_PROVIDER == 'openai' else 'minilm'}".replace("-", "_")


def llm_available() -> bool:
    return bool(OPENAI_API_KEY)
