"""Vector store (ChromaDB, persistent on disk) and embedding functions."""

from __future__ import annotations

import chromadb
from chromadb.utils import embedding_functions

from . import config

_client = None
_collection = None


def embedding_function():
    if config.EMBEDDING_PROVIDER == "openai":
        if not config.OPENAI_API_KEY:
            raise RuntimeError("EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is not set")
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=config.OPENAI_API_KEY, model_name=config.EMBEDDING_MODEL
        )
    # Local: all-MiniLM-L6-v2 (ONNX), downloaded once (~80 MB), no API cost.
    return embedding_functions.DefaultEmbeddingFunction()


def get_collection():
    global _client, _collection
    if _collection is None:
        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        _collection = _client.get_or_create_collection(
            name=config.COLLECTION_FULL,
            embedding_function=embedding_function(),
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def reset_collection() -> None:
    """Drop every indexed document."""
    global _collection
    get_collection()
    _client.delete_collection(config.COLLECTION_FULL)
    _collection = None
