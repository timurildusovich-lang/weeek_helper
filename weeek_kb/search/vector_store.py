from __future__ import annotations

import threading
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from weeek_kb.config import CHROMA_DIR, OPENAI_API_KEY, OPENAI_EMBED_MODEL

# PersistentClient нельзя безопасно создавать из разных потоков; один процесс — один клиент + lock.
_chroma_client: chromadb.PersistentClient | None = None
_chroma_lock = threading.RLock()


def _embedding_fn():
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "Set OPENAI_API_KEY in .env (also accepted: OPEN_APY_KEY, OPENAI_APY_KEY)"
        )
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=OPENAI_EMBED_MODEL,
    )


def get_client() -> chromadb.PersistentClient:
    global _chroma_client
    with _chroma_lock:
        if _chroma_client is None:
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        return _chroma_client


def get_collection(name: str):
    with _chroma_lock:
        client = get_client()
        ef = _embedding_fn()
        return client.get_collection(name=name, embedding_function=ef)


def create_collection(name: str, reset: bool = False):
    with _chroma_lock:
        client = get_client()
        ef = _embedding_fn()
        if reset:
            try:
                client.delete_collection(name)
            except Exception:
                pass
        return client.get_or_create_collection(name=name, embedding_function=ef)


def query_collection(
    collection_name: str,
    query_text: str,
    n_results: int,
) -> dict[str, Any]:
    with _chroma_lock:
        client = get_client()
        ef = _embedding_fn()
        col = client.get_collection(name=collection_name, embedding_function=ef)
        return col.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
