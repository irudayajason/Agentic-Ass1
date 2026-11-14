"""ChromaDB access. One persistent client, one LangChain `Chroma` wrapper per collection.

Collections:
  labelled_examples  categorised payees, used as dynamic few-shot examples by the Categorizer
  transactions       every transaction of the loaded statement (Phase 5)
  finance_kb         budgeting / saving guidance notes (Phase 5)
"""

from __future__ import annotations

from functools import lru_cache

import chromadb
from langchain_chroma import Chroma

from spendsense import config
from spendsense.llm import get_embeddings


@lru_cache(maxsize=1)
def _persistent_client() -> chromadb.ClientAPI:
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def get_store(collection: str, client: chromadb.ClientAPI | None = None) -> Chroma:
    """LangChain vector store over a Chroma collection (cosine similarity)."""
    return Chroma(
        collection_name=collection,
        embedding_function=get_embeddings(),
        client=client or _persistent_client(),
        collection_metadata={"hnsw:space": "cosine"},
    )


def ephemeral_client() -> chromadb.ClientAPI:
    """In-memory client for tests and throwaway runs."""
    return chromadb.EphemeralClient()


def reset_collection(collection: str, client: chromadb.ClientAPI | None = None) -> Chroma:
    client = client or _persistent_client()
    try:
        client.delete_collection(collection)
    except Exception:  # noqa: BLE001 - missing collection is fine
        pass
    return get_store(collection, client)
