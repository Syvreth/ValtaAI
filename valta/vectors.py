from __future__ import annotations

import os

os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from valta.config import Settings


class _StoredEmbeddings(EmbeddingFunction[Documents]):
    """Chroma still wants an embedding function. Vectors are computed by Ollama."""

    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:
        raise RuntimeError("Pass embeddings from Ollama. This function does not embed text.")

    @staticmethod
    def name() -> str:
        return "stored"

    @staticmethod
    def build_from_config(config: dict) -> "_StoredEmbeddings":
        return _StoredEmbeddings()

    def get_config(self) -> dict:
        return {}

    def default_space(self) -> str:
        return "cosine"


class Vectors:
    def __init__(self, settings: Settings):
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(settings.chroma_path))
        self.collection = self.client.get_or_create_collection(
            name="valta",
            embedding_function=_StoredEmbeddings(),
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, record_id: str, embedding: list[float], kind: str, text: str) -> None:
        self.collection.upsert(
            ids=[record_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{"kind": kind}],
        )

    def query(self, embedding: list[float], k: int, kind: str | None = None) -> list[str]:
        where = {"kind": kind} if kind else None
        result = self.collection.query(query_embeddings=[embedding], n_results=k, where=where)
        ids = result.get("ids") or [[]]
        return list(ids[0]) if ids else []

    def delete(self, record_id: str) -> None:
        self.collection.delete(ids=[record_id])
