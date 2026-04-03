from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma

from app.rag.config import get_rag_runtime_config
from app.rag.embeddings import get_embedding_model


def _ensure_persist_dir() -> str:
    path = get_rag_runtime_config().persist_path
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def get_collection_name(audience: str) -> str:
    audience = (audience or "").strip().lower()
    runtime = get_rag_runtime_config()
    if audience == "student":
        return runtime.student_collection
    if audience == "professor":
        return runtime.professor_collection
    raise ValueError(f"Unsupported audience: {audience}")


@lru_cache(maxsize=2)
def get_vector_store(audience: str) -> Chroma:
    collection_name = get_collection_name(audience)
    persist_directory = _ensure_persist_dir()

    return Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=get_embedding_model(),
    )
