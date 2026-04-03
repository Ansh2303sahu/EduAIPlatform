from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings
import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"

@dataclass(frozen=True)
class RagRuntimeConfig:
    persist_path: Path
    student_collection: str
    professor_collection: str
    chunk_size: int
    chunk_overlap: int
    min_confidence: float
    top_k_initial: int
    embedding_model: str


def get_rag_runtime_config() -> RagRuntimeConfig:
    return RagRuntimeConfig(
        persist_path=settings.rag_persist_path,
        student_collection=settings.rag_student_collection,
        professor_collection=settings.rag_professor_collection,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        min_confidence=settings.rag_min_confidence,
        top_k_initial=settings.rag_top_k_initial,
        embedding_model=settings.rag_embedding_model,
    )
