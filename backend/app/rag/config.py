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
    top_k_final: int
    embedding_model: str
    semantic_weight: float
    keyword_weight: float
    mmr_lambda: float
    mmr_fetch_k_min: int
    mmr_fetch_k_max: int
    query_excerpt_chars: int
    query_title_chars: int
    dynamic_keyword_limit: int


def get_rag_runtime_config() -> RagRuntimeConfig:
    return RagRuntimeConfig(
        persist_path=settings.rag_persist_path,
        student_collection=settings.rag_student_collection,
        professor_collection=settings.rag_professor_collection,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        min_confidence=settings.rag_min_confidence,
        top_k_initial=settings.rag_top_k_initial,
        top_k_final=settings.rag_top_k_final,
        embedding_model=settings.rag_embedding_model,
        semantic_weight=settings.rag_semantic_weight,
        keyword_weight=settings.rag_keyword_weight,
        mmr_lambda=settings.rag_mmr_lambda,
        mmr_fetch_k_min=settings.rag_mmr_fetch_k_min,
        mmr_fetch_k_max=settings.rag_mmr_fetch_k_max,
        query_excerpt_chars=settings.rag_query_excerpt_chars,
        query_title_chars=settings.rag_query_title_chars,
        dynamic_keyword_limit=settings.rag_dynamic_keyword_limit,
    )
