from __future__ import annotations

from app.rag import retriever as rag_retriever
from app.rag.schemas import RetrievalFilters, RetrievalQuery, RetrievalResult


def retrieve_professor_context(
    *,
    query: str,
    top_k: int = 4,
    category: str | None = None,
    version: str | None = None,
    ml_signals: dict | None = None,
    submission_type: str | None = None,
    official_only: bool | None = None,
    analysis_type: str | None = None,
    preferred_categories: list[str] | None = None,
    official_bias: float = 0.0,
) -> RetrievalResult:
    req = RetrievalQuery(
        audience="professor",
        query=query,
        top_k=top_k,
        filters=RetrievalFilters(
            category=category,
            version=version,
            status="active",
            is_official=official_only,
        ),
        ml_signals=ml_signals or {},
        submission_type=submission_type,
        analysis_type=analysis_type,
        preferred_categories=preferred_categories or [],
        official_bias=official_bias,
    )
    return rag_retriever.retrieve(req)
