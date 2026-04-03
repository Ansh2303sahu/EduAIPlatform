from __future__ import annotations

from typing import Iterable

from app.core.config import settings
from app.rag.hybrid import hybrid_overlap_score
from app.rag.security.source_priority import boost_score_with_source_priority
from app.rag.schemas import RetrievedChunk, RetrievalQuery

_PROJECT_FOCUS_TERMS = {
    "architecture",
    "backend",
    "frontend",
    "database",
    "security",
    "testing",
    "integration",
    "analytics",
    "evaluation",
    "limitations",
    "api",
    "authentication",
}

_ACADEMIC_FOCUS_TERMS = {
    "structure",
    "argument",
    "evidence",
    "analysis",
    "clarity",
    "coherence",
    "referencing",
    "citation",
    "writing",
    "critical",
}

_PROFESSOR_POLICY_CATEGORIES = {
    "rubrics",
    "marking_policy",
    "moderation",
    "feedback_templates",
    "academic_quality",
}


def _clamp_unit_interval(score: float) -> float:
    return max(0.0, min(1.0, float(score)))


def _analysis_term_bonus(req: RetrievalQuery, chunk: RetrievedChunk) -> float:
    analysis_type = (req.analysis_type or "").strip().lower()
    haystack = " ".join(
        [
            chunk.document_title,
            chunk.section,
            chunk.category,
            chunk.content[:2000],
        ]
    ).lower()

    if analysis_type.endswith("project_review"):
        hits = sum(1 for term in _PROJECT_FOCUS_TERMS if term in haystack)
        return min(hits * 0.012, 0.09)

    hits = sum(1 for term in _ACADEMIC_FOCUS_TERMS if term in haystack)
    return min(hits * 0.01, 0.07)


def _preferred_category_bonus(req: RetrievalQuery, chunk: RetrievedChunk) -> float:
    preferred = [c.lower() for c in req.preferred_categories]
    if not preferred:
        return 0.0
    try:
        rank = preferred.index(chunk.category.lower())
    except ValueError:
        return 0.0
    return max(0.0, 0.09 - (rank * 0.015))


def _audience_bonus(req: RetrievalQuery, chunk: RetrievedChunk) -> float:
    bonus = min(req.official_bias, 0.3) if chunk.is_official else 0.0

    if req.audience == "professor":
        if chunk.category in _PROFESSOR_POLICY_CATEGORIES:
            bonus += 0.05
        if chunk.is_official:
            bonus += 0.04
    elif (req.analysis_type or "").strip().lower() == "student_project_review":
        if chunk.category in {"research_support", "critical_thinking"}:
            bonus += 0.03
    else:
        if chunk.category in {"writing", "referencing", "critical_thinking"}:
            bonus += 0.025

    return bonus


def rerank_score(req: RetrievalQuery, chunk: RetrievedChunk) -> float:
    meta_text = " ".join([chunk.document_title, chunk.section, chunk.category])
    lexical_score = hybrid_overlap_score(req.query, chunk.content, meta_text)

    preferred_category_bonus = _preferred_category_bonus(req, chunk)
    analysis_bonus = _analysis_term_bonus(req, chunk)
    audience_bonus = _audience_bonus(req, chunk)

    blended = (
        (chunk.score * settings.rag_semantic_weight)
        + (lexical_score * settings.rag_keyword_weight)
        + preferred_category_bonus
        + analysis_bonus
        + audience_bonus
    )
    boosted = boost_score_with_source_priority(blended, chunk.source_priority)
    return _clamp_unit_interval(boosted)


def rerank_chunks(req: RetrievalQuery, chunks: Iterable[RetrievedChunk]) -> list[RetrievedChunk]:
    rescored = [
        chunk.model_copy(update={"score": rerank_score(req, chunk)})
        for chunk in list(chunks)
    ]
    return sorted(
        rescored,
        key=lambda item: (item.score, item.is_official, item.source_priority),
        reverse=True,
    )
