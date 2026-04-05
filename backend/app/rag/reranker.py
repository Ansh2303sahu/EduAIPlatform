from __future__ import annotations

import re
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

_PROJECT_ALIGNED_CATEGORIES = {
    "software_engineering",
    "project_evaluation",
}

_PROJECT_SECONDARY_CATEGORIES = {
    "research_support",
}

_ACADEMIC_ALIGNED_CATEGORIES = {
    "writing",
    "referencing",
    "critical_thinking",
    "research_support",
    "academic_integrity",
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
        return -max(0.0, settings.rag_rerank_off_topic_penalty)
    base = max(0.0, settings.rag_rerank_preferred_category_bonus - (rank * 0.015))
    return base


def _audience_bonus(req: RetrievalQuery, chunk: RetrievedChunk) -> float:
    bonus = min(req.official_bias, 0.3) if chunk.is_official else 0.0

    if req.audience == "professor":
        if chunk.category in _PROFESSOR_POLICY_CATEGORIES:
            bonus += settings.rag_rerank_professor_policy_bonus
        if chunk.is_official:
            bonus += settings.rag_rerank_official_bonus
    elif (req.analysis_type or "").strip().lower() == "student_project_review":
        if chunk.category in _PROJECT_ALIGNED_CATEGORIES:
            bonus += 0.045
        elif chunk.category in _PROJECT_SECONDARY_CATEGORIES:
            bonus += 0.012
    else:
        if chunk.category in {"writing", "referencing", "critical_thinking"}:
            bonus += 0.025

    return bonus


def _mode_alignment_bonus(req: RetrievalQuery, chunk: RetrievedChunk) -> float:
    normalized_mode = (req.mode or "").strip().lower()
    if req.audience == "student":
        if normalized_mode == "code" or (req.analysis_type or "").strip().lower().endswith("project_review"):
            if chunk.category in _PROJECT_ALIGNED_CATEGORIES:
                return 0.03
            if chunk.category in _PROJECT_SECONDARY_CATEGORIES:
                return 0.008
            if chunk.category in {"writing", "referencing", "academic_integrity"}:
                return -0.03
            return 0.0
        return 0.02 if chunk.category in _ACADEMIC_ALIGNED_CATEGORIES else 0.0

    if normalized_mode == "project" or (req.analysis_type or "").strip().lower().endswith("project_review"):
        return 0.02 if chunk.category in _PROJECT_ALIGNED_CATEGORIES | _PROFESSOR_POLICY_CATEGORIES else 0.0
    return 0.02 if chunk.category in _PROFESSOR_POLICY_CATEGORIES else 0.0


def _version_freshness_bonus(chunk: RetrievedChunk) -> float:
    version = str((chunk.metadata or {}).get("version") or "").strip().lower()
    if not version:
        return 0.0
    match = re.search(r"(\d+)", version)
    if not match:
        return 0.0
    numeric = int(match.group(1))
    return min(numeric * 0.003, 0.018)


def _query_signal_text(req: RetrievalQuery) -> str:
    parts = [req.query]
    if req.title_hint:
        parts.append(req.title_hint)
    if req.keywords:
        parts.append(" ".join(req.keywords[:10]))
    if req.text_excerpt:
        parts.append(req.text_excerpt[:300])
    if req.mode:
        parts.append(req.mode)
    return " ".join(part for part in parts if part)


def rerank_score(req: RetrievalQuery, chunk: RetrievedChunk) -> float:
    meta_text = " ".join([chunk.document_title, chunk.section, chunk.category])
    lexical_score = hybrid_overlap_score(_query_signal_text(req), chunk.content, meta_text)

    preferred_category_bonus = _preferred_category_bonus(req, chunk)
    analysis_bonus = _analysis_term_bonus(req, chunk)
    audience_bonus = _audience_bonus(req, chunk)
    mode_alignment_bonus = _mode_alignment_bonus(req, chunk)
    version_bonus = _version_freshness_bonus(chunk)

    blended = (
        (chunk.score * settings.rag_semantic_weight)
        + (lexical_score * settings.rag_keyword_weight)
        + preferred_category_bonus
        + analysis_bonus
        + audience_bonus
        + mode_alignment_bonus
        + version_bonus
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
