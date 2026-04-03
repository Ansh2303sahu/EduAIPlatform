from __future__ import annotations

from app.rag.rag_llm_pipeline import build_context as build_grounding_context
from app.rag.schemas import RetrievalFilters, RetrievalQuery
from app.rag.retrieval.student_retriever import retrieve_student_context
from app.rag.retrieval.professor_retriever import retrieve_professor_context


_STUDENT_ACADEMIC_CATEGORIES = [
    "writing",
    "critical_thinking",
    "research_support",
    "referencing",
    "academic_integrity",
]

_STUDENT_PROJECT_CATEGORIES = [
    "software_engineering",
    "project_evaluation",
    "research_support",
    "critical_thinking",
    "writing",
]

_PROFESSOR_ACADEMIC_CATEGORIES = [
    "rubrics",
    "marking_policy",
    "moderation",
    "academic_quality",
    "feedback_templates",
]

_PROFESSOR_PROJECT_CATEGORIES = [
    "rubrics",
    "marking_policy",
    "software_engineering",
    "moderation",
    "feedback_templates",
]
def _instruction(
    *,
    audience: str,
    confidence_label: str,
    safe_review: bool,
    analysis_type: str | None = None,
) -> str:
    base_caution = (
        "Use the retrieved sources cautiously. Evidence is limited. "
        "Do not invent academic rules, policies, or project details not grounded in the retrieved material."
    )

    if audience == "student":
        if analysis_type == "student_project_review":
            strong = (
                "Use the retrieved sources as grounding context for a student software/project review. "
                "Anchor feedback in project architecture, implementation quality, database design, security, "
                "testing, evaluation, limitations, and future improvements when the sources support those topics. "
                "Prefer specific evidence over generic advice."
            )
        else:
            strong = (
                "Use the retrieved sources as grounding context for academic student feedback. "
                "Anchor feedback in structure, evidence use, critical analysis, clarity, academic writing quality, "
                "and citation/referencing guidance where supported."
            )
    else:
        if analysis_type == "professor_project_review":
            strong = (
                "Use the retrieved sources as grounding context for professor-side project assessment. "
                "Prioritize rubric, moderation, architecture, implementation, security, testing, and technical quality guidance. "
                "Prefer official or policy-like sources when available and keep judgments moderation-safe."
            )
        else:
            strong = (
                "Use the retrieved sources as grounding context for rubric, marking policy, moderation, and feedback guidance. "
                "Prefer official or policy-like sources when available and do not go beyond the supplied evidence."
            )

    if safe_review or confidence_label == "low":
        return base_caution
    return strong


def _preferred_categories(audience: str, analysis_type: str | None) -> list[str]:
    normalized = (analysis_type or "").strip().lower()
    if audience == "student":
        if normalized == "student_project_review":
            return list(_STUDENT_PROJECT_CATEGORIES)
        return list(_STUDENT_ACADEMIC_CATEGORIES)

    if normalized == "professor_project_review":
        return list(_PROFESSOR_PROJECT_CATEGORIES)
    return list(_PROFESSOR_ACADEMIC_CATEGORIES)


def _official_bias(audience: str, analysis_type: str | None) -> float:
    normalized = (analysis_type or "").strip().lower()
    if audience == "professor":
        return 0.22 if normalized == "professor_project_review" else 0.28
    if normalized == "student_project_review":
        return 0.10
    return 0.06


def build_student_rag_payload(body: dict) -> dict:
    analysis_type = str(body.get("analysis_type") or "").strip().lower() or None
    _default_query = (
        "software project evaluation architecture implementation testing quality"
        if analysis_type == "student_project_review"
        else "academic writing structure critical analysis evidence referencing"
    )
    query = str(body.get("query") or body.get("prompt") or body.get("task") or _default_query)
    top_k = int(body.get("top_k") or 6)
    if analysis_type == "student_project_review":
        top_k = max(top_k, 8)
    category = body.get("category")
    version = body.get("version")
    submission_type = body.get("submission_type")
    ml_signals = body.get("ml") or {}

    result = retrieve_student_context(
        query=query,
        top_k=top_k,
        category=category,
        version=version,
        ml_signals=ml_signals,
        submission_type=submission_type,
        analysis_type=analysis_type,
        preferred_categories=_preferred_categories("student", analysis_type),
        official_bias=_official_bias("student", analysis_type),
    )

    return {
        "context": build_grounding_context(result.chunks),
        "citations": [c.model_dump() for c in result.citations],
        "retrieved_chunks": [c.model_dump() for c in result.chunks],
        "confidence_score": result.confidence_score,
        "confidence_label": result.confidence_label,
        "safe_review": result.safe_review,
        "trace": result.trace.model_dump(),
        "instruction": _instruction(
            audience="student",
            confidence_label=result.confidence_label,
            safe_review=result.safe_review,
            analysis_type=analysis_type,
        ),
    }


def build_professor_rag_payload(body: dict) -> dict:
    analysis_type = str(body.get("analysis_type") or "").strip().lower() or None
    _default_query = (
        "software project rubric marking criteria implementation assessment standards"
        if analysis_type == "professor_project_review"
        else "academic rubric marking policy moderation feedback guidance"
    )
    query = str(body.get("query") or body.get("prompt") or body.get("task") or _default_query)
    top_k = int(body.get("top_k") or 6)
    if analysis_type == "professor_project_review":
        top_k = max(top_k, 8)
    category = body.get("category")
    version = body.get("version")
    submission_type = body.get("submission_type")
    ml_signals = body.get("ml") or {}
    official_only = body.get("official_only")

    result = retrieve_professor_context(
        query=query,
        top_k=top_k,
        category=category,
        version=version,
        ml_signals=ml_signals,
        submission_type=submission_type,
        official_only=official_only,
        analysis_type=analysis_type,
        preferred_categories=_preferred_categories("professor", analysis_type),
        official_bias=_official_bias("professor", analysis_type),
    )

    return {
        "context": build_grounding_context(result.chunks),
        "citations": [c.model_dump() for c in result.citations],
        "retrieved_chunks": [c.model_dump() for c in result.chunks],
        "confidence_score": result.confidence_score,
        "confidence_label": result.confidence_label,
        "safe_review": result.safe_review,
        "trace": result.trace.model_dump(),
        "instruction": _instruction(
            audience="professor",
            confidence_label=result.confidence_label,
            safe_review=result.safe_review,
            analysis_type=analysis_type,
        ),
    }
