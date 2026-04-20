from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

from app.rag.config import get_rag_runtime_config
from app.rag.rag_llm_pipeline import build_context as build_grounding_context
from app.rag.retrieval.professor_retriever import retrieve_professor_context
from app.rag.retrieval.student_retriever import retrieve_student_context
from app.services import report_generation_support as report_support

logger = logging.getLogger("rag.context_builder")

_PROJECT_KW = {
    "django", "react", "next.js", "fastapi", "flask", "vue", "angular",
    "backend", "frontend", "database", "sql", "postgresql", "mysql", "mongodb",
    "api", "rest", "graphql", "auth", "authentication", "authorization",
    "docker", "kubernetes", "deployment", "cloud", "aws", "azure",
    "testing", "pytest", "unittest", "jest", "cypress",
    "analytics", "dashboard", "portfolio", "stock", "trading", "finance",
    "machine learning", "ai", "nlp", "algorithm", "chart.js", "sqlalchemy",
    "architecture", "microservices", "cache", "redis", "celery", "windows forms",
    "c#", ".net", "winforms", "github", "repository", "tailwindcss",
}

_ESSAY_KW = {
    "thesis", "argument", "literature", "reference", "references", "citation",
    "apa", "harvard", "mla", "bibliography", "essay", "conclusion",
    "critical analysis", "evidence", "framework", "methodology",
    "risk assessment", "cia triad", "nist", "stride", "gdpr", "policy",
    "evaluation", "discussion", "introduction", "abstract", "governance",
    "compliance", "incident response", "research question",
    "formative feedback", "higher education", "automated feedback", "pedagogical depth",
}

_PROFESSOR_KW = {
    "rubric", "rubrics", "marking policy", "assessment policy", "moderation",
    "feedback template", "feedback templates", "band descriptor", "criterion",
    "criteria", "learning outcome", "academic quality", "second marking",
    "consistency", "grading guidance", "comment bank", "mark scheme", "policy",
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+#/-]*")
_SURFACE_TERM_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9.+#/-]{2,}|[A-Za-z0-9.+#/-]+\.[A-Za-z0-9.+#/-]+|C#|\.NET)\b"
)

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "your", "their",
    "about", "have", "will", "when", "where", "were", "which", "should", "could",
    "would", "being", "been", "than", "then", "them", "they", "there", "because",
    "while", "does", "used", "using", "report", "student", "submission", "project",
    "essay", "review", "assessment", "analysis", "system", "platform", "quality",
    "week", "weeks", "exercise", "exercises", "basic", "introduced", "introduces",
    "demonstrates", "demonstrate", "created", "various", "task", "tasks", "extended",
}

_LOW_VALUE_TERMS = {
    "general", "guidance", "feedback", "structure", "evaluation", "discussion",
    "implementation", "methodology", "evidence", "technical", "academic",
    "this",
    "week", "weeks", "exercise", "exercises", "basic", "introduced", "introduces",
    "demonstrates", "demonstrate", "created", "various", "task", "tasks", "extended",
}

_GENERIC_QUERY_TERMS = {
    "student", "professor", "academic", "writing", "feedback", "report", "chapter",
    "essay", "submission", "review", "structure", "argument", "evidence",
    "referencing", "citation", "rubric", "policy", "moderation", "project",
    "architecture", "implementation", "testing", "security",
}

_STUDENT_MODE_HINTS = {
    "essay": "academic writing argument evidence",
    "report": "academic report structure evidence",
    "chapter": "introduction chapter academic argument",
    "code": "software project implementation review",
    "mixed": "submission critique grounded revision",
    "reflection": "reflective writing theory practice",
}

_PROFESSOR_MODE_HINTS_QUERY = {
    "essay": "academic writing rubric alignment",
    "report": "academic report rubric alignment",
    "chapter": "chapter structure rubric alignment",
    "project": "project rubric moderation",
    "feedback": "feedback template moderation",
    "policy": "marking policy moderation",
    "moderation": "moderation rubric",
    "rubric": "rubric criteria alignment",
}

_STUDENT_CODE_HINTS = (
    "function", "class", "method", "repository", "github", "database", "endpoint",
    "frontend", "backend", "api", "testing", "windows forms", "fastapi", "django",
    "react", "implementation", "c#", ".net", "sql", "module", "codebase",
)
_STUDENT_ESSAY_HINTS = (
    "thesis", "argument", "citation", "references", "critical analysis", "paragraph",
    "literature", "framework", "policy", "risk assessment", "methodology",
    "conclusion", "governance", "discussion",
)

_PROFESSOR_MODE_HINTS = {
    "feedback": ("feedback template", "comment bank", "feedback wording"),
    "policy": ("marking policy", "assessment policy", "grading guidance"),
    "rubric": ("rubric", "criterion", "criteria", "band descriptor"),
    "moderation": ("moderation", "second marking", "consistency"),
}

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


def _normalize_space(text: str | None) -> str:
    return " ".join((text or "").split())


def _extract_submission_title(text: str | None, *, fallback_query: str | None = None) -> str | None:
    runtime = get_rag_runtime_config()
    normalized = _normalize_space(text)
    if normalized:
        head = normalized[: runtime.query_excerpt_chars]
        segments = [
            segment.strip(" :-|")
            for segment in re.split(r"(?:[.!?](?=\s|$)|[\n\r]+)", head)
            if segment.strip()
        ]
        for segment in segments[:4]:
            if len(segment) < 12:
                continue
            if segment.lower().startswith(("introduction", "abstract", "table of contents")):
                continue
            return segment[: runtime.query_title_chars]
    fallback = _normalize_space(fallback_query)[: runtime.query_title_chars]
    return fallback or None


def _submission_text_excerpt(text: str | None, limit: int | None = None) -> str | None:
    """Return a clean, compact excerpt of the submission text."""
    if not text:
        return None
    runtime = get_rag_runtime_config()
    excerpt_limit = limit or runtime.query_excerpt_chars
    compact = _normalize_space(text)
    return compact[:excerpt_limit] or None


def _candidate_tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if len(token) >= 3 and token.lower() not in _STOPWORDS
    ]


def _dynamic_keyword_candidates(text: str, *, title_hint: str | None = None) -> Counter[str]:
    normalized = _normalize_space(text)
    lowered = normalized.lower()[:8000]
    counts: Counter[str] = Counter()
    tokens = _candidate_tokens(lowered)

    for token in tokens:
        if token in _LOW_VALUE_TERMS:
            continue
        counts[token] += 1

    for size in (3, 2):
        for idx in range(0, max(0, len(tokens) - size + 1)):
            phrase_tokens = tokens[idx : idx + size]
            if any(token in _LOW_VALUE_TERMS for token in phrase_tokens):
                continue
            phrase = " ".join(phrase_tokens)
            counts[phrase] += 1

    for match in _SURFACE_TERM_RE.findall(normalized[:4000]):
        surface = match.strip().lower()
        if surface and surface not in _LOW_VALUE_TERMS:
            counts[surface] += 2

    title_lower = (title_hint or "").lower()
    if title_lower:
        for token in _candidate_tokens(title_lower):
            counts[token] += 2
        for size in (3, 2):
            title_tokens = _candidate_tokens(title_lower)
            for idx in range(0, max(0, len(title_tokens) - size + 1)):
                phrase = " ".join(title_tokens[idx : idx + size])
                if phrase and phrase not in _LOW_VALUE_TERMS:
                    counts[phrase] += 2

    return counts


def _extract_submission_keywords(
    text: str | None,
    analysis_type: str | None,
    *,
    audience: str = "student",
    title_hint: str | None = None,
    mode: str | None = None,
) -> list[str]:
    """Return submission-specific keywords using curated + lightweight dynamic signals."""
    if not text:
        return []

    runtime = get_rag_runtime_config()
    lowered = _normalize_space(text).lower()[:8000]
    normalized_analysis = (analysis_type or "").strip().lower()
    normalized_mode = (mode or "").strip().lower()

    base_pool = set(_PROFESSOR_KW) if audience == "professor" else (
        set(_PROJECT_KW) if normalized_analysis.endswith("project_review") or normalized_mode == "code" else set(_ESSAY_KW)
    )

    scores: dict[str, int] = {}

    def bump(term: str, amount: int) -> None:
        key = term.strip().lower()
        if not key or key in _LOW_VALUE_TERMS or len(key) < 3:
            return
        scores[key] = scores.get(key, 0) + amount

    for kw in sorted(base_pool, key=len, reverse=True):
        if kw in lowered:
            bump(kw, 6 if " " in kw else 4)

    dynamic_counts = _dynamic_keyword_candidates(lowered, title_hint=title_hint)
    title_lower = (title_hint or "").lower()
    for term, count in dynamic_counts.items():
        if term in _LOW_VALUE_TERMS:
            continue
        if len(term.split()) >= 2 and count < 2 and term not in title_lower:
            continue
        if len(term.split()) == 1 and count < 2 and term not in base_pool and term not in title_lower:
            continue
        bump(term, min(count, 5) + (2 if term in title_lower else 0))

    if audience == "student":
        mode_terms = _STUDENT_CODE_HINTS if normalized_mode == "code" else _STUDENT_ESSAY_HINTS
        for term in mode_terms:
            if term in lowered or term in title_lower:
                bump(term, 2)
    else:
        for mode_name, hints in _PROFESSOR_MODE_HINTS.items():
            if normalized_mode != mode_name:
                continue
            for hint in hints:
                if hint in lowered or hint in title_lower:
                    bump(hint, 2)

    ranked = sorted(
        scores.items(),
        key=lambda item: (-item[1], -(len(item[0].split())), -len(item[0]), item[0]),
    )
    return [term for term, _score in ranked[: runtime.dynamic_keyword_limit]]


def _student_feedback_mode(
    *,
    analysis_type: str | None,
    submission_type: str | None,
    text: str,
    query: str,
    explicit_mode: str | None = None,
) -> str:
    normalized_analysis = (analysis_type or "").strip().lower()
    normalized_submission = (submission_type or "").strip().lower()
    normalized_mode = (explicit_mode or "").strip().lower()
    if normalized_mode in {"essay", "report", "chapter", "code", "mixed", "reflection", "generic"}:
        return normalized_mode
    if normalized_analysis.endswith("project_review") or normalized_submission in {
        "project",
        "software_project",
        "capstone",
        "dissertation_project",
    }:
        return "code"

    haystack = f"{query} {text}".lower()
    if any(token in haystack for token in ("reflection", "reflective", "placement", "self-awareness", "self awareness")):
        return "reflection"
    form = report_support.classify_submission_form({"text_content": text})
    if form in {"essay", "report", "chapter", "code", "mixed"}:
        return form
    code_hits = sum(1 for term in _STUDENT_CODE_HINTS if term in haystack)
    essay_hits = sum(1 for term in _STUDENT_ESSAY_HINTS if term in haystack)
    return "code" if code_hits >= essay_hits + 2 else "essay"


def _professor_feedback_mode(
    *,
    analysis_type: str | None,
    submission_type: str | None,
    text: str,
    query: str,
    explicit_mode: str | None = None,
) -> str:
    normalized_analysis = (analysis_type or "").strip().lower()
    normalized_submission = (submission_type or "").strip().lower()
    normalized_mode = (explicit_mode or "").strip().lower()
    if normalized_mode in {"essay", "report", "chapter", "project", "feedback", "policy", "moderation", "rubric"}:
        return normalized_mode
    if normalized_analysis.endswith("project_review") or normalized_submission in {
        "project",
        "software_project",
        "capstone",
        "dissertation_project",
    }:
        return "project"

    haystack = f"{query} {text}".lower()
    for mode_name in ("feedback", "policy", "moderation", "rubric"):
        if any(hint in haystack for hint in _PROFESSOR_MODE_HINTS[mode_name]):
            return mode_name
    return "rubric"


def _retrieval_source_text(body: dict[str, Any]) -> tuple[str, bool]:
    ingestion = body.get("ingestion") or {}
    ingestion_parts = [
        ingestion.get("text_content"),
        ingestion.get("ocr_text"),
        ingestion.get("audio_transcript"),
        body.get("text_content"),
    ]
    ingestion_text = "\n".join(_normalize_space(part) for part in ingestion_parts if _normalize_space(part))
    if ingestion_text:
        return ingestion_text, False

    fallback_parts: list[str] = []
    for field in ("query", "prompt", "task"):
        value = _normalize_space(body.get(field))
        if value:
            fallback_parts.append(value)
    for field in ("analysis_focus", "notes", "instructions"):
        value = body.get(field)
        if isinstance(value, list):
            fallback_parts.extend(_normalize_space(str(item)) for item in value if _normalize_space(str(item)))
        else:
            normalized = _normalize_space(str(value or ""))
            if normalized:
                fallback_parts.append(normalized)

    fallback_text = " ".join(part for part in fallback_parts if part).strip()
    return fallback_text, bool(fallback_text)


def _query_term_set(text: str | None) -> set[str]:
    return {
        token
        for token in _candidate_tokens(_normalize_space(text).lower())
        if token not in _LOW_VALUE_TERMS
    }


def _ranked_focus_terms(text: str | None, *, title_hint: str | None = None, limit: int = 6) -> list[str]:
    if not text:
        return []
    ranked: list[str] = []
    for term, _count in _dynamic_keyword_candidates(text, title_hint=title_hint).most_common():
        if term in _LOW_VALUE_TERMS or len(term) < 3:
            continue
        ranked.append(term)
        if len(ranked) >= limit:
            break
    return ranked


def _anchor_phrases(
    *,
    title_hint: str | None,
    keywords: list[str],
    text_excerpt: str | None,
) -> list[str]:
    anchors: list[str] = []
    for phrase in keywords[:8]:
        normalized = _normalize_space(phrase)
        if normalized:
            anchors.append(normalized)
    anchors.extend(_ranked_focus_terms(title_hint, title_hint=title_hint, limit=4))
    anchors.extend(_ranked_focus_terms(text_excerpt, title_hint=title_hint, limit=4))

    deduped: list[str] = []
    seen: set[str] = set()
    for phrase in anchors:
        lowered = phrase.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(phrase)
    return deduped[:8]


def _anchor_token_set(
    *,
    title_hint: str | None,
    keywords: list[str],
    text_excerpt: str | None,
) -> set[str]:
    tokens: set[str] = set()
    for phrase in _anchor_phrases(title_hint=title_hint, keywords=keywords, text_excerpt=text_excerpt):
        tokens.update(_query_term_set(phrase))
    return {
        token
        for token in tokens
        if token not in _GENERIC_QUERY_TERMS
    }


def _query_is_submission_grounded(
    query: str | None,
    *,
    title_hint: str | None,
    keywords: list[str],
    text_excerpt: str | None,
) -> bool:
    normalized = _normalize_space(query).lower()
    if not normalized:
        return False
    query_tokens = _query_term_set(normalized)
    if not query_tokens:
        return False
    anchor_tokens = _anchor_token_set(
        title_hint=title_hint,
        keywords=keywords,
        text_excerpt=text_excerpt,
    )
    overlap = len(query_tokens & anchor_tokens)
    generic_ratio = (
        sum(1 for token in query_tokens if token in _GENERIC_QUERY_TERMS) / max(1, len(query_tokens))
    )
    if overlap >= 2:
        return True
    if overlap >= 1 and generic_ratio < 0.6:
        return True
    return False


def _build_submission_query(
    *,
    audience: str,
    mode: str,
    title_hint: str | None,
    keywords: list[str],
    text_excerpt: str | None,
    fallback_query: str,
) -> str:
    anchors = _anchor_phrases(
        title_hint=title_hint,
        keywords=keywords,
        text_excerpt=text_excerpt,
    )
    mode_hint = (
        _STUDENT_MODE_HINTS.get(mode, "submission critique")
        if audience == "student"
        else _PROFESSOR_MODE_HINTS_QUERY.get(mode, "rubric alignment")
    )
    parts = anchors[:6]
    if mode_hint:
        parts.append(mode_hint)
    query = _normalize_space(" ".join(part for part in parts if part))
    if query:
        return query[:240]
    return _normalize_space(fallback_query)[:240]


def _select_retrieval_query(
    *,
    audience: str,
    mode: str,
    provided_query: str,
    fallback_query: str,
    title_hint: str | None,
    keywords: list[str],
    text_excerpt: str | None,
) -> str:
    if _query_is_submission_grounded(
        provided_query,
        title_hint=title_hint,
        keywords=keywords,
        text_excerpt=text_excerpt,
    ):
        return _normalize_space(provided_query)[:240]
    return _build_submission_query(
        audience=audience,
        mode=mode,
        title_hint=title_hint,
        keywords=keywords,
        text_excerpt=text_excerpt,
        fallback_query=fallback_query or provided_query,
    )


def _chunk_haystack(chunk: Any) -> str:
    return _normalize_space(
        " ".join(
            [
                str(getattr(chunk, "document_title", "") or ""),
                str(getattr(chunk, "section", "") or ""),
                str(getattr(chunk, "category", "") or ""),
                str(getattr(chunk, "content", "") or ""),
            ]
        )
    ).lower()


def _apply_grounding_gate(
    *,
    chunks: list[Any],
    citations: list[Any],
    confidence_score: float,
    confidence_label: str,
    safe_review: bool,
    title_hint: str | None,
    keywords: list[str],
    text_excerpt: str | None,
) -> tuple[list[Any], list[Any], str, float, str, bool, dict[str, Any]]:
    anchor_phrases = _anchor_phrases(
        title_hint=title_hint,
        keywords=keywords,
        text_excerpt=text_excerpt,
    )
    anchor_tokens = _anchor_token_set(
        title_hint=title_hint,
        keywords=keywords,
        text_excerpt=text_excerpt,
    )
    aligned_count = 0
    for chunk in chunks:
        haystack = _chunk_haystack(chunk)
        phrase_hits = sum(
            1
            for phrase in anchor_phrases
            if len(phrase.split()) >= 2 and phrase.lower() in haystack
        )
        token_hits = len(_query_term_set(haystack) & anchor_tokens)
        if phrase_hits >= 1 or token_hits >= 2:
            aligned_count += 1

    top_score = max((float(getattr(chunk, "score", 0.0) or 0.0) for chunk in chunks), default=0.0)
    rejection_reasons: list[str] = []
    if safe_review:
        rejection_reasons.append("safe_review")
    if not chunks:
        rejection_reasons.append("no_chunks")
    if top_score < 0.36:
        rejection_reasons.append("top_chunk_score_low")
    if anchor_tokens and aligned_count < 2:
        rejection_reasons.append("insufficient_topic_alignment")
    if str(confidence_label or "").strip().lower() == "low":
        rejection_reasons.append("low_confidence")

    reject_grounding = bool(rejection_reasons)
    trace_meta = {
        "grounding_rejected": reject_grounding,
        "grounding_rejection_reasons": rejection_reasons,
        "submission_alignment_count": aligned_count,
        "submission_alignment_required": bool(anchor_tokens),
        "submission_alignment_terms": anchor_phrases[:6],
        "top_chunk_score": round(top_score, 3),
        "approved_chunk_count": 0 if reject_grounding else len(chunks),
        "rejected_chunk_count": len(chunks) if reject_grounding else 0,
    }
    if reject_grounding:
        return [], [], "", min(float(confidence_score or 0.0), 0.2), "low", True, trace_meta
    return (
        chunks,
        citations,
        build_grounding_context(chunks),
        float(confidence_score or 0.0),
        str(confidence_label or "low"),
        bool(safe_review),
        trace_meta,
    )


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
                "Use the retrieved sources only as secondary grounding for academic student feedback. "
                "Start with the submission itself: paragraph clarity, claim specificity, evidence integration, "
                "citation consistency, and argument flow. "
                "Use retrieved material only to support referencing rules, structure expectations, or rubric alignment "
                "when the retrieved material clearly matches the submission topic."
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


def _preferred_categories(audience: str, analysis_type: str | None, mode: str | None) -> list[str]:
    normalized = (analysis_type or "").strip().lower()
    normalized_mode = (mode or "").strip().lower()
    if audience == "student":
        if normalized == "student_project_review" or normalized_mode == "code":
            return list(_STUDENT_PROJECT_CATEGORIES)
        return list(_STUDENT_ACADEMIC_CATEGORIES)

    if normalized == "professor_project_review" or normalized_mode == "project":
        return list(_PROFESSOR_PROJECT_CATEGORIES)
    if normalized_mode == "feedback":
        return ["feedback_templates", "moderation", "academic_quality", "rubrics", "marking_policy"]
    if normalized_mode == "policy":
        return ["marking_policy", "moderation", "rubrics", "feedback_templates", "academic_quality"]
    if normalized_mode == "moderation":
        return ["moderation", "marking_policy", "rubrics", "feedback_templates", "academic_quality"]
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
    submission_type = body.get("submission_type")
    raw_text, degraded_input = _retrieval_source_text(body)
    explicit_mode = str(body.get("submission_form") or body.get("mode") or "").strip().lower() or None
    query_fallback = (
        "software project evaluation architecture implementation testing quality"
        if analysis_type == "student_project_review"
        else "academic writing structure critical analysis evidence referencing"
    )
    provided_query = str(body.get("query") or body.get("prompt") or body.get("task") or "")
    mode = _student_feedback_mode(
        analysis_type=analysis_type,
        submission_type=submission_type,
        text=raw_text,
        query=provided_query or query_fallback,
        explicit_mode=explicit_mode,
    )
    title_hint = _extract_submission_title(raw_text, fallback_query=provided_query or query_fallback)
    keywords = _extract_submission_keywords(
        raw_text,
        analysis_type,
        audience="student",
        title_hint=title_hint,
        mode=mode,
    )
    text_excerpt = _submission_text_excerpt(raw_text)
    query = _select_retrieval_query(
        audience="student",
        mode=mode,
        provided_query=provided_query,
        fallback_query=query_fallback,
        title_hint=title_hint,
        keywords=keywords,
        text_excerpt=text_excerpt,
    )
    top_k = int(body.get("top_k") or 6)
    if analysis_type == "student_project_review" or mode == "code":
        top_k = max(top_k, 8)
    category = body.get("category")
    version = body.get("version")
    ml_signals = body.get("ml") or {}

    logger.info(
        "rag.context_builder student mode=%s analysis_type=%s degraded_input=%s title=%r excerpt_len=%s keywords=%s",
        mode,
        analysis_type or "unset",
        degraded_input,
        title_hint,
        len(text_excerpt or ""),
        keywords[:8],
    )

    result = retrieve_student_context(
        query=query,
        top_k=top_k,
        category=category,
        version=version,
        ml_signals=ml_signals,
        submission_type=submission_type,
        analysis_type=analysis_type,
        preferred_categories=_preferred_categories("student", analysis_type, mode),
        official_bias=_official_bias("student", analysis_type),
        text_excerpt=text_excerpt,
        keywords=keywords,
        title_hint=title_hint,
        mode=mode,
        degraded_input=degraded_input,
    )
    approved_chunks, approved_citations, approved_context, approved_confidence, approved_label, approved_safe_review, trace_meta = _apply_grounding_gate(
        chunks=list(result.chunks),
        citations=list(result.citations),
        confidence_score=result.confidence_score,
        confidence_label=result.confidence_label,
        safe_review=result.safe_review,
        title_hint=title_hint,
        keywords=keywords,
        text_excerpt=text_excerpt,
    )
    trace_payload = result.trace.model_dump()
    trace_payload.update(trace_meta)

    return {
        "context": approved_context,
        "citations": [c.model_dump() for c in approved_citations],
        "retrieved_chunks": [c.model_dump() for c in approved_chunks],
        "confidence_score": approved_confidence,
        "confidence_label": approved_label,
        "safe_review": approved_safe_review,
        "trace": trace_payload,
        "instruction": _instruction(
            audience="student",
            confidence_label=approved_label,
            safe_review=approved_safe_review,
            analysis_type=analysis_type,
        ),
    }


def build_professor_rag_payload(body: dict) -> dict:
    analysis_type = str(body.get("analysis_type") or "").strip().lower() or None
    submission_type = body.get("submission_type")
    raw_text, degraded_input = _retrieval_source_text(body)
    explicit_mode = str(body.get("submission_form") or body.get("mode") or "").strip().lower() or None
    query_fallback = (
        "software project rubric marking criteria implementation assessment standards"
        if analysis_type == "professor_project_review"
        else "academic rubric marking policy moderation feedback guidance"
    )
    provided_query = str(body.get("query") or body.get("prompt") or body.get("task") or "")
    mode = _professor_feedback_mode(
        analysis_type=analysis_type,
        submission_type=submission_type,
        text=raw_text,
        query=provided_query or query_fallback,
        explicit_mode=explicit_mode,
    )
    title_hint = _extract_submission_title(raw_text, fallback_query=provided_query or query_fallback)
    keywords = _extract_submission_keywords(
        raw_text,
        analysis_type,
        audience="professor",
        title_hint=title_hint,
        mode=mode,
    )
    text_excerpt = _submission_text_excerpt(raw_text)
    query = _select_retrieval_query(
        audience="professor",
        mode=mode,
        provided_query=provided_query,
        fallback_query=query_fallback,
        title_hint=title_hint,
        keywords=keywords,
        text_excerpt=text_excerpt,
    )
    top_k = int(body.get("top_k") or 6)
    if analysis_type == "professor_project_review" or mode == "project":
        top_k = max(top_k, 8)
    category = body.get("category")
    version = body.get("version")
    ml_signals = body.get("ml") or {}
    official_only = body.get("official_only")

    logger.info(
        "rag.context_builder professor mode=%s analysis_type=%s degraded_input=%s title=%r excerpt_len=%s keywords=%s",
        mode,
        analysis_type or "unset",
        degraded_input,
        title_hint,
        len(text_excerpt or ""),
        keywords[:8],
    )

    result = retrieve_professor_context(
        query=query,
        top_k=top_k,
        category=category,
        version=version,
        ml_signals=ml_signals,
        submission_type=submission_type,
        official_only=official_only,
        analysis_type=analysis_type,
        preferred_categories=_preferred_categories("professor", analysis_type, mode),
        official_bias=_official_bias("professor", analysis_type),
        text_excerpt=text_excerpt,
        keywords=keywords,
        title_hint=title_hint,
        mode=mode,
        degraded_input=degraded_input,
    )
    approved_chunks, approved_citations, approved_context, approved_confidence, approved_label, approved_safe_review, trace_meta = _apply_grounding_gate(
        chunks=list(result.chunks),
        citations=list(result.citations),
        confidence_score=result.confidence_score,
        confidence_label=result.confidence_label,
        safe_review=result.safe_review,
        title_hint=title_hint,
        keywords=keywords,
        text_excerpt=text_excerpt,
    )
    trace_payload = result.trace.model_dump()
    trace_payload.update(trace_meta)

    return {
        "context": approved_context,
        "citations": [c.model_dump() for c in approved_citations],
        "retrieved_chunks": [c.model_dump() for c in approved_chunks],
        "confidence_score": approved_confidence,
        "confidence_label": approved_label,
        "safe_review": approved_safe_review,
        "trace": trace_payload,
        "instruction": _instruction(
            audience="professor",
            confidence_label=approved_label,
            safe_review=approved_safe_review,
            analysis_type=analysis_type,
        ),
    }
