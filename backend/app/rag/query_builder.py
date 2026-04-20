from __future__ import annotations

import re
from typing import Any

from app.core.config import settings
from app.rag.config import get_rag_runtime_config
from app.rag.schemas import RetrievalQuery
from app.rag.topic_detector import detect_topic
from app.rag.utils.text_cleaner import normalize_whitespace


_MAX_QUERY_CHARS = 320
_MAX_QUERY_COUNT = 7
_MAX_FOCUS_TERMS = 7
_MAX_FOCUS_CHARS = 80
_MAX_PRIMARY_QUERY_CHARS = 120

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
    "using",
    "use",
    "used",
    "review",
    "guidance",
    "feedback",
    "student",
    "professor",
    "submission",
    "about",
    "built",
    "build",
    "reviewing",
    "assess",
    "assessment",
    "based",
    "system",
    "platform",
    "project",
    "week",
    "weeks",
    "exercise",
    "exercises",
    "basic",
    "introduced",
    "introduces",
    "demonstrates",
    "demonstrate",
    "created",
    "various",
    "task",
    "tasks",
    "extended",
}

_LOW_VALUE_FOCUS_TERMS = {
    "academic",
    "api",
    "architecture",
    "auth",
    "authentication",
    "authorization",
    "backend",
    "based",
    "criterion",
    "data",
    "database",
    "dissertation",
    "discussion",
    "engineering",
    "evaluation",
    "evidence",
    "evaluation",
    "external",
    "feature",
    "features",
    "feedback",
    "flow",
    "frontend",
    "future",
    "generated",
    "guidance",
    "implementation",
    "improvements",
    "insight",
    "insights",
    "integration",
    "implementation",
    "justification",
    "level",
    "limitations",
    "marking",
    "management",
    "methodology",
    "metric",
    "metrics",
    "methodology",
    "moderation",
    "objectives",
    "policy",
    "project",
    "quality",
    "ratio",
    "report",
    "review",
    "rubric",
    "schema",
    "security",
    "software",
    "system",
    "student",
    "submission",
    "technical",
    "testing",
    "usability",
    "validation",
    "web",
    "web-based",
    "built",
    "build",
    "about",
    "assess",
    "assessment",
    "portfolio management system",
    "stock portfolio management system",
    "student software engineering project",
    "notes",
    "this",
    "week",
    "weeks",
    "exercise",
    "exercises",
    "basic",
    "introduced",
    "introduces",
    "demonstrates",
    "demonstrate",
    "created",
    "various",
    "task",
    "tasks",
    "extended",
}


_STUDENT_HINTS = {
    "citation": ["citation", "reference", "referencing", "harvard", "apa", "mla"],
    "structure": ["structure", "essay structure", "report structure", "introduction", "conclusion"],
    "clarity": ["clarity", "coherence", "academic writing", "paragraphing"],
    "integrity": ["academic integrity", "plagiarism", "original work"],
}

_STUDENT_PROJECT_HINTS = [
    "software project evaluation",
    "system architecture backend frontend database",
    "authentication security testing limitations",
    "implementation quality integration quality",
    "software engineering testing security maintainability code review",
]

_PROFESSOR_HINTS = {
    "rubric": ["rubric", "band descriptor", "criterion"],
    "policy": ["marking policy", "assessment policy", "grading guidance"],
    "moderation": ["moderation", "consistency", "second marking"],
    "feedback": ["feedback template", "feedback wording", "formative feedback"],
}

_PROFESSOR_PROJECT_HINTS = [
    "software project rubric architecture implementation",
    "backend frontend database security testing evaluation",
    "moderation consistency project dissertation marking",
    "technical quality integration quality limitations",
]

_QUERY_TERM_HINTS = {
    "student": {
        "django": ["django backend architecture", "server side implementation"],
        "react": ["frontend components interface usability", "react frontend implementation"],
        "next.js": ["frontend routing interface implementation"],
        "database": ["database schema data model persistence"],
        "sql": ["database schema queries persistence"],
        "portfolio": ["portfolio tracking analytics financial metrics"],
        "stock": ["stock data integration analytics refresh strategy"],
        "api": ["external api integration data flow"],
        "auth": ["authentication authorization security"],
        "security": ["authentication authorization security"],
        "testing": ["testing evaluation validation"],
        "ai": ["ai insights model limitations evaluation"],
        "machine learning": ["ai insights model limitations evaluation"],
        "c#": ["c# windows forms architecture event handling"],
        ".net": ["dotnet windows forms architecture maintainability"],
        "windows forms": ["windows forms event handling ui state validation"],
        "algorithm": ["algorithm correctness performance maintainability"],
    },
    "professor": {
        "project": ["project rubric technical implementation evaluation"],
        "django": ["backend implementation rubric project architecture"],
        "react": ["frontend implementation usability rubric"],
        "database": ["database design technical quality rubric"],
        "security": ["security access control evaluation rubric"],
        "testing": ["testing evaluation evidence moderation"],
        "api": ["integration quality external data evaluation"],
        "stock": ["domain-specific analytics technical quality"],
        "ai": ["ai feature limitations evaluation"],
    },
}


def _normalize(text: str) -> str:
    return normalize_whitespace(text)


def _clip_query(text: str, max_chars: int = _MAX_QUERY_CHARS) -> str:
    normalized = _normalize(text)
    if len(normalized) <= max_chars:
        return normalized

    clipped = normalized[: max_chars + 1]
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped.strip()


def _join_parts(*parts: str) -> str:
    joined = " ".join(_normalize(part) for part in parts if _normalize(part))
    if not joined:
        return ""

    seen: set[str] = set()
    deduped_words: list[str] = []

    for word in joined.split():
        normalized = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", word.lower())
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped_words.append(word)

    return _clip_query(" ".join(deduped_words))


def _query_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9.+#/-]*", text.lower())


def _all_known_terms(audience: str) -> list[str]:
    hints: list[str] = []
    mapping = _QUERY_TERM_HINTS[audience]

    for key in mapping:
        hints.append(key)
    for values in mapping.values():
        hints.extend(values)

    if audience == "student":
        for values in _STUDENT_HINTS.values():
            hints.extend(values)
    else:
        for values in _PROFESSOR_HINTS.values():
            hints.extend(values)

    return sorted({_normalize(term).lower() for term in hints if term}, key=len, reverse=True)


def _focus_terms(req: RetrievalQuery) -> list[str]:
    text = _normalize(req.query).lower()
    if not text:
        return []

    candidates: list[str] = []

    for term in _all_known_terms(req.audience):
        if term and term in text:
            term_tokens = set(_query_tokens(term))
            if term in _LOW_VALUE_FOCUS_TERMS:
                continue
            if term_tokens and term_tokens.issubset(_LOW_VALUE_FOCUS_TERMS):
                continue
            candidates.append(term)

    for token in _query_tokens(text):
        if token in _STOPWORDS:
            continue
        if token in _LOW_VALUE_FOCUS_TERMS:
            continue
        if len(token) < 2 and token != "ai":
            continue
        candidates.append(token)

    seen: set[str] = set()
    focus: list[str] = []
    current_len = 0

    for term in candidates:
        normalized = _normalize(term).lower()
        if not normalized or normalized in seen:
            continue
        extra_len = len(normalized) + (1 if focus else 0)
        if len(focus) >= _MAX_FOCUS_TERMS or current_len + extra_len > _MAX_FOCUS_CHARS:
            break
        seen.add(normalized)
        focus.append(normalized)
        current_len += extra_len

    return focus


def _focus_hint(req: RetrievalQuery) -> str:
    return " ".join(_focus_terms(req))


def _semantic_summary(req: RetrievalQuery, focus_hint: str) -> str:
    if focus_hint:
        return focus_hint

    audience_label = "student project" if req.audience == "student" else "project marking review"
    if _project_like(req):
        return audience_label
    return "student academic writing" if req.audience == "student" else "academic marking guidance"


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def _project_like(req: RetrievalQuery) -> bool:
    submission_type = _normalize(req.submission_type or "").lower()
    analysis_type = _normalize(req.analysis_type or "").lower()
    mode = _normalize(req.mode or "").lower()
    return (
        submission_type in {"project", "software_project", "dissertation_project", "capstone"}
        or "project" in analysis_type
        or mode in {"code", "project"}
    )


def _keyword_hint(req: RetrievalQuery) -> str:
    """Return a compact string of submission-extracted keywords for query anchoring."""
    if req.title_hint:
        title_terms = _focus_terms_from_text(req.title_hint, req.audience)
        if title_terms:
            return " ".join(title_terms[:6])
    if req.keywords:
        return " ".join(req.keywords[:8])
    if req.text_excerpt:
        terms = _focus_terms_from_text(req.text_excerpt, req.audience)
        return " ".join(terms[:8])
    return ""


def _mode_hint(req: RetrievalQuery) -> str:
    mode = _normalize(req.mode or "").lower()
    if not mode:
        if req.audience == "student":
            return "software project" if _project_like(req) else "academic writing"
        return "project rubric" if _project_like(req) else "rubric policy"
    if mode == "code":
        return "software project"
    if mode == "essay":
        return "academic writing"
    if mode == "report":
        return "academic report writing"
    if mode == "chapter":
        return "academic chapter writing"
    if mode == "mixed":
        return "submission critique"
    if mode == "feedback":
        return "feedback template moderation"
    if mode == "policy":
        return "marking policy moderation"
    if mode == "moderation":
        return "moderation rubric"
    if mode == "rubric":
        return "rubric criteria"
    return mode


def _primary_anchor_query(req: RetrievalQuery, focus_hint: str) -> str:
    runtime = get_rag_runtime_config()
    parts: list[str] = []
    title_terms = _focus_terms_from_text(req.title_hint or "", req.audience)
    if title_terms:
        parts.append(" ".join(title_terms[:4]))
    parts.append(_mode_hint(req))
    if focus_hint:
        parts.append(focus_hint)
    keyword_hint = _keyword_hint(req)
    if keyword_hint:
        parts.append(keyword_hint)
    if req.analysis_type:
        parts.append(_normalize(req.analysis_type).replace("_", " "))
    anchor = _join_parts(*parts)
    if anchor:
        return _clip_query(anchor, max_chars=min(runtime.query_excerpt_chars, _MAX_PRIMARY_QUERY_CHARS))
    return _clip_query(_join_parts(req.query, _mode_hint(req)), max_chars=_MAX_PRIMARY_QUERY_CHARS)


def _focus_terms_from_text(text: str, audience: str) -> list[str]:
    """Like _focus_terms but operates on a raw text excerpt instead of req.query."""
    lowered = _normalize(text).lower()[:2000]
    candidates: list[str] = []
    for term in _all_known_terms(audience):
        if term and term in lowered:
            term_tokens = set(_query_tokens(term))
            if term in _LOW_VALUE_FOCUS_TERMS:
                continue
            if term_tokens and term_tokens.issubset(_LOW_VALUE_FOCUS_TERMS):
                continue
            candidates.append(term)
    for token in _query_tokens(lowered):
        if token in _STOPWORDS or token in _LOW_VALUE_FOCUS_TERMS:
            continue
        if len(token) < 2 and token != "ai":
            continue
        candidates.append(token)
    seen: set[str] = set()
    focus: list[str] = []
    for term in candidates:
        norm = _normalize(term).lower()
        if norm and norm not in seen:
            seen.add(norm)
            focus.append(norm)
            if len(focus) >= _MAX_FOCUS_TERMS:
                break
    return focus


def _student_project_queries(req: RetrievalQuery, focus_hint: str) -> list[str]:
    finance_focus = _contains_any(
        req.query,
        ("stock", "portfolio", "finance", "financial", "trading", "sharpe", "dashboard", "analytics", "ai"),
    )
    # Also check the text excerpt for finance terms
    if not finance_focus and req.text_excerpt:
        finance_focus = _contains_any(
            req.text_excerpt,
            ("stock", "portfolio", "finance", "financial", "trading", "sharpe", "dashboard"),
        )
    kw_hint = _keyword_hint(req)
    implementation_summary = (
        "django stock portfolio management implementation transaction analytics dashboard"
        if finance_focus
        else _join_parts("software project implementation integration features", focus_hint)
    )

    return [
        _primary_anchor_query(req, focus_hint),
        "student software engineering architecture implementation integration",
        _join_parts(implementation_summary, kw_hint),
        "student project testing security validation limitations",
        _join_parts(
            "portfolio analytics stock dashboard financial metrics"
            if finance_focus
            else "technical quality correctness modularity error handling maintainability",
        ),
        "code review architecture testing security maintainability next steps",
    ]


def _professor_project_queries(req: RetrievalQuery, focus_hint: str) -> list[str]:
    finance_focus = _contains_any(
        req.query,
        ("stock", "portfolio", "finance", "financial", "trading", "sharpe", "dashboard", "analytics", "ai"),
    )
    implementation_summary = (
        "django stock portfolio management implementation analytics dashboard"
        if finance_focus
        else _join_parts("project technical implementation integration quality", focus_hint)
    )

    return [
        _primary_anchor_query(req, focus_hint),
        "project rubric moderation technical quality architecture implementation",
        _join_parts(
            "criterion justification backend frontend database testing",
            implementation_summary,
        ),
        "marking policy moderation consistency limitations evidence",
        _join_parts(
            "portfolio analytics metrics ai feature limitations"
            if finance_focus
            else "integration quality testing evidence usability limitations",
        ),
    ]


def _student_academic_queries(req: RetrievalQuery, focus_hint: str) -> list[str]:
    return [
        _primary_anchor_query(req, focus_hint),
        "student academic writing structure argument evidence",
        "critical analysis comparison reasoning methodology",
        "citation referencing attribution academic integrity",
        "clarity coherence paragraphing conclusion revision advice",
    ]


def _professor_academic_queries(req: RetrievalQuery, focus_hint: str) -> list[str]:
    return [
        _primary_anchor_query(req, focus_hint),
        "rubric marking policy moderation academic writing structure evidence",
        "criterion band descriptor critical analysis source use clarity",
        "feedback wording moderation safety consistency defensible judgement",
        "referencing academic quality methodology discussion limitations",
    ]


def _expand_from_submission_context(req: RetrievalQuery) -> list[str]:
    """Generate submission-specific query variants from text_excerpt and keywords.

    This is the primary mechanism for topic-specific retrieval: when the caller
    passes extracted submission keywords or a text excerpt, we build focused
    queries from them instead of (or in addition to) generic base queries.
    """
    if not req.keywords and not req.text_excerpt:
        return []

    extras: list[str] = []

    if req.title_hint:
        title_terms = _focus_terms_from_text(req.title_hint, req.audience)
        title_str = _join_parts(_mode_hint(req), " ".join(title_terms[:4]))
        if title_str:
            extras.append(title_str)

    # Keyword-focused query: most submission-specific signal available
    if req.keywords:
        kw_str = _join_parts(" ".join(req.keywords[:12]))
        if kw_str:
            extras.append(kw_str)

    # Excerpt-focused query: extract meaningful terms from the raw excerpt
    if req.text_excerpt:
        excerpt_terms = _focus_terms_from_text(req.text_excerpt, req.audience)
        if excerpt_terms:
            excerpt_str = _join_parts(" ".join(excerpt_terms[:8]))
            if excerpt_str:
                extras.append(excerpt_str)

    return extras


def _expand_from_ml_signals(audience: str, ml_signals: dict[str, Any]) -> list[str]:
    extras: list[str] = []
    data = {str(k).lower(): str(v).lower() for k, v in (ml_signals or {}).items()}

    if audience == "student":
        if any("citation" in v or "citation" in k for k, v in data.items()):
            extras.extend(_STUDENT_HINTS["citation"])
        if any("structure" in v or "structure" in k for k, v in data.items()):
            extras.extend(_STUDENT_HINTS["structure"])
        if any("clarity" in v or "clarity" in k for k, v in data.items()):
            extras.extend(_STUDENT_HINTS["clarity"])
        if any("integrity" in v or "integrity" in k for k, v in data.items()):
            extras.extend(_STUDENT_HINTS["integrity"])
    else:
        if any("rubric" in v or "band" in v or "rubric" in k for k, v in data.items()):
            extras.extend(_PROFESSOR_HINTS["rubric"])
        if any("policy" in v or "policy" in k for k, v in data.items()):
            extras.extend(_PROFESSOR_HINTS["policy"])
        if any("moderation" in v or "consistency" in v or "moderation" in k for k, v in data.items()):
            extras.extend(_PROFESSOR_HINTS["moderation"])
        if any("feedback" in v or "feedback" in k for k, v in data.items()):
            extras.extend(_PROFESSOR_HINTS["feedback"])

    seen = set()
    out: list[str] = []
    for x in extras:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _expand_from_submission_type(audience: str, submission_type: str | None) -> list[str]:
    kind = _normalize(submission_type or "").lower()
    if kind in {"project", "software_project", "dissertation_project", "capstone"}:
        return _STUDENT_PROJECT_HINTS if audience == "student" else _PROFESSOR_PROJECT_HINTS
    return []


def _expand_from_query_terms(audience: str, query: str) -> list[str]:
    haystack = query.lower()
    mapping = _QUERY_TERM_HINTS[audience]
    extras: list[str] = []

    for term, hints in mapping.items():
        if term in haystack:
            extras.extend(hints)

    seen = set()
    out: list[str] = []
    for x in extras:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _expand_from_detected_topic(req: RetrievalQuery) -> list[str]:
    if not settings.rag_enable_topic_detection:
        return []

    topic = detect_topic(req.query, req.audience)
    if not topic:
        return []

    source = _STUDENT_HINTS if req.audience == "student" else _PROFESSOR_HINTS
    extras = source.get(topic, [])

    seen = set()
    out: list[str] = []
    for x in extras:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _near_duplicate(candidate: str, existing: str) -> bool:
    candidate_norm = _normalize(candidate).lower()
    existing_norm = _normalize(existing).lower()

    if not candidate_norm or not existing_norm:
        return candidate_norm == existing_norm
    if candidate_norm == existing_norm:
        return True

    candidate_tokens = set(_query_tokens(candidate_norm))
    existing_tokens = set(_query_tokens(existing_norm))
    if not candidate_tokens or not existing_tokens:
        return False

    intersection = len(candidate_tokens & existing_tokens)
    union = len(candidate_tokens | existing_tokens)
    jaccard = intersection / max(1, union)
    return jaccard >= 0.94


def dedupe_queries(queries: list[str]) -> list[str]:
    deduped: list[str] = []

    for query in queries:
        clipped = _clip_query(query)
        if not clipped:
            continue
        if any(_near_duplicate(clipped, existing) for existing in deduped):
            continue
        deduped.append(clipped)

    return deduped


def build_queries(req: RetrievalQuery) -> list[str]:
    focus_hint = _focus_hint(req)

    if _project_like(req):
        queries = (
            _student_project_queries(req, focus_hint)
            if req.audience == "student"
            else _professor_project_queries(req, focus_hint)
        )
    else:
        queries = (
            _student_academic_queries(req, focus_hint)
            if req.audience == "student"
            else _professor_academic_queries(req, focus_hint)
        )

    for extra in _expand_from_submission_type(req.audience, req.submission_type):
        queries.append(_join_parts(extra, focus_hint))

    for extra in _expand_from_detected_topic(req):
        queries.append(_join_parts(extra, focus_hint))

    combined_hint_text = " ".join(
        part
        for part in [req.query, req.title_hint or "", req.text_excerpt or ""]
        if part
    )
    for extra in _expand_from_query_terms(req.audience, combined_hint_text):
        queries.append(_join_parts(extra, focus_hint))

    for extra in _expand_from_ml_signals(req.audience, req.ml_signals):
        queries.append(_join_parts(extra, focus_hint))

    # Submission-context expansion: prepend so submission-specific queries
    # survive the _MAX_QUERY_COUNT=7 cap before the generic base queries.
    submission_extras = _expand_from_submission_context(req)
    queries = submission_extras + queries

    return dedupe_queries(queries)[:_MAX_QUERY_COUNT]
