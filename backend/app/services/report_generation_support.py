from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from typing import Any, Dict, Literal

import httpx
from fastapi import HTTPException
from pydantic import BaseModel, Field, ValidationError

from app.core.config import settings
from app.core.deps import CurrentUser
from app.langchain.parsers.normalizer import (
    normalize_professor_payload as _phase10_normalize_professor_payload,
    normalize_student_payload as _phase10_normalize_student_payload,
)
from app.services.uuid_normalization import (
    normalize_uuid_insert_payload as _normalize_uuid_insert_payload,
    uuid_or_none as _uuid_or_none,
)

Severity = Literal["low", "med", "high"]


class _StoredIssue(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    evidence: str = Field(min_length=1, max_length=2000)
    severity: Severity


class _StoredStrength(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    evidence: str = Field(min_length=1, max_length=2000)


class _StoredImprovementAction(BaseModel):
    action: str = Field(min_length=1, max_length=300)
    why: str = Field(min_length=1, max_length=800)
    how: str = Field(min_length=1, max_length=800)
    priority: int = Field(ge=1, le=10)


class _StoredChecklistItem(BaseModel):
    item: str = Field(min_length=1, max_length=200)
    done: bool = False


class _StoredArchitectureReview(BaseModel):
    overview: str = Field(default="Not assessed.", min_length=1, max_length=1200)
    backend: str = Field(default="Not assessed.", min_length=1, max_length=1200)
    frontend: str = Field(default="Not assessed.", min_length=1, max_length=1200)
    database: str = Field(default="Not assessed.", min_length=1, max_length=1200)
    security: str = Field(default="Not assessed.", min_length=1, max_length=1200)


class _StoredImplementationReview(BaseModel):
    features_built: list[str] = Field(default_factory=list)
    technical_quality: str = Field(default="Not assessed.", min_length=1, max_length=1200)
    integration_quality: str = Field(default="Not assessed.", min_length=1, max_length=1200)


class _StoredEvaluationReview(BaseModel):
    testing_present: str = Field(default="Not assessed.", min_length=1, max_length=1200)
    limitations: str = Field(default="Not assessed.", min_length=1, max_length=1200)
    academic_quality: str = Field(default="Not assessed.", min_length=1, max_length=1200)


class _StoredConfidenceSummary(BaseModel):
    mode: Literal["normal", "restricted"] = "normal"
    overall: float = Field(default=0.0, ge=0.0, le=1.0)


class _StoredModelAgreement(BaseModel):
    ml_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    llm_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    final_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class _StoredSafety(BaseModel):
    needs_review: bool = False
    reason: str = ""


class _StoredStudentReport(BaseModel):
    summary: str = Field(min_length=1, max_length=1200)
    issues: list[_StoredIssue] = Field(default_factory=list)
    strengths: list[_StoredStrength] = Field(default_factory=list)
    architecture_review: _StoredArchitectureReview = Field(default_factory=_StoredArchitectureReview)
    implementation_review: _StoredImplementationReview = Field(default_factory=_StoredImplementationReview)
    evaluation_review: _StoredEvaluationReview = Field(default_factory=_StoredEvaluationReview)
    improvement_plan: list[_StoredImprovementAction] = Field(default_factory=list)
    checklist: list[_StoredChecklistItem] = Field(default_factory=list)
    confidence: _StoredConfidenceSummary = Field(default_factory=_StoredConfidenceSummary)
    model_agreement: _StoredModelAgreement = Field(default_factory=_StoredModelAgreement)
    safety: _StoredSafety = Field(default_factory=_StoredSafety)


class _StoredRubricRow(BaseModel):
    criterion: str = Field(min_length=1, max_length=200)
    band: str = Field(min_length=1, max_length=80)
    justification: str = Field(min_length=1, max_length=1200)


class _StoredModerationNote(BaseModel):
    risk: str = Field(min_length=1, max_length=120)
    note: str = Field(min_length=1, max_length=800)


class _StoredProfessorReport(BaseModel):
    rubric_breakdown: list[_StoredRubricRow] = Field(default_factory=list)
    feedback_explanation: str = Field(min_length=1, max_length=1600)
    moderation_notes: list[_StoredModerationNote] = Field(default_factory=list)
    safety: _StoredSafety = Field(default_factory=_StoredSafety)


SUPABASE_TIMEOUT = httpx.Timeout(
    connect=15.0,
    read=120.0,
    write=30.0,
    pool=30.0,
)

AI_TIMEOUT = httpx.Timeout(
    connect=15.0,
    read=120.0,
    write=30.0,
    pool=30.0,
)

_RATE: Dict[str, list[float]] = {}
_RATE_MAX = 15
_RATE_WINDOW = 3600.0

_PROJECT_HINT_TERMS = {
    "django",
    "react",
    "next.js",
    "frontend",
    "backend",
    "database",
    "api",
    "authentication",
    "authorization",
    "dashboard",
    "portfolio",
    "stock",
    "analytics",
    "implementation",
    "system design",
    "web application",
    "software project",
    "architecture",
    "testing",
}

_TECHNICAL_EXERCISE_HINT_TERMS = {
    "winforms",
    "system.drawing",
    "c#",
    ".net",
    "recursion",
    "recursive",
    "midpoint",
    "computer graphics",
    "graphics",
    "geometric subdivision",
    "coordinate alignment",
}

_ACADEMIC_HEADING_TERMS = {
    "introduction",
    "background and context",
    "background",
    "literature review",
    "methodology",
    "methods",
    "discussion",
    "analysis",
    "conclusion",
    "references",
    "bibliography",
}

_CHAPTER_HEADING_TERMS = {
    "chapter",
    "background and context",
    "literature review",
    "research questions",
    "methodology",
    "discussion",
    "conclusion",
}

_REPORT_HEADING_TERMS = {
    "executive summary",
    "findings",
    "recommendations",
    "limitations",
    "results",
    "report",
}

_CODE_LINE_RE = re.compile(
    r"(?im)^\s*(?:```|def\s+\w+\(|class\s+\w+[\(:]|function\s+\w+\(|public\s+(?:class|static|void)|"
    r"private\s+\w+|protected\s+\w+|import\s+\w|from\s+\w+\s+import|const\s+\w+\s*=|let\s+\w+\s*=|"
    r"var\s+\w+\s*=|return\s+.+;|if\s*\(|for\s*\(|while\s*\(|SELECT\s+.+\s+FROM\s+.+)\s*$"
)
_CITATION_PATTERNS = (
    re.compile(r"\([A-Z][A-Za-z'`-]+(?:\s+et al\.)?,\s*(?:19|20)\d{2}[a-z]?\)"),
    re.compile(r"\[[0-9]{1,3}\]"),
    re.compile(r"\bet al\.\b", re.IGNORECASE),
    re.compile(r"\bdoi:\s*10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE),
)
_PROSE_SENTENCE_RE = re.compile(r"[A-Z][^.!?]{50,}[.!?]")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'/-]{2,}")


def _clean_base_url(value: str | None) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        raw = raw[1:-1].strip()
    raw = raw.rstrip("/")
    if raw.endswith("/rest/v1"):
        raw = raw[:-8].rstrip("/")
    return raw


def _require_supabase_config() -> None:
    if not _clean_base_url(settings.supabase_url):
        raise HTTPException(status_code=500, detail="SUPABASE_URL not configured")
    if not settings.supabase_service_role_key:
        raise HTTPException(status_code=500, detail="SUPABASE_SERVICE_ROLE_KEY not configured")


def _supabase_base() -> str:
    _require_supabase_config()
    return _clean_base_url(settings.supabase_url)


def _service_headers(*, prefer_return: bool = False) -> Dict[str, str]:
    key = settings.supabase_service_role_key
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation" if prefer_return else "return=minimal",
    }


async def _supabase_get(path: str, *, retries: int = 3) -> httpx.Response:
    url = f"{_supabase_base()}/rest/v1/{path}"
    last_exc: Exception | None = None
    backoff = 0.6

    for attempt in range(1, retries + 1):
        try:
            async with httpx.AsyncClient(timeout=SUPABASE_TIMEOUT) as client:
                return await client.get(url, headers=_service_headers(prefer_return=False))
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
            last_exc = exc
            if attempt < retries:
                await asyncio.sleep(backoff)
                backoff *= 2

    raise HTTPException(
        status_code=504,
        detail=(
            "Supabase GET timeout/error after retries: "
            f"path={path} error={type(last_exc).__name__}: {last_exc}"
        ),
    )


async def _supabase_post(table: str, payload: dict[str, Any], *, retries: int = 3) -> httpx.Response:
    url = f"{_supabase_base()}/rest/v1/{table}"
    last_exc: Exception | None = None
    backoff = 0.6

    for attempt in range(1, retries + 1):
        try:
            async with httpx.AsyncClient(timeout=SUPABASE_TIMEOUT) as client:
                return await client.post(
                    url,
                    headers=_service_headers(prefer_return=True),
                    json=payload,
                )
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
            last_exc = exc
            if attempt < retries:
                await asyncio.sleep(backoff)
                backoff *= 2

    raise HTTPException(
        status_code=504,
        detail=(
            "Supabase POST timeout/error after retries: "
            f"table={table} error={type(last_exc).__name__}: {last_exc}"
        ),
    )


async def get_rows(path: str) -> list[dict[str, Any]]:
    response = await _supabase_get(path)
    if response.status_code >= 400:
        raise HTTPException(
            status_code=400,
            detail=f"Supabase fetch failed: {response.status_code} {response.text}",
        )
    return response.json() or []


async def post_row(table: str, payload: dict[str, Any]) -> dict[str, Any]:
    safe_payload = normalize_uuid_insert_payload(payload)
    print(
        "DEBUG UUID VALUES:",
        safe_payload.get("file_id"),
        safe_payload.get("report_id"),
        safe_payload.get("user_id"),
        safe_payload.get("submission_id"),
    )
    response = await _supabase_post(table, safe_payload)
    if response.status_code >= 400:
        raise HTTPException(
            status_code=400,
            detail=f"Supabase insert failed: {response.status_code} {response.text}",
        )
    rows = response.json() or []
    if not rows:
        raise HTTPException(status_code=500, detail="Supabase did not return created row")
    return rows[0]


def rate_limit(user_id: str) -> None:
    now = time.time()
    timestamps = [value for value in _RATE.get(user_id, []) if now - value < _RATE_WINDOW]
    if len(timestamps) >= _RATE_MAX:
        raise HTTPException(
            status_code=429,
            detail="Rate limit: too many report generations. Try later.",
        )
    timestamps.append(now)
    _RATE[user_id] = timestamps


def sha256_json(obj: Any) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def uuid_or_none(value: Any) -> str | None:
    return _uuid_or_none(value)


def normalize_uuid_insert_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return an insert payload with blank UUID-like fields converted to None."""

    return _normalize_uuid_insert_payload(payload)


def _submission_signal_text(ingestion: dict[str, Any]) -> str:
    return "\n".join(
        part
        for part in (
            str(ingestion.get("text_content") or "").strip(),
            str(ingestion.get("ocr_text") or "").strip(),
            str(ingestion.get("audio_transcript") or "").strip(),
            json.dumps(ingestion.get("tables_json") or {}, ensure_ascii=False),
        )
        if part
    )


def _heading_hits(lowered: str, headings: set[str]) -> int:
    return sum(1 for heading in headings if heading in lowered)


def _code_line_hits(text: str) -> int:
    return len(_CODE_LINE_RE.findall(text[:16000]))


def _citation_hits(text: str) -> int:
    return sum(len(pattern.findall(text[:16000])) for pattern in _CITATION_PATTERNS)


def _prose_signal_count(text: str) -> int:
    compact = " ".join((text or "").split())
    sentence_hits = len(_PROSE_SENTENCE_RE.findall(compact[:16000]))
    paragraph_hits = len(
        [
            part
            for part in re.split(r"\n\s*\n", text[:16000])
            if len(_WORD_RE.findall(part)) >= 45
        ]
    )
    return sentence_hits + paragraph_hits


def classify_submission_form(ingestion: dict[str, Any]) -> str:
    """Classify the submission surface form for prompt/routing decisions.

    The classifier intentionally prefers academic-writing modes whenever the
    submission shows chapter/report headings, citations, and sustained prose,
    even if the topic mentions architecture, testing, or implementation.
    """

    text = _submission_signal_text(ingestion)
    if not text.strip():
        haystack = text.lower()
        hits = sum(1 for term in _PROJECT_HINT_TERMS if term in haystack)
        return "code" if hits >= 3 else "essay"

    lowered = text.lower()
    project_term_hits = sum(1 for term in _PROJECT_HINT_TERMS if term in lowered)
    technical_term_hits = sum(1 for term in _TECHNICAL_EXERCISE_HINT_TERMS if term in lowered)
    academic_heading_hits = _heading_hits(lowered, _ACADEMIC_HEADING_TERMS)
    chapter_heading_hits = _heading_hits(lowered, _CHAPTER_HEADING_TERMS)
    report_heading_hits = _heading_hits(lowered, _REPORT_HEADING_TERMS)
    citation_hits = _citation_hits(text)
    prose_signal_hits = _prose_signal_count(text)
    code_line_hits = _code_line_hits(text)
    code_fence_hits = lowered.count("```")
    code_symbol_hits = sum(1 for ch in text[:12000] if ch in "{}();[]=<>")
    alpha_chars = sum(1 for ch in text[:12000] if ch.isalpha())
    symbol_ratio = (code_symbol_hits / alpha_chars) if alpha_chars else 0.0

    academic_score = (
        academic_heading_hits * 2
        + chapter_heading_hits * 2
        + report_heading_hits
        + min(citation_hits, 4) * 2
        + min(prose_signal_hits, 4)
    )
    chapter_score = chapter_heading_hits * 2 + min(citation_hits, 3)
    report_score = report_heading_hits * 2 + min(prose_signal_hits, 3)
    code_score = (
        min(project_term_hits, 5)
        + min(technical_term_hits, 4) * 2
        + code_line_hits * 2
        + code_fence_hits * 2
        + (2 if symbol_ratio >= 0.10 else 0)
    )

    if chapter_score >= 5 and academic_score >= code_score:
        return "chapter"
    if chapter_heading_hits >= 1 and academic_score >= 4 and code_score <= academic_score:
        return "chapter"
    if report_score >= 4 and academic_score + 1 >= code_score:
        return "report"
    if code_score >= 8 and academic_score <= 3:
        return "code"
    if code_score >= 6 and academic_score >= 6:
        return "mixed"
    if academic_score >= max(4, code_score):
        return "essay"
    if code_score >= academic_score + 3:
        return "code"
    if academic_heading_hits or citation_hits or prose_signal_hits >= 2:
        return "essay"
    return "code" if project_term_hits >= 3 else "essay"


def detect_submission_kind(ingestion: dict[str, Any]) -> str:
    submission_form = classify_submission_form(ingestion)
    if submission_form == "code":
        return "project"

    if submission_form in {"mixed", "report"}:
        lowered = _submission_signal_text(ingestion).lower()
        technical_signal_hits = sum(1 for term in _PROJECT_HINT_TERMS if term in lowered) + sum(
            1 for term in _TECHNICAL_EXERCISE_HINT_TERMS if term in lowered
        )
        if technical_signal_hits >= 4:
            return "project"

    return "academic"


def _phase7_as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _phase7_as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _phase7_text(value: Any, default: str, *, limit: int) -> str:
    if value is None:
        text = default
    elif isinstance(value, str):
        text = value.strip()
    else:
        text = str(value).strip()
    return (text or default)[:limit]


def _phase7_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off", ""}:
            return False
    return default


def _phase7_float(
    value: Any,
    *,
    default: float = 0.0,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, number))


def _phase7_int(
    value: Any,
    *,
    default: int = 1,
    minimum: int = 1,
    maximum: int = 10,
) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        try:
            number = int(float(value))
        except (TypeError, ValueError):
            return default
    return max(minimum, min(maximum, number))


def _phase7_restricted_mode(report: dict[str, Any] | None) -> bool:
    if not isinstance(report, dict):
        return False
    confidence = _phase7_as_dict(report.get("confidence"))
    safety = _phase7_as_dict(report.get("safety"))
    return (
        str(confidence.get("mode") or "").strip().lower() == "restricted"
        or _phase7_bool(safety.get("needs_review"), default=False)
    )


def _safe_mode_student(ml: dict[str, Any], *, reason: str = "") -> dict[str, Any]:
    confidence_0_to_4 = _phase7_int(
        ml.get("confidence_0_to_4"),
        default=0,
        minimum=0,
        maximum=4,
    )
    safe_reason = _phase7_text(
        reason,
        "Restricted mode was triggered due to safety or confidence checks.",
        limit=2000,
    )
    return {
        "summary": (
            "The submission triggered safety or confidence checks, so the system "
            "returned restricted feedback for manual review."
        ),
        "issues": [
            {
                "title": "Safety or confidence checks triggered",
                "evidence": (
                    "The submission contained safety, prompt-injection, or low-confidence patterns that "
                    "reduced trust in a full automated review. Manual review is recommended before relying "
                    "on this output."
                ),
                "severity": "high",
            }
        ],
        "strengths": [],
        "architecture_review": {
            "overview": "Restricted review mode.",
            "backend": "Restricted review mode.",
            "frontend": "Restricted review mode.",
            "database": "Restricted review mode.",
            "security": "Restricted review mode.",
        },
        "implementation_review": {
            "features_built": [],
            "technical_quality": "Restricted review mode.",
            "integration_quality": "Restricted review mode.",
        },
        "evaluation_review": {
            "testing_present": "Restricted review mode.",
            "limitations": "Automated evaluation was limited by safety or confidence checks.",
            "academic_quality": "Restricted review mode.",
        },
        "improvement_plan": [
            {
                "action": "Review the submission manually",
                "why": "Automated confidence was too limited for a full assessment.",
                "how": (
                    "Check the content for instruction-like text, ambiguity, weak grounding, or low-confidence "
                    "signals and regenerate after cleanup."
                ),
                "priority": 1,
            }
        ],
        "checklist": [
            {"item": "Remove instruction-like text", "done": False},
            {"item": "Ensure content is clearly academic", "done": False},
            {"item": "Re-run after improving evidence quality", "done": False},
        ],
        "confidence": {"mode": "restricted", "overall": 0.0},
        "model_agreement": {
            "ml_confidence": max(0.0, min(1.0, confidence_0_to_4 / 4.0)),
            "llm_confidence": 0.0,
            "final_confidence": 0.0,
        },
        "safety": {"needs_review": True, "reason": safe_reason},
    }


def _safe_mode_professor(_ml: dict[str, Any], *, reason: str = "") -> dict[str, Any]:
    safe_reason = _phase7_text(
        reason,
        "Restricted mode was triggered due to safety or moderation-confidence checks.",
        limit=2000,
    )
    return {
        "rubric_breakdown": [
            {
                "criterion": "Overall academic quality",
                "band": "Needs review",
                "justification": (
                    "Safety or moderation-confidence checks limited the automated assessment. "
                    "A human marker should review the submission before relying on this output."
                ),
            }
        ],
        "feedback_explanation": (
            "The submission triggered safety or moderation-confidence checks, so only restricted feedback "
            "is provided. Manual review is recommended before a rubric judgement is finalized."
        ),
        "moderation_notes": [
            {
                "risk": "Safety or confidence checks triggered",
                "note": "Review the submission manually before relying on this automated output.",
            },
            {
                "risk": "Limited moderation certainty",
                "note": "Regenerate after removing instruction-like text or improving evidence quality if needed.",
            },
        ],
        "safety": {"needs_review": True, "reason": safe_reason},
    }


def _as_issue_object(item: Any) -> dict[str, Any] | None:
    if isinstance(item, str):
        text = item.strip()
        if text:
            return {"title": text, "evidence": "Evidence was not provided.", "severity": "med"}
        return None
    if not isinstance(item, dict):
        return None

    title = str(
        item.get("title")
        or item.get("issue")
        or item.get("weakness")
        or item.get("label")
        or item.get("text")
        or ""
    ).strip()
    evidence = str(
        item.get("evidence")
        or item.get("details")
        or item.get("description")
        or ""
    ).strip()
    severity = str(item.get("severity") or item.get("level") or "med").strip().lower()
    if severity not in {"low", "med", "high"}:
        severity = "med"
    if not title and not evidence:
        return None
    return {
        "title": title or "Issue",
        "evidence": evidence or "Evidence was not provided.",
        "severity": severity,
    }


def _as_strength_object(item: Any) -> dict[str, Any] | None:
    if isinstance(item, str):
        text = item.strip()
        if text:
            return {"title": text, "evidence": "Supporting evidence was not provided."}
        return None
    if not isinstance(item, dict):
        return None

    title = str(
        item.get("title")
        or item.get("strength")
        or item.get("label")
        or item.get("text")
        or ""
    ).strip()
    evidence = str(
        item.get("evidence")
        or item.get("details")
        or item.get("description")
        or ""
    ).strip()
    if not title and not evidence:
        return None
    return {
        "title": title or "Strength",
        "evidence": evidence or "Supporting evidence was not provided.",
    }


def _as_improvement_object(item: Any, index: int) -> dict[str, Any] | None:
    if isinstance(item, str):
        text = item.strip()
        if text:
            return {
                "action": text,
                "why": "The reason was not provided.",
                "how": "The implementation detail was not provided.",
                "priority": index + 1,
            }
        return None
    if not isinstance(item, dict):
        return None

    action = str(
        item.get("action")
        or item.get("item")
        or item.get("step")
        or item.get("title")
        or item.get("text")
        or ""
    ).strip()
    why = str(item.get("why") or item.get("reason") or item.get("rationale") or "").strip()
    how = str(item.get("how") or item.get("details") or "").strip()
    if not how:
        steps = [
            str(step).strip()
            for step in _phase7_as_list(item.get("steps"))
            if str(step).strip()
        ]
        if steps:
            how = "; ".join(steps)
    priority = _student_priority_rank(item.get("priority"), index)
    if not action and not why and not how:
        return None
    return {
        "action": action or "Suggested improvement",
        "why": why or "The reason was not provided.",
        "how": how or "The implementation detail was not provided.",
        "priority": priority,
    }


def _as_checklist_object(item: Any) -> dict[str, Any] | None:
    if isinstance(item, str):
        text = item.strip()
        if text:
            return {"item": text, "done": False}
        return None
    if not isinstance(item, dict):
        return None

    text = str(
        item.get("item")
        or item.get("action")
        or item.get("label")
        or item.get("title")
        or item.get("objective")
        or item.get("text")
        or ""
    ).strip()
    done = item.get("done")
    if not isinstance(done, bool):
        done = bool(item.get("checked") or item.get("complete") or False)
    if not text:
        return None
    return {"item": text, "done": done}


def _normalize_confidence_mode(value: Any, *, restricted: bool) -> str:
    if restricted:
        return "restricted"
    return "restricted" if str(value or "").strip().lower() == "restricted" else "normal"


def _student_priority_rank(value: Any, index: int) -> int:
    if isinstance(value, str):
        mapped = {
            "high": 1,
            "medium": 2,
            "med": 2,
            "low": 3,
        }.get(value.strip().lower())
        if mapped is not None:
            return mapped
    return _phase7_int(value, default=index + 1)


def _student_improvement_source(data: dict[str, Any]) -> list[Any]:
    improvement_plan = _phase7_as_dict(data.get("improvement_plan"))
    actions = _phase7_as_list(improvement_plan.get("actions"))
    if actions:
        return actions

    for key in ("improvement_plan", "suggestions", "recommendations", "next_steps"):
        if key == "improvement_plan" and isinstance(data.get(key), dict):
            continue
        items = _phase7_as_list(data.get(key))
        if items:
            return items

    return []


def _student_checklist_source(data: dict[str, Any]) -> list[Any]:
    checklist = _phase7_as_list(data.get("checklist"))
    if checklist:
        return checklist

    learning_path = _phase7_as_dict(data.get("learning_path"))
    generated: list[Any] = []

    for item in _phase7_as_list(learning_path.get("recommended_practice")):
        if str(item or "").strip():
            generated.append(item)

    for milestone in _phase7_as_list(learning_path.get("milestones")):
        if isinstance(milestone, str):
            text = milestone.strip()
            if text:
                generated.append(text)
            continue
        if not isinstance(milestone, dict):
            continue
        title = str(milestone.get("title") or milestone.get("objective") or "").strip()
        if title:
            generated.append({"item": title, "done": False})
        for activity in _phase7_as_list(milestone.get("activities")):
            if str(activity or "").strip():
                generated.append(activity)

    if generated:
        return generated

    for item in _student_improvement_source(data):
        if isinstance(item, dict):
            text = str(
                item.get("action")
                or item.get("title")
                or item.get("item")
                or item.get("step")
                or ""
            ).strip()
            if text:
                generated.append(text)
        elif str(item or "").strip():
            generated.append(item)

    return generated


def _canonical_student_report(report: dict[str, Any] | None, *, restricted: bool) -> dict[str, Any]:
    data = report if isinstance(report, dict) else {}
    summary = data.get("summary")
    if (
        isinstance(summary, str)
        and summary.strip() == "The system could not confidently generate full feedback for this submission."
    ):
        summary = "The submission triggered safety or confidence checks, so the system returned limited feedback for manual review."

    issues_source = _phase7_as_list(data.get("issues")) or _phase7_as_list(data.get("weaknesses"))
    issues = [
        converted
        for item in issues_source
        if (converted := _as_issue_object(item)) is not None
    ]
    strengths = [
        converted
        for item in _phase7_as_list(data.get("strengths"))
        if (converted := _as_strength_object(item)) is not None
    ]
    plan = [
        converted
        for index, item in enumerate(_student_improvement_source(data))
        if (converted := _as_improvement_object(item, index)) is not None
    ]
    checklist = [
        converted
        for item in _student_checklist_source(data)
        if (converted := _as_checklist_object(item)) is not None
    ]

    architecture_review = _phase7_as_dict(data.get("architecture_review"))
    implementation_review = _phase7_as_dict(data.get("implementation_review"))
    evaluation_review = _phase7_as_dict(data.get("evaluation_review"))
    confidence = _phase7_as_dict(data.get("confidence"))
    model_agreement = _phase7_as_dict(data.get("model_agreement"))
    safety = _phase7_as_dict(data.get("safety"))

    final_confidence = _phase7_float(model_agreement.get("final_confidence"), default=-1.0)
    confidence_score = _phase7_float(confidence.get("score"), default=-1.0)
    overall = _phase7_float(confidence.get("overall"), default=-1.0)
    if overall < 0.0:
        overall = (
            final_confidence
            if final_confidence >= 0.0
            else (confidence_score if confidence_score >= 0.0 else (0.35 if restricted else 0.75))
        )

    return {
        "summary": _phase7_text(summary, "Automated review generated with limited confidence.", limit=1200),
        "issues": issues,
        "strengths": strengths,
        "architecture_review": {
            "overview": _phase7_text(architecture_review.get("overview"), "Not assessed.", limit=1200),
            "backend": _phase7_text(architecture_review.get("backend"), "Not assessed.", limit=1200),
            "frontend": _phase7_text(architecture_review.get("frontend"), "Not assessed.", limit=1200),
            "database": _phase7_text(architecture_review.get("database"), "Not assessed.", limit=1200),
            "security": _phase7_text(architecture_review.get("security"), "Not assessed.", limit=1200),
        },
        "implementation_review": {
            "features_built": [
                rendered
                for item in _phase7_as_list(implementation_review.get("features_built"))
                if (rendered := _phase7_text(item, "", limit=300))
            ],
            "technical_quality": _phase7_text(
                implementation_review.get("technical_quality"),
                "Not assessed.",
                limit=1200,
            ),
            "integration_quality": _phase7_text(
                implementation_review.get("integration_quality"),
                "Not assessed.",
                limit=1200,
            ),
        },
        "evaluation_review": {
            "testing_present": _phase7_text(
                evaluation_review.get("testing_present"),
                "Not assessed.",
                limit=1200,
            ),
            "limitations": _phase7_text(evaluation_review.get("limitations"), "Not assessed.", limit=1200),
            "academic_quality": _phase7_text(
                evaluation_review.get("academic_quality"),
                "Not assessed.",
                limit=1200,
            ),
        },
        "improvement_plan": plan,
        "checklist": checklist,
        "confidence": {
            "mode": _normalize_confidence_mode(confidence.get("mode"), restricted=restricted),
            "overall": overall,
        },
        "model_agreement": {
            "ml_confidence": _phase7_float(model_agreement.get("ml_confidence"), default=0.0),
            "llm_confidence": _phase7_float(
                model_agreement.get("llm_confidence"),
                default=(overall if overall >= 0.0 else 0.0),
            ),
            "final_confidence": _phase7_float(
                model_agreement.get("final_confidence"),
                default=(overall if overall >= 0.0 else 0.0),
            ),
        },
        "safety": {
            "needs_review": _phase7_bool(safety.get("needs_review"), default=restricted),
            "reason": _phase7_text(safety.get("reason"), "", limit=2000),
        },
    }


def _as_rubric_row(item: Any, index: int) -> dict[str, Any] | None:
    if isinstance(item, str):
        text = item.strip()
        if text:
            return {
                "criterion": f"Criterion {index + 1}",
                "band": "Needs review",
                "justification": text,
            }
        return None
    if not isinstance(item, dict):
        return None

    criterion = str(item.get("criterion") or item.get("title") or "").strip()
    band = str(item.get("band") or item.get("level") or "Needs review").strip()
    justification = str(
        item.get("justification")
        or item.get("note")
        or item.get("evidence")
        or item.get("description")
        or ""
    ).strip()
    if not criterion and not justification:
        return None
    return {
        "criterion": criterion or f"Criterion {index + 1}",
        "band": band or "Needs review",
        "justification": justification or "A detailed justification was not provided.",
    }


def _as_moderation_note(item: Any, index: int) -> dict[str, Any] | None:
    if isinstance(item, str):
        text = item.strip()
        if text:
            return {"risk": f"Risk {index + 1}", "note": text}
        return None
    if not isinstance(item, dict):
        return None

    risk = str(item.get("risk") or item.get("title") or item.get("label") or "").strip()
    note = str(item.get("note") or item.get("details") or item.get("description") or "").strip()
    if not risk and not note:
        return None
    return {
        "risk": risk or f"Risk {index + 1}",
        "note": note or "A moderation note was not provided.",
    }


def _canonical_professor_report(report: dict[str, Any] | None, *, restricted: bool) -> dict[str, Any]:
    data = report if isinstance(report, dict) else {}
    safety = _phase7_as_dict(data.get("safety"))
    rubric_breakdown = [
        converted
        for index, item in enumerate(_phase7_as_list(data.get("rubric_breakdown")))
        if (converted := _as_rubric_row(item, index)) is not None
    ]
    moderation_notes = [
        converted
        for index, item in enumerate(_phase7_as_list(data.get("moderation_notes")))
        if (converted := _as_moderation_note(item, index)) is not None
    ]
    feedback_explanation = _phase7_text(
        data.get("feedback_explanation") or data.get("summary"),
        "Detailed feedback explanation unavailable.",
        limit=1600,
    )
    if not rubric_breakdown:
        rubric_breakdown = [
            {
                "criterion": "Overall academic quality",
                "band": "Needs review",
                "justification": feedback_explanation,
            }
        ]

    return {
        "rubric_breakdown": rubric_breakdown,
        "feedback_explanation": feedback_explanation,
        "moderation_notes": moderation_notes,
        "safety": {
            "needs_review": _phase7_bool(safety.get("needs_review"), default=restricted),
            "reason": _phase7_text(safety.get("reason"), "", limit=2000),
        },
    }


def _normalize_student_report(report: dict[str, Any] | None) -> dict[str, Any]:
    try:
        return _phase10_normalize_student_payload(report or {}).model_dump()
    except Exception:
        restricted = _phase7_restricted_mode(report)
        fallback = _safe_mode_student(
            {},
            reason="A stored student report could not be normalized safely.",
        )
        fallback["confidence"]["mode"] = "restricted" if restricted else fallback["confidence"]["mode"]
        return _phase10_normalize_student_payload(fallback).model_dump()


def _normalize_professor_report(report: dict[str, Any] | None) -> dict[str, Any]:
    try:
        return _phase10_normalize_professor_payload(report or {}).model_dump()
    except Exception:
        fallback = _safe_mode_professor(
            {},
            reason="A stored professor report could not be normalized safely.",
        )
        return _phase10_normalize_professor_payload(fallback).model_dump()


def _normalize_rag_fields(row: dict[str, Any], *, nested_rag: dict[str, Any] | None = None) -> dict[str, Any]:
    normalized = dict(row)
    report = normalized.get("report_json")
    if not isinstance(nested_rag, dict):
        nested_rag = (
            report.get("rag_meta")
            if isinstance(report, dict) and isinstance(report.get("rag_meta"), dict)
            else {}
        )
    if not nested_rag and isinstance(normalized.get("rag_meta"), dict):
        nested_rag = normalized.get("rag_meta") or {}

    if not nested_rag:
        return normalized

    if not isinstance(normalized.get("citations"), list) or len(normalized.get("citations") or []) == 0:
        normalized["citations"] = nested_rag.get("citations", [])

    if not isinstance(normalized.get("retrieved_chunks"), list) or len(normalized.get("retrieved_chunks") or []) == 0:
        normalized["retrieved_chunks"] = nested_rag.get("retrieved_chunks", [])

    if not isinstance(normalized.get("rag_trace"), dict) or len(normalized.get("rag_trace") or {}) == 0:
        normalized["rag_trace"] = nested_rag.get("trace", {})

    if not isinstance(normalized.get("retrieval_confidence"), (int, float)):
        normalized["retrieval_confidence"] = nested_rag.get("confidence_score", 0.0)

    if not isinstance(normalized.get("retrieval_confidence_label"), str) or not str(
        normalized.get("retrieval_confidence_label") or ""
    ).strip():
        normalized["retrieval_confidence_label"] = nested_rag.get("confidence_label", "low")

    if not isinstance(normalized.get("safe_review"), bool):
        normalized["safe_review"] = bool(nested_rag.get("safe_review", False))

    return normalized


def normalize_report_row(role: str, row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return row
    normalized = dict(row)
    original_report = normalized.get("report_json")
    nested_rag = (
        original_report.get("rag_meta")
        if isinstance(original_report, dict) and isinstance(original_report.get("rag_meta"), dict)
        else {}
    )
    if role == "student":
        normalized["report_json"] = _normalize_student_report(original_report)
    elif role == "professor":
        normalized["report_json"] = _normalize_professor_report(original_report)
    return _normalize_rag_fields(normalized, nested_rag=nested_rag)


async def load_file(file_id: str, user: CurrentUser) -> dict[str, Any]:
    if user.role == "admin":
        rows = await get_rows(
            f"files?id=eq.{file_id}&select=id,status,mime_type,submission_id,created_at,user_id&limit=1"
        )
    else:
        rows = await get_rows(
            f"files?id=eq.{file_id}&user_id=eq.{user.id}"
            "&select=id,status,mime_type,submission_id,created_at,user_id&limit=1"
        )

    if not rows:
        raise HTTPException(status_code=404, detail="File not found")

    return rows[0]


async def build_ingestion_bundle(file_id: str, user: CurrentUser) -> Dict[str, Any]:
    user_filter = f"&user_id=eq.{user.id}" if user.role != "admin" else ""

    text_rows = await get_rows(
        f"extracted_text?file_id=eq.{file_id}&select=redacted_text&order=created_at.desc&limit=1{user_filter}"
    )
    transcript_rows = await get_rows(
        f"transcripts?file_id=eq.{file_id}&select=redacted_transcript&order=created_at.desc&limit=1{user_filter}"
    )
    tables = await get_rows(
        f"extracted_tables?file_id=eq.{file_id}&select=table_index,sheet_name,columns,rows"
        f"&order=created_at.desc&limit=25{user_filter}"
    )

    return {
        "text_content": (text_rows[0].get("redacted_text") if text_rows else "") or "",
        "ocr_text": "",
        "audio_transcript": (transcript_rows[0].get("redacted_transcript") if transcript_rows else "") or "",
        "tables_json": {"tables": tables},
    }


def _require_ai_service() -> None:
    if not settings.ai_service_url:
        raise HTTPException(status_code=500, detail="AI_SERVICE_URL not set")
    if not settings.ai_service_secret:
        raise HTTPException(status_code=500, detail="AI_SERVICE_SECRET not set")


def _ai_headers(user: CurrentUser) -> Dict[str, str]:
    return {
        "x-ai-secret": settings.ai_service_secret,
        "x-user-id": str(user.id),
        "x-role": str(user.role),
        "Content-Type": "application/json",
    }


def _map_quality_band_from_confidence(c04: int) -> str:
    if c04 <= 1:
        return "low"
    if c04 == 2:
        return "med"
    return "high"


def _map_depth(label: str) -> str:
    mapping = {
        "shallow": "low",
        "basic": "med",
        "medium": "med",
        "developed": "high",
        "deep": "high",
    }
    return mapping.get((label or "").lower(), "med")


def _map_consistency(label: str) -> str:
    mapping = {"inconsistent": "low", "mixed": "med", "consistent": "high"}
    return mapping.get((label or "").lower(), "med")


async def call_ai_student_multimodal(user: CurrentUser, ingestion: Dict[str, Any]) -> Dict[str, Any]:
    _require_ai_service()
    payload = {
        "text": ingestion.get("text_content", ""),
        "ocr": ingestion.get("ocr_text", ""),
        "audio": ingestion.get("audio_transcript", ""),
        "table": ingestion.get("tables_json", {}),
    }

    try:
        async with httpx.AsyncClient(timeout=AI_TIMEOUT) as client:
            feedback_response = await client.post(
                f"{settings.ai_service_url.rstrip('/')}/api/infer/student/feedback_multimodal",
                json=payload,
                headers=_ai_headers(user),
            )
            if feedback_response.status_code >= 400:
                raise HTTPException(
                    status_code=502,
                    detail=f"ai-service feedback failed: {feedback_response.text}",
                )
            confidence_response = await client.post(
                f"{settings.ai_service_url.rstrip('/')}/api/infer/student/confidence_multimodal",
                json=payload,
                headers=_ai_headers(user),
            )
            if confidence_response.status_code >= 400:
                raise HTTPException(
                    status_code=502,
                    detail=f"ai-service confidence failed: {confidence_response.text}",
                )
    except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
        raise HTTPException(
            status_code=504,
            detail=f"ai-service timeout/error: {type(exc).__name__}: {exc}",
        ) from exc

    feedback = feedback_response.json()
    confidence = confidence_response.json()

    probability = float((confidence.get("prediction") or {}).get("confidence") or 0.0)
    if probability < 0.35:
        confidence_bucket = 0
    elif probability < 0.55:
        confidence_bucket = 1
    elif probability < 0.70:
        confidence_bucket = 2
    elif probability < 0.85:
        confidence_bucket = 3
    else:
        confidence_bucket = 4

    feedback_prediction = feedback.get("prediction") or {}
    return {
        "feedback_category": str(feedback_prediction.get("label") or "other"),
        "quality_band": _map_quality_band_from_confidence(confidence_bucket),
        "confidence_0_to_4": confidence_bucket,
        "raw": {"feedback": feedback, "confidence": confidence},
    }


async def call_ai_professor_multimodal(user: CurrentUser, ingestion: Dict[str, Any]) -> Dict[str, Any]:
    _require_ai_service()
    payload = {
        "text": ingestion.get("text_content", ""),
        "ocr": ingestion.get("ocr_text", ""),
        "audio": ingestion.get("audio_transcript", ""),
        "table": ingestion.get("tables_json", {}),
    }

    try:
        async with httpx.AsyncClient(timeout=AI_TIMEOUT) as client:
            response = await client.post(
                f"{settings.ai_service_url.rstrip('/')}/api/infer/professor/multimodal/rubric-suite",
                json=payload,
                headers=_ai_headers(user),
            )
    except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
        raise HTTPException(
            status_code=504,
            detail=f"ai-service timeout/error: {type(exc).__name__}: {exc}",
        ) from exc

    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"ai-service rubric suite failed: {response.text}")

    payload_out = response.json()
    predictions = payload_out.get("predictions") or {}
    depth_raw = (predictions.get("argument_depth") or {}).get("label") or "medium"
    consistency_raw = (predictions.get("moderation_consistency") or {}).get("label") or "mixed"

    return {
        "rubric_band": (predictions.get("rubric_band") or {}).get("label") or "adequate",
        "argument_depth": _map_depth(depth_raw),
        "moderation_consistency": _map_consistency(consistency_raw),
        "raw": payload_out,
        "raw_labels": {
            "argument_depth": depth_raw,
            "moderation_consistency": consistency_raw,
        },
    }


__all__ = [
    "build_ingestion_bundle",
    "call_ai_professor_multimodal",
    "call_ai_student_multimodal",
    "classify_submission_form",
    "detect_submission_kind",
    "get_rows",
    "load_file",
    "normalize_report_row",
    "normalize_uuid_insert_payload",
    "post_row",
    "rate_limit",
    "sha256_json",
    "uuid_or_none",
]
