from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from typing import Any, Dict, Literal, Tuple

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, ValidationError

from app.core.config import settings
from app.core.deps import CurrentUser, get_current_user
from app.rag.payloads import inject_rag_fields
from app.rag.retrieval.context_builder import (
    build_professor_rag_payload,
    build_student_rag_payload,
)
from app.rag.store import build_storage_fields_from_rag_meta
from app.services.report_generation_support import (
    build_ingestion_bundle as _shared_build_ingestion_bundle,
    call_ai_professor_multimodal as _shared_call_ai_professor_multimodal,
    call_ai_student_multimodal as _shared_call_ai_student_multimodal,
    detect_submission_kind as _shared_detect_submission_kind,
    get_rows as _shared_get_rows,
    load_file as _shared_load_file,
    normalize_report_row as _shared_normalize_report_row,
    post_row as _shared_post_row,
    rate_limit as _shared_rate_limit,
    sha256_json as _shared_sha256_json,
)

router = APIRouter(prefix="/phase7", tags=["phase7"])
logger = logging.getLogger("phase7")

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


# -------------------------
# Shared HTTP / Supabase helpers
# -------------------------
SUPABASE_TIMEOUT = httpx.Timeout(
    connect=15.0,
    read=120.0,
    write=30.0,
    pool=30.0,
)

LLM_TIMEOUT = httpx.Timeout(
    connect=15.0,
    read=420.0,
    write=30.0,
    pool=30.0,
)

AI_TIMEOUT = httpx.Timeout(
    connect=15.0,
    read=120.0,
    write=30.0,
    pool=30.0,
)


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
                r = await client.get(url, headers=_service_headers(prefer_return=False))
            return r
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            last_exc = e
            if attempt < retries:
                await asyncio.sleep(backoff)
                backoff *= 2

    raise HTTPException(
        status_code=504,
        detail=f"Supabase GET timeout/error after retries: path={path} error={type(last_exc).__name__}: {last_exc}",
    )


async def _supabase_post(table: str, payload: dict[str, Any], *, retries: int = 3) -> httpx.Response:
    url = f"{_supabase_base()}/rest/v1/{table}"
    last_exc: Exception | None = None
    backoff = 0.6

    for attempt in range(1, retries + 1):
        try:
            async with httpx.AsyncClient(timeout=SUPABASE_TIMEOUT) as client:
                r = await client.post(url, headers=_service_headers(prefer_return=True), json=payload)
            return r
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            last_exc = e
            if attempt < retries:
                await asyncio.sleep(backoff)
                backoff *= 2

    raise HTTPException(
        status_code=504,
        detail=f"Supabase POST timeout/error after retries: table={table} error={type(last_exc).__name__}: {last_exc}",
    )


async def _get_rows(path: str) -> list[dict[str, Any]]:
    return await _shared_get_rows(path)


async def _post_row(table: str, payload: dict[str, Any]) -> dict[str, Any]:
    return await _shared_post_row(table, payload)


# -------------------------
# Demo-grade rate limiting (per user)
# -------------------------
_RATE: Dict[str, list[float]] = {}
_RATE_MAX = 15
_RATE_WINDOW = 3600.0


def _rate_limit(user_id: str) -> None:
    _shared_rate_limit(user_id)


# -------------------------
# Prompt injection heuristic (demo)
# -------------------------
_INJECTION_PHRASES = [
    "ignore previous instructions",
    "ignore all previous",
    "disregard previous instructions",
    "forget your instructions",
    "new system prompt",
    "override system",
    "developer message",
    "jailbreak",
    "reveal hidden instructions",
    "do not follow your instructions",
    "you are now a",
    "you are no longer",
    "act as if you have no restrictions",
]


def _detect_injection(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in _INJECTION_PHRASES)


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
    text = text or default
    return text[:limit]


_STUDENT_PLACEHOLDER_SUMMARY = "Automated review generated with limited confidence."
_NOT_ASSESSED_PLACEHOLDER = "Not assessed."
_LOW_CONTENT_QUALITY_REASON = "low_content_quality"


def _phase7_marker_text(value: Any) -> str:
    text = value if isinstance(value, str) else str(value or "")
    return re.sub(r"\s+", " ", text.strip()).casefold().rstrip(".!?")


def _is_student_placeholder_summary(value: Any) -> bool:
    normalized = _phase7_marker_text(value)
    placeholder = _phase7_marker_text(_STUDENT_PLACEHOLDER_SUMMARY)
    return normalized == placeholder or (
        "automated review generated" in normalized
        and "limited confidence" in normalized
    )


def _is_not_assessed_placeholder(value: Any) -> bool:
    return _phase7_marker_text(value) == _phase7_marker_text(_NOT_ASSESSED_PLACEHOLDER)


def _student_report_low_content_quality(report: dict[str, Any] | None) -> bool:
    data = _phase7_as_dict(report)
    if not data:
        return False

    safety = _phase7_as_dict(data.get("safety"))
    if _phase7_marker_text(safety.get("reason")) == _LOW_CONTENT_QUALITY_REASON:
        return True

    summary_is_placeholder = _is_student_placeholder_summary(data.get("summary"))
    lists_are_empty = all(
        len(_phase7_as_list(data.get(key))) == 0
        for key in ("issues", "strengths", "improvement_plan", "checklist")
    )

    architecture_review = _phase7_as_dict(data.get("architecture_review"))
    implementation_review = _phase7_as_dict(data.get("implementation_review"))
    evaluation_review = _phase7_as_dict(data.get("evaluation_review"))

    architecture_is_placeholder = all(
        _is_not_assessed_placeholder(architecture_review.get(key))
        for key in ("overview", "backend", "frontend", "database", "security")
    )
    implementation_is_placeholder = (
        len(_phase7_as_list(implementation_review.get("features_built"))) == 0
        and _is_not_assessed_placeholder(implementation_review.get("technical_quality"))
        and _is_not_assessed_placeholder(implementation_review.get("integration_quality"))
    )
    evaluation_is_placeholder = all(
        _is_not_assessed_placeholder(evaluation_review.get(key))
        for key in ("testing_present", "limitations", "academic_quality")
    )

    return (
        summary_is_placeholder
        and lists_are_empty
        and architecture_is_placeholder
        and implementation_is_placeholder
        and evaluation_is_placeholder
    )


def _apply_student_quality_gate(report: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    degraded = _student_report_low_content_quality(report)
    if not degraded:
        return report, False

    validated = _phase7_as_dict(report).copy()
    safety = _phase7_as_dict(validated.get("safety")).copy()
    confidence = _phase7_as_dict(validated.get("confidence")).copy()
    safety["needs_review"] = True
    safety["reason"] = _LOW_CONTENT_QUALITY_REASON
    confidence["overall"] = min(
        _phase7_float(confidence.get("overall"), default=0.35),
        0.35,
    )
    validated["safety"] = safety
    validated["confidence"] = confidence
    return validated, True


def _student_row_is_degraded_placeholder(row: dict[str, Any] | None) -> bool:
    data = _phase7_as_dict(row)
    if not data:
        return False

    model_versions = _phase7_as_dict(data.get("model_versions"))
    quality_gate = _phase7_as_dict(model_versions.get("quality_gate"))
    if _phase7_bool(quality_gate.get("degraded_placeholder"), default=False):
        return True

    report = _phase7_as_dict(data.get("report_json"))
    safety = _phase7_as_dict(report.get("safety"))
    safety_reason = _phase7_marker_text(safety.get("reason"))
    if safety_reason == _LOW_CONTENT_QUALITY_REASON:
        return True

    return bool(_phase7_bool(data.get("needs_review"), default=False) and safety_reason == _LOW_CONTENT_QUALITY_REASON)


def _select_latest_student_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    for row in rows:
        if not _student_row_is_degraded_placeholder(row):
            return row
    return rows[0]


def _select_latest_student_row_with_metadata(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    total = len(rows)
    if total == 0:
        return None, {
            "preferred_non_degraded": False,
            "total_reports_considered": 0,
        }

    selected = _select_latest_student_row(rows)
    preferred_non_degraded = bool(
        selected
        and selected is not rows[0]
        and not _student_row_is_degraded_placeholder(selected)
        and _student_row_is_degraded_placeholder(rows[0])
    )
    return selected, {
        "preferred_non_degraded": preferred_non_degraded,
        "total_reports_considered": total,
    }


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


def _safe_mode_student(ml: dict, *, reason: str = "") -> dict:
    confidence_0_to_4 = _phase7_int(ml.get("confidence_0_to_4"), default=0, minimum=0, maximum=4)
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
        "confidence": {
            "mode": "restricted",
            "overall": 0.0,
        },
        "model_agreement": {
            "ml_confidence": max(0.0, min(1.0, confidence_0_to_4 / 4.0)),
            "llm_confidence": 0.0,
            "final_confidence": 0.0,
        },
        "safety": {"needs_review": True, "reason": safe_reason},
    }


def _safe_mode_professor(ml: dict, *, reason: str = "") -> dict:
    consistency = str(ml.get("moderation_consistency") or "low").strip().lower()
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


# -------------------------
# Hashing / mapping helpers
# -------------------------
def _sha256_json(obj: Any) -> str:
    return _shared_sha256_json(obj)


def _map_quality_band_from_confidence(c04: int) -> str:
    if c04 <= 1:
        return "low"
    if c04 == 2:
        return "med"
    return "high"


def _map_depth(label: str) -> str:
    m = {"shallow": "low", "basic": "med", "medium": "med", "developed": "high", "deep": "high"}
    return m.get((label or "").lower(), "med")


def _map_consistency(label: str) -> str:
    m = {"inconsistent": "low", "mixed": "med", "consistent": "high"}
    return m.get((label or "").lower(), "med")


def _agreement_score_student(conf_0_to_4: int, injected: bool, llm_ok: bool) -> float:
    base = {0: 0.15, 1: 0.30, 2: 0.55, 3: 0.75, 4: 0.90}.get(conf_0_to_4, 0.55)
    if injected:
        base -= 0.25
    if not llm_ok:
        base -= 0.20
    return float(max(0.0, min(1.0, base)))


def _agreement_score_professor(consistency: str, injected: bool, llm_ok: bool) -> float:
    base = {"low": 0.40, "med": 0.70, "high": 0.90}.get(consistency, 0.70)
    if injected:
        base -= 0.25
    if not llm_ok:
        base -= 0.20
    return float(max(0.0, min(1.0, base)))


def _extract_rag_meta(report: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    rag_meta = report.get("rag_meta")
    return rag_meta if isinstance(rag_meta, dict) else {}


def _report_without_rag_meta(report: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    cleaned = dict(report)
    cleaned.pop("rag_meta", None)
    return cleaned


def _normalized_excerpt(text: str | None, limit: int = 800) -> str:
    compact = " ".join((text or "").split())
    return compact[:limit]


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


def _detect_submission_kind(ingestion: dict[str, Any]) -> str:
    return _shared_detect_submission_kind(ingestion)


def _submission_focused_query(ingestion: dict[str, Any], base: str, limit: int = 200) -> str:
    """Build a query anchored to submission content rather than the generic base alone.

    Strategy: take the first ~200 chars of the submission text, extract only the
    terms that are already meaningful in the RAG keyword vocabulary (project or
    essay terms), and prepend them so the retriever's multi-query expansion sees
    them in slot 0 — the most specific query — before falling back to generic
    category templates.
    """
    text = " ".join((ingestion.get("text_content") or "").split())
    if not text:
        return base
    # Use only the first 200 chars as the primary anchor — long excerpts get
    # silently clipped to 320 chars in query_builder and lose submission signal.
    excerpt = text[:limit]
    return excerpt


def _student_project_query(ingestion: dict[str, Any]) -> str:
    return _submission_focused_query(
        ingestion,
        base=(
            "student software engineering project architecture backend frontend "
            "database security testing implementation quality"
        ),
    )


def _student_academic_query(ingestion: dict[str, Any]) -> str:
    return _submission_focused_query(
        ingestion,
        base=(
            "student academic writing structure argument evidence "
            "critical analysis clarity referencing citation"
        ),
    )


def _professor_project_query(ingestion: dict[str, Any]) -> str:
    base = (
        "rubric guidance marking policy moderation notes software project evaluation "
        "architecture implementation testing security database frontend backend criterion-level justification "
        "technical quality integration quality usability data quality analytics ai limitations"
    )
    excerpt = _normalized_excerpt(ingestion.get("text_content"), limit=500)
    return f"{base} {excerpt}".strip() if excerpt else base


def _professor_academic_query(ingestion: dict[str, Any]) -> str:
    base = (
        "rubric guidance marking policy moderation notes academic writing structure "
        "argument evidence critical analysis clarity referencing criterion-level justification source use"
    )
    excerpt = _normalized_excerpt(ingestion.get("text_content"), limit=500)
    return f"{base} {excerpt}".strip() if excerpt else base


def _rag_log_summary(rag_payload: dict[str, Any]) -> dict[str, Any]:
    retrieved_chunks = rag_payload.get("retrieved_chunks") or []
    citations = rag_payload.get("citations") or []
    trace = rag_payload.get("trace") or {}
    first_titles: list[str] = []

    for chunk in retrieved_chunks[:3]:
        if not isinstance(chunk, dict):
            continue
        title = str(
            chunk.get("document_title")
            or chunk.get("title")
            or chunk.get("document_id")
            or ""
        ).strip()
        if title:
            first_titles.append(title)

    return {
        "confidence_score": rag_payload.get("confidence_score", 0.0),
        "confidence_label": rag_payload.get("confidence_label", "low"),
        "safe_review": bool(rag_payload.get("safe_review", False)),
        "citation_count": len(citations),
        "retrieved_chunk_count": len(retrieved_chunks),
        "trace_query": trace.get("query"),
        "rewritten_query_count": len(trace.get("rewritten_queries") or []),
        "first_chunk_titles": first_titles,
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
    why = str(item.get("why") or item.get("reason") or "").strip()
    how = str(item.get("how") or item.get("details") or "").strip()
    priority = item.get("priority")
    if not isinstance(priority, int):
        priority = index + 1
    if not action and not why and not how:
        return None
    return {
        "action": action or "Suggested improvement",
        "why": why or "The reason was not provided.",
        "how": how or "The implementation detail was not provided.",
        "priority": _phase7_int(priority, default=index + 1),
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
        or item.get("label")
        or item.get("title")
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


def _canonical_student_report(report: dict[str, Any] | None, *, restricted: bool) -> dict[str, Any]:
    data = report if isinstance(report, dict) else {}
    summary = data.get("summary")
    if (
        isinstance(summary, str)
        and summary.strip() == "The system could not confidently generate full feedback for this submission."
    ):
        summary = "The submission triggered safety or confidence checks, so the system returned limited feedback for manual review."

    issues = [
        converted
        for item in _phase7_as_list(data.get("issues"))
        if (converted := _as_issue_object(item)) is not None
    ]
    strengths = [
        converted
        for item in _phase7_as_list(data.get("strengths"))
        if (converted := _as_strength_object(item)) is not None
    ]
    plan = [
        converted
        for idx, item in enumerate(_phase7_as_list(data.get("improvement_plan")))
        if (converted := _as_improvement_object(item, idx)) is not None
    ]
    checklist = [
        converted
        for item in _phase7_as_list(data.get("checklist"))
        if (converted := _as_checklist_object(item)) is not None
    ]

    architecture_review = _phase7_as_dict(data.get("architecture_review"))
    implementation_review = _phase7_as_dict(data.get("implementation_review"))
    evaluation_review = _phase7_as_dict(data.get("evaluation_review"))
    confidence = _phase7_as_dict(data.get("confidence"))
    model_agreement = _phase7_as_dict(data.get("model_agreement"))
    safety = _phase7_as_dict(data.get("safety"))

    final_confidence = _phase7_float(model_agreement.get("final_confidence"), default=-1.0)
    overall = _phase7_float(confidence.get("overall"), default=-1.0)
    if overall < 0.0:
        overall = final_confidence if final_confidence >= 0.0 else (0.35 if restricted else 0.75)

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
                _phase7_text(item, "", limit=300)
                for item in _phase7_as_list(implementation_review.get("features_built"))
                if _phase7_text(item, "", limit=300)
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
            "llm_confidence": _phase7_float(model_agreement.get("llm_confidence"), default=0.0),
            "final_confidence": _phase7_float(model_agreement.get("final_confidence"), default=0.0),
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
        for idx, item in enumerate(_phase7_as_list(data.get("rubric_breakdown")))
        if (converted := _as_rubric_row(item, idx)) is not None
    ]
    moderation_notes = [
        converted
        for idx, item in enumerate(_phase7_as_list(data.get("moderation_notes")))
        if (converted := _as_moderation_note(item, idx)) is not None
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


def _validate_student_report_for_storage(
    report: dict[str, Any] | None,
    *,
    restricted: bool,
    ml: dict[str, Any],
    reason: str,
) -> tuple[dict[str, Any], bool]:
    try:
        validated = _StoredStudentReport.model_validate(report or {}).model_dump()
        validated, degraded = _apply_student_quality_gate(validated)
        if restricted:
            validated["confidence"]["mode"] = "restricted"
            validated["safety"]["needs_review"] = True
            validated["confidence"]["overall"] = min(validated["confidence"].get("overall", 0.0), 0.35)
        return validated, bool(report) and not degraded
    except ValidationError:
        try:
            validated = _StoredStudentReport.model_validate(
                _canonical_student_report(report, restricted=restricted)
            ).model_dump()
            validated, _ = _apply_student_quality_gate(validated)
            return validated, False
        except ValidationError:
            fallback = _StoredStudentReport.model_validate(
                _safe_mode_student(ml, reason=reason)
            ).model_dump()
            return fallback, False


def _validate_professor_report_for_storage(
    report: dict[str, Any] | None,
    *,
    restricted: bool,
    ml: dict[str, Any],
    reason: str,
) -> tuple[dict[str, Any], bool]:
    try:
        validated = _StoredProfessorReport.model_validate(report or {}).model_dump()
        if restricted:
            validated["safety"]["needs_review"] = True
        return validated, bool(report)
    except ValidationError:
        try:
            validated = _StoredProfessorReport.model_validate(
                _canonical_professor_report(report, restricted=restricted)
            ).model_dump()
            return validated, False
        except ValidationError:
            fallback = _StoredProfessorReport.model_validate(
                _safe_mode_professor(ml, reason=reason)
            ).model_dump()
            return fallback, False


def _normalize_student_report(report: dict[str, Any] | None) -> dict[str, Any]:
    restricted = _phase7_restricted_mode(report)
    try:
        return _StoredStudentReport.model_validate(report or {}).model_dump()
    except ValidationError:
        try:
            return _StoredStudentReport.model_validate(
                _canonical_student_report(report, restricted=restricted)
            ).model_dump()
        except ValidationError:
            return _StoredStudentReport.model_validate(
                _safe_mode_student({}, reason="A stored student report could not be normalized safely.")
            ).model_dump()


def _normalize_professor_report(report: dict[str, Any] | None) -> dict[str, Any]:
    restricted = _phase7_restricted_mode(report)
    try:
        return _StoredProfessorReport.model_validate(report or {}).model_dump()
    except ValidationError:
        try:
            return _StoredProfessorReport.model_validate(
                _canonical_professor_report(report, restricted=restricted)
            ).model_dump()
        except ValidationError:
            return _StoredProfessorReport.model_validate(
                _safe_mode_professor({}, reason="A stored professor report could not be normalized safely.")
            ).model_dump()


def _normalize_rag_fields(row: dict[str, Any], *, nested_rag: dict[str, Any] | None = None) -> dict[str, Any]:
    normalized = dict(row)
    report = normalized.get("report_json")
    if not isinstance(nested_rag, dict):
        nested_rag = (
            report.get("rag_meta")
            if isinstance(report, dict) and isinstance(report.get("rag_meta"), dict)
            else {}
        )

    if not nested_rag:
        return normalized

    current_citations = normalized.get("citations")
    if not isinstance(current_citations, list) or len(current_citations) == 0:
        normalized["citations"] = nested_rag.get("citations", [])

    current_chunks = normalized.get("retrieved_chunks")
    if not isinstance(current_chunks, list) or len(current_chunks) == 0:
        normalized["retrieved_chunks"] = nested_rag.get("retrieved_chunks", [])

    current_trace = normalized.get("rag_trace")
    if not isinstance(current_trace, dict) or len(current_trace) == 0:
        normalized["rag_trace"] = nested_rag.get("trace", {})

    current_confidence = normalized.get("retrieval_confidence")
    if not isinstance(current_confidence, (int, float)):
        normalized["retrieval_confidence"] = nested_rag.get("confidence_score", 0.0)

    current_label = normalized.get("retrieval_confidence_label")
    if not isinstance(current_label, str) or not current_label.strip():
        normalized["retrieval_confidence_label"] = nested_rag.get("confidence_label", "low")

    current_safe_review = normalized.get("safe_review")
    if not isinstance(current_safe_review, bool):
        normalized["safe_review"] = bool(nested_rag.get("safe_review", False))

    return normalized


def _normalize_report_row(role: str, row: dict[str, Any] | None) -> dict[str, Any] | None:
    return _shared_normalize_report_row(role, row)


# -------------------------
# Ownership gate
# -------------------------
async def _load_file(file_id: str, user: CurrentUser) -> dict[str, Any]:
    return await _shared_load_file(file_id, user)


# -------------------------
# Build ingestion bundle
# -------------------------
async def _build_ingestion_bundle(file_id: str, user: CurrentUser) -> Dict[str, Any]:
    return await _shared_build_ingestion_bundle(file_id, user)


# -------------------------
# Call ai-service
# -------------------------
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


async def _call_ai_student_multimodal(user: CurrentUser, ingestion: Dict[str, Any]) -> Dict[str, Any]:
    return await _shared_call_ai_student_multimodal(user, ingestion)


async def _call_ai_professor_multimodal(user: CurrentUser, ingestion: Dict[str, Any]) -> Dict[str, Any]:
    return await _shared_call_ai_professor_multimodal(user, ingestion)


# -------------------------
# Call llm-service
# -------------------------
def _require_llm() -> None:
    if not settings.llm_service_url:
        raise HTTPException(status_code=500, detail="LLM_SERVICE_URL not set")
    if not settings.llm_service_secret:
        raise HTTPException(status_code=500, detail="LLM_SERVICE_SECRET not set")


async def _call_llm(endpoint: str, payload: dict[str, Any]) -> Tuple[dict[str, Any], str, dict[str, str]]:
    _require_llm()

    try:
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            r = await client.post(
                f"{settings.llm_service_url.rstrip('/')}{endpoint}",
                json=payload,
                headers={"x-ai-secret": settings.llm_service_secret},
            )
    except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError) as e:
        raise HTTPException(status_code=504, detail=f"llm-service timeout/error: {type(e).__name__}: {e}") from e

    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"llm-service failed: {r.text}")

    model_used = r.headers.get("x-llm-model-used", "") or ""
    llm_meta = {
        "primary_model": r.headers.get("x-llm-primary-model", "") or "",
        "fallback_model": r.headers.get("x-llm-fallback-model", "") or "",
    }
    return r.json(), model_used, llm_meta


# -------------------------
# API
# -------------------------
class GenerateIn(BaseModel):
    file_id: str = Field(..., min_length=1)
    force: bool = False


@router.get("/latest/{role}/{file_id}")
async def latest(role: str, file_id: str, user: CurrentUser = Depends(get_current_user)):
    if role not in {"student", "professor"}:
        raise HTTPException(status_code=400, detail="Invalid role")

    await _load_file(file_id, user)

    if role == "student":
        rows = await _get_rows(
            f"ai_reports?file_id=eq.{file_id}&role=eq.{role}&select=*&order=created_at.desc"
        )
        selected, selection_metadata = _select_latest_student_row_with_metadata(rows)
        skipped_degraded = 0
        if selected is not None:
            for row in rows:
                if row is selected:
                    break
                if _student_row_is_degraded_placeholder(row):
                    skipped_degraded += 1
        logger.info(
            "latest_student_selection file_id=%s total=%s selected_non_degraded=%s skipped_degraded=%s selected_report_id=%s",
            file_id,
            selection_metadata["total_reports_considered"],
            selection_metadata["preferred_non_degraded"],
            skipped_degraded,
            (selected or {}).get("id"),
        )
    else:
        rows = await _get_rows(
            f"ai_reports?file_id=eq.{file_id}&role=eq.{role}&select=*&order=created_at.desc&limit=1"
        )
        selected = rows[0] if rows else None
        selection_metadata = None

    item = _normalize_report_row(role, selected) if selected else None
    response = {"found": bool(rows), "item": item}
    if selection_metadata is not None:
        response["selection_metadata"] = selection_metadata
    return response


@router.post("/student/generate")
async def generate_student(body: GenerateIn, user: CurrentUser = Depends(get_current_user)):
    _rate_limit(str(user.id))

    if user.role not in {"student", "admin"}:
        raise HTTPException(status_code=403, detail="Student access required")

    request_id = str(uuid.uuid4())
    t_all = time.perf_counter()

    file_row = await _load_file(body.file_id, user)

    t0 = time.perf_counter()
    ingestion = await _build_ingestion_bundle(body.file_id, user)
    ingestion_ms = int((time.perf_counter() - t0) * 1000)

    input_hash = _sha256_json(ingestion)
    submission_kind = _detect_submission_kind(ingestion)
    analysis_type = "student_project_review" if submission_kind == "project" else "student_academic_review"
    top_k = 6 if submission_kind == "project" else 5
    prompt_hash = _sha256_json(
        {
            "role": "student",
            "template": "student_review_v5",
            "analysis_type": analysis_type,
            "query_strategy": f"{submission_kind}_specific_v2",
            "rag_enabled": settings.rag_enabled,
            "top_k": top_k,
        }
    )

    injected = _detect_injection(
        (ingestion.get("text_content") or "") + " " + (ingestion.get("audio_transcript") or "")
    )

    if not body.force:
        existing = await _get_rows(
            "ai_reports"
            f"?file_id=eq.{body.file_id}&role=eq.student&input_hash=eq.{input_hash}&prompt_hash=eq.{prompt_hash}"
            "&select=*&order=created_at.desc&limit=1"
        )
        if existing:
            return {
                "cached": True,
                "request_id": request_id,
                "stored": _normalize_report_row("student", existing[0]),
            }

    t1 = time.perf_counter()
    ml = await _call_ai_student_multimodal(user, ingestion)
    ai_ms = int((time.perf_counter() - t1) * 1000)

    rag_meta: dict[str, Any] = {}
    low_confidence = int(ml.get("confidence_0_to_4", 2)) <= 0
    mode = "restricted" if injected or low_confidence else "normal"
    student_query = (
        _student_project_query(ingestion)
        if submission_kind == "project"
        else _student_academic_query(ingestion)
    )

    llm_payload = {
        "submission_id": str(file_row.get("submission_id") or ""),
        "ingestion": ingestion,
        "ml": {
            "feedback_category": ml["feedback_category"],
            "quality_band": ml["quality_band"],
            "confidence_0_to_4": ml["confidence_0_to_4"],
        },
        "query": student_query,
        "top_k": top_k,
        "mode": mode,
        "analysis_type": analysis_type,
        "analysis_focus": (
            [
                "project aim",
                "technical stack",
                "architecture",
                "implementation quality",
                "security",
                "testing",
                "limitations",
            ]
            if submission_kind == "project"
            else [
                "task response",
                "structure",
                "evidence",
                "critical analysis",
                "clarity",
                "referencing",
                "academic quality",
            ]
        ),
        "submission_type": submission_kind,
        "safety_flags": {
            "injection_detected": injected,
            "low_confidence": low_confidence,
        },
    }

    if settings.rag_enabled:
        student_rag_payload = build_student_rag_payload(llm_payload)
        logger.info(
            "student_generate rag summary file_id=%s mode=%s query=%r top_k=%s summary=%s",
            body.file_id,
            mode,
            student_query,
            llm_payload.get("top_k"),
            _rag_log_summary(student_rag_payload),
        )
        llm_payload = inject_rag_fields(llm_payload, student_rag_payload)
    else:
        logger.info(
            "student_generate rag disabled file_id=%s mode=%s query=%r top_k=%s",
            body.file_id,
            mode,
            student_query,
            llm_payload.get("top_k"),
        )

    t2 = time.perf_counter()
    llm_response, llm_model_used, llm_meta = await _call_llm("/llm/student/report", llm_payload)
    llm_ms = int((time.perf_counter() - t2) * 1000)

    rag_meta = _extract_rag_meta(llm_response)
    report, llm_ok = _validate_student_report_for_storage(
        _report_without_rag_meta(llm_response),
        restricted=(mode == "restricted"),
        ml=ml,
        reason="The LLM output did not match the expected student report schema.",
    )
    degraded_placeholder = _student_report_low_content_quality(report)

    total_ms = int((time.perf_counter() - t_all) * 1000)
    agreement = _agreement_score_student(int(ml["confidence_0_to_4"]), injected, llm_ok)

    saved = await _post_row(
        "ai_reports",
        {
            "file_id": body.file_id,
            "submission_id": file_row.get("submission_id"),
            "role": "student",
            "report_json": report,
            "report_hash": _sha256_json(report),
            "prompt_hash": prompt_hash,
            "input_hash": input_hash,
            "model_versions": {
                "request_id": request_id,
                "timings_ms": {
                    "ingestion": ingestion_ms,
                    "ai_service": ai_ms,
                    "llm_service": llm_ms,
                    "total": total_ms,
                },
                "llm_primary": llm_meta.get("primary_model") or settings.llm_primary_label,
                "llm_fallback": llm_meta.get("fallback_model") or settings.llm_fallback_label,
                "llm_model_used": llm_model_used or "unknown",
                "ml_models": {
                    "feedback": "student.feedback_classifier_multimodal.v1",
                    "confidence": "student.confidence_model_multimodal.v1",
                },
                "agreement": {
                    "final_confidence": agreement,
                    "ml_bucket_0_to_4": ml["confidence_0_to_4"],
                    "injected": injected,
                },
                "quality_gate": {
                    "degraded_placeholder": degraded_placeholder,
                    "reason": _LOW_CONTENT_QUALITY_REASON if degraded_placeholder else "",
                },
            },
            **build_storage_fields_from_rag_meta(rag_meta),
            "needs_review": bool((report.get("safety") or {}).get("needs_review", False))
            or injected
            or bool(rag_meta.get("safe_review", False)),
        },
    )

    return {
        "cached": False,
        "request_id": request_id,
        "ml": ml,
        "report": report,
        "rag_meta": rag_meta,
        "stored": saved,
    }


@router.post("/professor/generate")
async def generate_professor(body: GenerateIn, user: CurrentUser = Depends(get_current_user)):
    _rate_limit(str(user.id))

    if user.role not in {"professor", "admin"}:
        raise HTTPException(status_code=403, detail="Professor access required")

    request_id = str(uuid.uuid4())
    t_all = time.perf_counter()

    file_row = await _load_file(body.file_id, user)

    t0 = time.perf_counter()
    ingestion = await _build_ingestion_bundle(body.file_id, user)
    ingestion_ms = int((time.perf_counter() - t0) * 1000)

    input_hash = _sha256_json(ingestion)
    submission_kind = _detect_submission_kind(ingestion)
    analysis_type = "professor_project_review" if submission_kind == "project" else "professor_academic_review"
    top_k = 6 if submission_kind == "project" else 5
    prompt_hash = _sha256_json(
        {
            "role": "professor",
            "template": "professor_review_v5",
            "analysis_type": analysis_type,
            "query_strategy": f"{submission_kind}_specific_v2",
            "rag_enabled": settings.rag_enabled,
            "top_k": top_k,
        }
    )

    injected = _detect_injection(
        (ingestion.get("text_content") or "") + " " + (ingestion.get("audio_transcript") or "")
    )

    if not body.force:
        existing = await _get_rows(
            "ai_reports"
            f"?file_id=eq.{body.file_id}&role=eq.professor&input_hash=eq.{input_hash}&prompt_hash=eq.{prompt_hash}"
            "&select=*&order=created_at.desc&limit=1"
        )
        if existing:
            return {
                "cached": True,
                "request_id": request_id,
                "stored": _normalize_report_row("professor", existing[0]),
            }

    t1 = time.perf_counter()
    ml = await _call_ai_professor_multimodal(user, ingestion)
    ai_ms = int((time.perf_counter() - t1) * 1000)

    rag_meta: dict[str, Any] = {}
    mode = "restricted" if injected else "normal"
    professor_query = (
        _professor_project_query(ingestion)
        if submission_kind == "project"
        else _professor_academic_query(ingestion)
    )

    llm_payload = {
        "submission_id": str(file_row.get("submission_id") or ""),
        "ingestion": ingestion,
        "ml": {
            "rubric_band": ml["rubric_band"],
            "argument_depth": ml["argument_depth"],
            "moderation_consistency": ml["moderation_consistency"],
        },
        "query": professor_query,
        "top_k": top_k,
        "mode": mode,
        "analysis_type": analysis_type,
        "analysis_focus": (
            [
                "project scope",
                "architecture",
                "technical implementation",
                "security",
                "testing",
                "limitations",
                "moderation risk",
            ]
            if submission_kind == "project"
            else [
                "structure",
                "argument quality",
                "evidence use",
                "critical analysis",
                "clarity",
                "referencing",
                "moderation risk",
            ]
        ),
        "submission_type": submission_kind,
        "official_only": True if (submission_kind == "academic" or settings.rag_require_official_for_professor) else None,
        "safety_flags": {
            "injection_detected": injected,
        },
    }

    if settings.rag_enabled:
        professor_rag_payload = build_professor_rag_payload(llm_payload)
        logger.info(
            "professor_generate rag summary file_id=%s mode=%s query=%r top_k=%s summary=%s",
            body.file_id,
            mode,
            professor_query,
            llm_payload.get("top_k"),
            _rag_log_summary(professor_rag_payload),
        )
        llm_payload = inject_rag_fields(llm_payload, professor_rag_payload)
    else:
        logger.info(
            "professor_generate rag disabled file_id=%s mode=%s query=%r top_k=%s",
            body.file_id,
            mode,
            professor_query,
            llm_payload.get("top_k"),
        )

    t2 = time.perf_counter()
    llm_response, llm_model_used, llm_meta = await _call_llm("/llm/professor/report", llm_payload)
    llm_ms = int((time.perf_counter() - t2) * 1000)

    rag_meta = _extract_rag_meta(llm_response)
    report, llm_ok = _validate_professor_report_for_storage(
        _report_without_rag_meta(llm_response),
        restricted=(mode == "restricted"),
        ml=ml,
        reason="The LLM output did not match the expected professor report schema.",
    )

    total_ms = int((time.perf_counter() - t_all) * 1000)
    agreement = _agreement_score_professor(str(ml["moderation_consistency"]), injected, llm_ok)

    saved = await _post_row(
        "ai_reports",
        {
            "file_id": body.file_id,
            "submission_id": file_row.get("submission_id"),
            "role": "professor",
            "report_json": report,
            "report_hash": _sha256_json(report),
            "prompt_hash": prompt_hash,
            "input_hash": input_hash,
            "model_versions": {
                "request_id": request_id,
                "timings_ms": {
                    "ingestion": ingestion_ms,
                    "ai_service": ai_ms,
                    "llm_service": llm_ms,
                    "total": total_ms,
                },
                "llm_primary": llm_meta.get("primary_model") or settings.llm_primary_label,
                "llm_fallback": llm_meta.get("fallback_model") or settings.llm_fallback_label,
                "llm_model_used": llm_model_used or "unknown",
                "ml_models": {"rubric_suite": "professor.rubric_suite_multimodal.v1"},
                "agreement": {
                    "final_confidence": agreement,
                    "injected": injected,
                    "consistency": ml["moderation_consistency"],
                },
            },
            **build_storage_fields_from_rag_meta(rag_meta),
            "needs_review": bool((report.get("safety") or {}).get("needs_review", False))
            or injected
            or bool(rag_meta.get("safe_review", False)),
        },
    )

    return {
        "cached": False,
        "request_id": request_id,
        "ml": ml,
        "report": report,
        "rag_meta": rag_meta,
        "stored": saved,
    }


@router.get("/history/{role}")
async def history(role: str, limit: int = 30, user: CurrentUser = Depends(get_current_user)):
    if role not in {"student", "professor"}:
        raise HTTPException(status_code=400, detail="Invalid role")
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="Invalid limit")

    select_cols = (
        "id,file_id,submission_id,role,created_at,needs_review,model_versions,"
        "citations,retrieval_confidence,retrieval_confidence_label,safe_review"
    )

    if user.role == "admin":
        rows = await _get_rows(
            f"ai_reports?role=eq.{role}&select={select_cols}&order=created_at.desc&limit={limit}"
        )
        return {"items": rows}

    files = await _get_rows(f"files?user_id=eq.{user.id}&select=id&limit=5000")
    allowed = {str(x["id"]) for x in files}

    if not allowed:
        return {"items": []}

    rows = await _get_rows(
        f"ai_reports?role=eq.{role}&select={select_cols}&order=created_at.desc&limit={min(limit * 5, 200)}"
    )
    rows = [_normalize_report_row(role, r) for r in rows if str(r.get("file_id")) in allowed][:limit]
    return {"items": rows}


@router.get("/compare/{role}")
async def compare(role: str, a: str, b: str, user: CurrentUser = Depends(get_current_user)):
    if role not in {"student", "professor"}:
        raise HTTPException(status_code=400, detail="Invalid role")

    rows = await _get_rows(f"ai_reports?id=in.({a},{b})&select=*")
    if len(rows) != 2:
        raise HTTPException(status_code=404, detail="Reports not found")

    by_id = {str(x.get("id")): x for x in rows}
    r1 = by_id.get(a)
    r2 = by_id.get(b)
    if not r1 or not r2:
        raise HTTPException(status_code=404, detail="Reports not found")

    if str(r1.get("role")) != role or str(r2.get("role")) != role:
        raise HTTPException(status_code=400, detail="Role mismatch")

    if user.role != "admin":
        fids = {str(r1.get("file_id")), str(r2.get("file_id"))}
        files = await _get_rows(f"files?user_id=eq.{user.id}&select=id&limit=5000")
        allowed = {str(x['id']) for x in files}
        if not fids.issubset(allowed):
            raise HTTPException(status_code=404, detail="Not found")

    j1 = (_normalize_report_row(role, r1) or {}).get("report_json") or {}
    j2 = (_normalize_report_row(role, r2) or {}).get("report_json") or {}

    def as_list(x):
        out: list[str] = []
        for i in (x or []):
            if isinstance(i, str):
                s = i.strip()
            elif isinstance(i, dict):
                s = str(i.get("title") or i.get("item") or i.get("action") or i.get("text") or "").strip()
            else:
                s = str(i).strip()
            if s:
                out.append(s)
        return out

    issues1 = set(as_list(j1.get("issues")))
    issues2 = set(as_list(j2.get("issues")))
    chk1 = set(as_list(j1.get("checklist")))
    chk2 = set(as_list(j2.get("checklist")))

    return {
        "a": {"id": r1.get("id"), "created_at": r1.get("created_at"), "file_id": r1.get("file_id")},
        "b": {"id": r2.get("id"), "created_at": r2.get("created_at"), "file_id": r2.get("file_id")},
        "diff": {
            "issues_removed": sorted(list(issues1 - issues2)),
            "issues_added": sorted(list(issues2 - issues1)),
            "checklist_removed": sorted(list(chk1 - chk2)),
            "checklist_added": sorted(list(chk2 - chk1)),
            "summary_a": j1.get("summary") or "",
            "summary_b": j2.get("summary") or "",
        },
    }


@router.get("/professor/queue")
async def professor_queue(limit: int = 30, user: CurrentUser = Depends(get_current_user)):
    if user.role not in {"professor", "admin"}:
        raise HTTPException(status_code=403, detail="Professor access required")

    lim = min(max(limit, 1), 200)
    rows = await _get_rows(
        "ai_reports?role=eq.professor&select=id,file_id,created_at,needs_review,model_versions,"
        "citations,retrieval_confidence,retrieval_confidence_label,safe_review"
        "&order=needs_review.desc,created_at.desc"
        f"&limit={lim}"
    )
    return {"items": rows}
