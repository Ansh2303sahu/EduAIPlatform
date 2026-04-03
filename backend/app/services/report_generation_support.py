from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any, Dict, Literal

import httpx
from fastapi import HTTPException
from pydantic import BaseModel, Field, ValidationError

from app.core.config import settings
from app.core.deps import CurrentUser

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
    response = await _supabase_post(table, payload)
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


def detect_submission_kind(ingestion: dict[str, Any]) -> str:
    haystack = " ".join(
        [
            str(ingestion.get("text_content") or ""),
            str(ingestion.get("audio_transcript") or ""),
            json.dumps(ingestion.get("tables_json") or {}, ensure_ascii=False),
        ]
    ).lower()
    hits = sum(1 for term in _PROJECT_HINT_TERMS if term in haystack)
    return "project" if hits >= 3 else "academic"


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
        for index, item in enumerate(_phase7_as_list(data.get("improvement_plan")))
        if (converted := _as_improvement_object(item, index)) is not None
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
    restricted = _phase7_restricted_mode(report)
    try:
        return _StoredStudentReport.model_validate(report or {}).model_dump()
    except ValidationError:
        try:
            canonical = _canonical_student_report(report, restricted=restricted)
            return _StoredStudentReport.model_validate(canonical).model_dump()
        except ValidationError:
            fallback = _safe_mode_student(
                {},
                reason="A stored student report could not be normalized safely.",
            )
            return _StoredStudentReport.model_validate(fallback).model_dump()


def _normalize_professor_report(report: dict[str, Any] | None) -> dict[str, Any]:
    restricted = _phase7_restricted_mode(report)
    try:
        return _StoredProfessorReport.model_validate(report or {}).model_dump()
    except ValidationError:
        try:
            canonical = _canonical_professor_report(report, restricted=restricted)
            return _StoredProfessorReport.model_validate(canonical).model_dump()
        except ValidationError:
            fallback = _safe_mode_professor(
                {},
                reason="A stored professor report could not be normalized safely.",
            )
            return _StoredProfessorReport.model_validate(fallback).model_dump()


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
    "detect_submission_kind",
    "get_rows",
    "load_file",
    "normalize_report_row",
    "post_row",
    "rate_limit",
    "sha256_json",
]
