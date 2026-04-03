"""
Output normalizers that coerce repaired LLM JSON into canonical report shapes.

The new ``normalize_*_payload`` functions return fully validated Pydantic
models aligned to ``app/langchain/schemas.py``. Older ``normalize_*_output``
wrappers remain for the current Phase 7-compatible pipeline code.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app.langchain.schemas import Phase10ProfessorReport, Phase10StudentReport

StudentReportOut = Phase10StudentReport
ProfessorReportOut = Phase10ProfessorReport


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _truncate(value: Any, limit: int, default: str) -> str:
    if value is None:
        text = default
    elif isinstance(value, str):
        text = value.strip()
    else:
        text = str(value).strip()
    text = text or default
    return text[:limit]


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off", ""}:
            return False
    return default


def _coerce_float(
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


def _coerce_int(
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


def _normalize_severity(value: Any) -> str:
    aliases = {
        "low": "low",
        "minor": "low",
        "med": "med",
        "medium": "med",
        "moderate": "med",
        "high": "high",
        "major": "high",
        "severe": "high",
        "critical": "high",
    }
    normalized = _truncate(value, 20, "low").lower()
    return aliases.get(normalized, "low")


def _is_restricted(value: Any) -> bool:
    data = _as_dict(value)
    confidence = _as_dict(data.get("confidence"))
    safety = _as_dict(data.get("safety"))
    return (
        str(confidence.get("mode") or "").strip().lower() == "restricted"
        or _coerce_bool(safety.get("needs_review"), default=False)
    )


def _as_issue_object(item: Any, index: int) -> dict[str, Any] | None:
    if isinstance(item, str):
        text = item.strip()
        if text:
            return {
                "title": f"Issue {index + 1}",
                "evidence": text[:2000],
                "severity": "med",
            }
        return None
    if not isinstance(item, dict):
        return None

    title = _truncate(
        item.get("title") or item.get("issue") or item.get("label") or item.get("text"),
        200,
        "",
    )
    evidence = _truncate(
        item.get("evidence") or item.get("details") or item.get("description"),
        2000,
        "",
    )
    if not title and not evidence:
        return None
    return {
        "title": title or f"Issue {index + 1}",
        "evidence": evidence or "Evidence was not provided.",
        "severity": _normalize_severity(item.get("severity") or item.get("level") or "med"),
    }


def _as_strength_object(item: Any, index: int) -> dict[str, Any] | None:
    if isinstance(item, str):
        text = item.strip()
        if text:
            return {
                "title": text[:200],
                "evidence": "Supporting evidence was not provided.",
            }
        return None
    if not isinstance(item, dict):
        return None

    title = _truncate(
        item.get("title") or item.get("strength") or item.get("label") or item.get("text"),
        200,
        "",
    )
    evidence = _truncate(
        item.get("evidence") or item.get("details") or item.get("description"),
        2000,
        "",
    )
    if not title and not evidence:
        return None
    return {
        "title": title or f"Strength {index + 1}",
        "evidence": evidence or "Supporting evidence was not provided.",
    }


def _as_improvement_object(item: Any, index: int) -> dict[str, Any] | None:
    if isinstance(item, str):
        text = item.strip()
        if text:
            return {
                "action": text[:300],
                "why": "The reason was not provided.",
                "how": "The implementation detail was not provided.",
                "priority": _coerce_int(index + 1, default=1),
            }
        return None
    if not isinstance(item, dict):
        return None

    action = _truncate(
        item.get("action") or item.get("item") or item.get("step") or item.get("title") or item.get("text"),
        300,
        "",
    )
    why = _truncate(item.get("why") or item.get("reason"), 800, "")
    how = _truncate(item.get("how") or item.get("details"), 800, "")
    if not action and not why and not how:
        return None
    return {
        "action": action or f"Improvement action {index + 1}",
        "why": why or "The reason was not provided.",
        "how": how or "The implementation detail was not provided.",
        "priority": _coerce_int(item.get("priority"), default=index + 1),
    }


def _as_checklist_object(item: Any, index: int) -> dict[str, Any] | None:
    if isinstance(item, str):
        text = item.strip()
        if text:
            return {"item": text[:200], "done": False}
        return None
    if not isinstance(item, dict):
        return None

    text = _truncate(
        item.get("item") or item.get("label") or item.get("title") or item.get("text"),
        200,
        "",
    )
    if not text:
        return None
    return {
        "item": text or f"Checklist item {index + 1}",
        "done": _coerce_bool(item.get("done"), default=_coerce_bool(item.get("checked") or item.get("complete"))),
    }


def _as_rubric_row(item: Any, index: int) -> dict[str, Any] | None:
    if isinstance(item, str):
        text = item.strip()
        if text:
            return {
                "criterion": f"Criterion {index + 1}",
                "band": "Needs review",
                "justification": text[:1200],
            }
        return None
    if not isinstance(item, dict):
        return None

    criterion = _truncate(item.get("criterion") or item.get("title"), 200, "")
    band = _truncate(item.get("band") or item.get("level"), 80, "Needs review")
    justification = _truncate(
        item.get("justification") or item.get("note") or item.get("evidence") or item.get("description"),
        1200,
        "",
    )
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
            return {"risk": f"Risk {index + 1}", "note": text[:800]}
        return None
    if not isinstance(item, dict):
        return None

    risk = _truncate(item.get("risk") or item.get("title") or item.get("label"), 120, "")
    note = _truncate(item.get("note") or item.get("details") or item.get("description"), 800, "")
    if not risk and not note:
        return None
    return {
        "risk": risk or f"Risk {index + 1}",
        "note": note or "A moderation note was not provided.",
    }


def _student_fallback_payload(*, safe_mode: bool) -> dict[str, Any]:
    return {
        "summary": "Automated review could not be fully normalized. Manual review recommended.",
        "issues": [
            {
                "title": "Normalization fallback used",
                "evidence": "The LLM output remained incomplete or malformed after repair and normalization.",
                "severity": "high",
            }
        ],
        "strengths": [],
        "architecture_review": {
            field: "Not assessed."
            for field in ("overview", "backend", "frontend", "database", "security")
        },
        "implementation_review": {
            "features_built": [],
            "technical_quality": "Not assessed.",
            "integration_quality": "Not assessed.",
        },
        "evaluation_review": {
            "testing_present": "Not assessed.",
            "limitations": "Evidence is limited.",
            "academic_quality": "Not assessed.",
        },
        "improvement_plan": [],
        "checklist": [],
        "confidence": {
            "mode": "restricted" if safe_mode else "normal",
            "overall": 0.1 if safe_mode else 0.35,
        },
        "model_agreement": {
            "ml_confidence": 0.0,
            "llm_confidence": 0.0,
            "final_confidence": 0.0,
        },
        "safety": {
            "needs_review": True,
            "reason": "The student payload could not be fully validated and was replaced with a safe fallback.",
        },
    }


def _professor_fallback_payload(*, safe_mode: bool) -> dict[str, Any]:
    return {
        "rubric_breakdown": [
            {
                "criterion": "Overall academic quality",
                "band": "Needs review",
                "justification": "The professor payload remained incomplete or malformed after repair and normalization.",
            }
        ],
        "feedback_explanation": "Automated moderation output could not be fully normalized. Manual review recommended.",
        "moderation_notes": [
            {
                "risk": "Normalization fallback used",
                "note": "The professor payload could not be fully validated and was replaced with a safe fallback.",
            }
        ],
        "safety": {
            "needs_review": True if safe_mode else True,
            "reason": "The professor payload could not be fully validated and was replaced with a safe fallback.",
        },
    }


def _canonical_student_payload(payload: Any, *, safe_mode: bool) -> dict[str, Any]:
    data = _as_dict(payload).copy()
    data.pop("rag_meta", None)

    summary = data.get("summary")
    if (
        isinstance(summary, str)
        and summary.strip() == "The system could not confidently generate full feedback for this submission."
    ):
        summary = "The submission triggered safety or confidence checks, so the system returned limited feedback for manual review."

    issues = [
        converted
        for idx, item in enumerate(_as_list(data.get("issues")))
        if (converted := _as_issue_object(item, idx)) is not None
    ]
    strengths = [
        converted
        for idx, item in enumerate(_as_list(data.get("strengths")))
        if (converted := _as_strength_object(item, idx)) is not None
    ]
    improvement_plan = [
        converted
        for idx, item in enumerate(_as_list(data.get("improvement_plan")))
        if (converted := _as_improvement_object(item, idx)) is not None
    ]
    checklist = [
        converted
        for idx, item in enumerate(_as_list(data.get("checklist")))
        if (converted := _as_checklist_object(item, idx)) is not None
    ]

    architecture_review = _as_dict(data.get("architecture_review"))
    implementation_review = _as_dict(data.get("implementation_review"))
    evaluation_review = _as_dict(data.get("evaluation_review"))
    confidence = _as_dict(data.get("confidence"))
    model_agreement = _as_dict(data.get("model_agreement"))
    safety = _as_dict(data.get("safety"))

    final_confidence = _coerce_float(model_agreement.get("final_confidence"), default=-1.0)
    overall = _coerce_float(confidence.get("overall"), default=-1.0)
    if overall < 0.0:
        overall = final_confidence if final_confidence >= 0.0 else (0.35 if safe_mode else 0.75)

    return {
        "summary": _truncate(summary, 1200, "Automated review generated with limited confidence."),
        "issues": issues,
        "strengths": strengths,
        "architecture_review": {
            "overview": _truncate(architecture_review.get("overview"), 1200, "Not assessed."),
            "backend": _truncate(architecture_review.get("backend"), 1200, "Not assessed."),
            "frontend": _truncate(architecture_review.get("frontend"), 1200, "Not assessed."),
            "database": _truncate(architecture_review.get("database"), 1200, "Not assessed."),
            "security": _truncate(architecture_review.get("security"), 1200, "Not assessed."),
        },
        "implementation_review": {
            "features_built": [
                _truncate(item, 300, "")
                for item in _as_list(implementation_review.get("features_built"))
                if _truncate(item, 300, "")
            ],
            "technical_quality": _truncate(
                implementation_review.get("technical_quality"),
                1200,
                "Not assessed.",
            ),
            "integration_quality": _truncate(
                implementation_review.get("integration_quality"),
                1200,
                "Not assessed.",
            ),
        },
        "evaluation_review": {
            "testing_present": _truncate(evaluation_review.get("testing_present"), 1200, "Not assessed."),
            "limitations": _truncate(evaluation_review.get("limitations"), 1200, "Not assessed."),
            "academic_quality": _truncate(evaluation_review.get("academic_quality"), 1200, "Not assessed."),
        },
        "improvement_plan": improvement_plan,
        "checklist": checklist,
        "confidence": {
            "mode": "restricted" if safe_mode else "normal",
            "overall": overall,
        },
        "model_agreement": {
            "ml_confidence": _coerce_float(model_agreement.get("ml_confidence"), default=0.0),
            "llm_confidence": _coerce_float(model_agreement.get("llm_confidence"), default=0.0),
            "final_confidence": _coerce_float(model_agreement.get("final_confidence"), default=0.0),
        },
        "safety": {
            "needs_review": _coerce_bool(safety.get("needs_review"), default=safe_mode),
            "reason": _truncate(safety.get("reason"), 2000, ""),
        },
    }


def _canonical_professor_payload(payload: Any, *, safe_mode: bool) -> dict[str, Any]:
    data = _as_dict(payload).copy()
    data.pop("rag_meta", None)

    rubric_breakdown = [
        converted
        for idx, item in enumerate(_as_list(data.get("rubric_breakdown")))
        if (converted := _as_rubric_row(item, idx)) is not None
    ]
    moderation_notes = [
        converted
        for idx, item in enumerate(_as_list(data.get("moderation_notes")))
        if (converted := _as_moderation_note(item, idx)) is not None
    ]
    feedback_explanation = _truncate(
        data.get("feedback_explanation") or data.get("summary"),
        1600,
        "Detailed feedback explanation unavailable.",
    )
    safety = _as_dict(data.get("safety"))

    if not rubric_breakdown:
        rubric_breakdown = [
            {
                "criterion": "Overall academic quality",
                "band": "Needs review",
                "justification": _truncate(
                    feedback_explanation,
                    1200,
                    "Structured rubric evidence was not returned by the model.",
                ),
            }
        ]

    return {
        "rubric_breakdown": rubric_breakdown,
        "feedback_explanation": feedback_explanation,
        "moderation_notes": moderation_notes,
        "safety": {
            "needs_review": _coerce_bool(safety.get("needs_review"), default=safe_mode),
            "reason": _truncate(safety.get("reason"), 2000, ""),
        },
    }


def _normalize_student_model(payload: Any, *, safe_mode: bool | None = None) -> StudentReportOut:
    restricted = _is_restricted(payload) if safe_mode is None else safe_mode
    candidate = _canonical_student_payload(payload, safe_mode=restricted)
    try:
        return Phase10StudentReport.model_validate(candidate)
    except ValidationError:
        return Phase10StudentReport.model_validate(_student_fallback_payload(safe_mode=restricted))


def _normalize_professor_model(payload: Any, *, safe_mode: bool | None = None) -> ProfessorReportOut:
    restricted = _is_restricted(payload) if safe_mode is None else safe_mode
    candidate = _canonical_professor_payload(payload, safe_mode=restricted)
    try:
        return Phase10ProfessorReport.model_validate(candidate)
    except ValidationError:
        return Phase10ProfessorReport.model_validate(_professor_fallback_payload(safe_mode=restricted))


def normalize_student_payload(payload: dict[str, Any]) -> StudentReportOut:
    """
    Normalize a student payload into a schema-valid ``Phase10StudentReport``.
    """
    return _normalize_student_model(payload)


def normalize_professor_payload(payload: dict[str, Any]) -> ProfessorReportOut:
    """
    Normalize a professor payload into a schema-valid ``Phase10ProfessorReport``.
    """
    return _normalize_professor_model(payload)


def normalize_student_output(obj: Any, *, safe_mode: bool) -> dict[str, Any]:
    """
    Backward-compatible wrapper returning a canonical student report dict.
    """
    return _normalize_student_model(obj, safe_mode=safe_mode).model_dump(exclude={"rag_meta"})


def normalize_professor_output(obj: Any) -> dict[str, Any]:
    """
    Backward-compatible wrapper returning a canonical professor report dict.
    """
    return _normalize_professor_model(obj).model_dump(exclude={"rag_meta"})
