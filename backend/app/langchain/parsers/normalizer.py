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
                "title": text[:200],
                "evidence": text[:2000],
                "detail": text[:2000],
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
        "detail": _truncate(item.get("detail") or evidence, 2000, evidence or "Evidence was not provided."),
        "severity": _normalize_severity(item.get("severity") or item.get("level") or "med"),
    }


def _as_strength_object(item: Any, index: int) -> dict[str, Any] | None:
    if isinstance(item, str):
        text = item.strip()
        if text:
            return {
                "title": text[:200],
                "evidence": "Supporting evidence was not provided.",
                "detail": text[:2000],
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
        "detail": _truncate(item.get("detail") or evidence, 2000, evidence or "Supporting evidence was not provided."),
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
    why = _truncate(item.get("why") or item.get("reason") or item.get("rationale"), 800, "")
    steps = [str(step).strip() for step in _as_list(item.get("steps")) if str(step).strip()]
    how = _truncate(
        item.get("how") or item.get("details") or item.get("description") or "; ".join(steps),
        800,
        "",
    )
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


def _as_text_row(item: Any, index: int, *, prefix: str = "Item") -> str:
    if isinstance(item, str):
        return _truncate(item, 400, "")
    if not isinstance(item, dict):
        return ""
    return _truncate(
        item.get("text")
        or item.get("title")
        or item.get("label")
        or item.get("criterion")
        or item.get("action")
        or item.get("item")
        or f"{prefix} {index + 1}",
        400,
        "",
    )


def _as_section_feedback_object(item: Any, index: int) -> dict[str, Any] | None:
    if isinstance(item, str):
        text = _truncate(item, 1600, "")
        if not text:
            return None
        return {
            "section_name": f"Section {index + 1}",
            "what_works": "",
            "what_needs_improvement": text,
            "recommended_fix": "",
        }
    if not isinstance(item, dict):
        return None
    section_name = _truncate(item.get("section_name") or item.get("section") or item.get("name"), 200, "")
    what_works = _truncate(item.get("what_works") or item.get("strength") or item.get("observation"), 1200, "")
    needs_improvement = _truncate(
        item.get("what_needs_improvement") or item.get("weakness") or item.get("concern") or item.get("detail"),
        1200,
        "",
    )
    recommended_fix = _truncate(item.get("recommended_fix") or item.get("next_step") or item.get("fix"), 1200, "")
    if not any((section_name, what_works, needs_improvement, recommended_fix)):
        return None
    return {
        "section_name": section_name or f"Section {index + 1}",
        "what_works": what_works,
        "what_needs_improvement": needs_improvement,
        "recommended_fix": recommended_fix,
    }


def _as_priority_issue(item: Any, *, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(item, dict):
        title = _truncate(item.get("title"), 200, "")
        why_it_matters = _truncate(item.get("why_it_matters") or item.get("detail") or item.get("evidence"), 1200, "")
        how_to_fix_it = _truncate(item.get("how_to_fix_it") or item.get("recommended_fix") or item.get("how"), 1200, "")
        if title or why_it_matters or how_to_fix_it:
            return {
                "title": title,
                "why_it_matters": why_it_matters,
                "how_to_fix_it": how_to_fix_it,
            }
    if fallback:
        return {
            "title": _truncate(fallback.get("title"), 200, ""),
            "why_it_matters": _truncate(fallback.get("detail") or fallback.get("evidence"), 1200, ""),
            "how_to_fix_it": "",
        }
    return {
        "title": "",
        "why_it_matters": "",
        "how_to_fix_it": "",
    }


def _as_section_observation(item: Any, index: int) -> dict[str, Any] | None:
    if isinstance(item, str):
        text = _truncate(item, 1600, "")
        if not text:
            return None
        return {
            "section_name": f"Section {index + 1}",
            "observation": text,
            "concern": "",
            "next_step": "",
        }
    if not isinstance(item, dict):
        return None
    section_name = _truncate(item.get("section_name") or item.get("section") or item.get("name"), 200, "")
    observation = _truncate(item.get("observation") or item.get("what_works") or item.get("summary"), 1200, "")
    concern = _truncate(item.get("concern") or item.get("what_needs_improvement") or item.get("detail"), 1200, "")
    next_step = _truncate(item.get("next_step") or item.get("recommended_fix") or item.get("action"), 1200, "")
    if not any((section_name, observation, concern, next_step)):
        return None
    return {
        "section_name": section_name or f"Section {index + 1}",
        "observation": observation,
        "concern": concern,
        "next_step": next_step,
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
        "overall_judgment": "Evidence is too limited for a reliable automated judgment.",
        "issues": [
            {
                "title": "Normalization fallback used",
                "evidence": "The LLM output remained incomplete or malformed after repair and normalization.",
                "detail": "The LLM output remained incomplete or malformed after repair and normalization.",
                "severity": "high",
            }
        ],
        "strengths": [],
        "weaknesses": [],
        "section_feedback": [],
        "priority_issue": {
            "title": "Manual review recommended",
            "why_it_matters": "The automated output could not be normalized safely.",
            "how_to_fix_it": "Review the submission manually or regenerate after improving the input evidence.",
        },
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
        "confidence_explanation": "Confidence was lowered because normalization and repair could not recover a reliable report.",
        "evidence_coverage": "Evidence coverage is insufficient because the normalized report fell back to a safe minimal structure.",
        "grounding_summary": "Grounded detail could not be preserved during fallback normalization.",
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
        "summary": "Automated moderation output could not be fully normalized. Manual review recommended.",
        "evaluator_overview": "Evidence is too limited for a reliable moderation judgment.",
        "rubric_breakdown": [
            {
                "criterion": "Overall academic quality",
                "band": "Needs review",
                "justification": "The professor payload remained incomplete or malformed after repair and normalization.",
            }
        ],
        "rubric_alignment": [],
        "feedback_explanation": "Automated moderation output could not be fully normalized. Manual review recommended.",
        "strengths": [],
        "concerns": [],
        "weaknesses": [],
        "section_observations": [],
        "marking_considerations": [],
        "moderation_notes": [
            {
                "risk": "Normalization fallback used",
                "note": "The professor payload could not be fully validated and was replaced with a safe fallback.",
            }
        ],
        "action_recommendations": [],
        "confidence_explanation": "Confidence was lowered because normalization and repair could not recover a reliable moderation report.",
        "evidence_coverage": "Evidence coverage is insufficient because the normalized report fell back to a safe minimal structure.",
        "grounding_summary": "Grounded detail could not be preserved during fallback normalization.",
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
    weaknesses = [
        converted
        for idx, item in enumerate(_as_list(data.get("weaknesses") or data.get("issues")))
        if (converted := _as_issue_object(item, idx)) is not None
    ]
    strengths = [
        converted
        for idx, item in enumerate(_as_list(data.get("strengths")))
        if (converted := _as_strength_object(item, idx)) is not None
    ]
    raw_improvement_items = _as_list(data.get("improvement_plan") or data.get("suggestions"))
    if len(raw_improvement_items) == 1 and isinstance(raw_improvement_items[0], dict):
        nested_actions = _as_list(raw_improvement_items[0].get("actions"))
        if nested_actions:
            raw_improvement_items = nested_actions
    improvement_plan = [
        converted
        for idx, item in enumerate(raw_improvement_items)
        if (converted := _as_improvement_object(item, idx)) is not None
    ]
    learning_path = _as_dict(data.get("learning_path"))
    raw_checklist_items = _as_list(data.get("checklist"))
    if not raw_checklist_items:
        raw_checklist_items = _as_list(learning_path.get("recommended_practice"))
    checklist = [
        converted
        for idx, item in enumerate(raw_checklist_items)
        if (converted := _as_checklist_object(item, idx)) is not None
    ]
    section_feedback = [
        converted
        for idx, item in enumerate(_as_list(data.get("section_feedback")))
        if (converted := _as_section_feedback_object(item, idx)) is not None
    ]

    architecture_review = _as_dict(data.get("architecture_review"))
    implementation_review = _as_dict(data.get("implementation_review"))
    evaluation_review = _as_dict(data.get("evaluation_review"))
    confidence = _as_dict(data.get("confidence"))
    model_agreement = _as_dict(data.get("model_agreement"))
    safety = _as_dict(data.get("safety"))
    priority_issue = _as_priority_issue(data.get("priority_issue"), fallback=(weaknesses or issues or [{}])[0])

    final_confidence = _coerce_float(model_agreement.get("final_confidence"), default=-1.0)
    overall = _coerce_float(confidence.get("overall"), default=-1.0)
    if overall < 0.0:
        overall = _coerce_float(confidence.get("score"), default=-1.0)
    if overall < 0.0:
        overall = final_confidence if final_confidence >= 0.0 else (0.35 if safe_mode else 0.75)
    if final_confidence < 0.0:
        final_confidence = overall
    llm_confidence = _coerce_float(model_agreement.get("llm_confidence"), default=-1.0)
    if llm_confidence < 0.0:
        llm_confidence = final_confidence
    ml_confidence = _coerce_float(model_agreement.get("ml_confidence"), default=-1.0)
    if ml_confidence < 0.0:
        ml_confidence = overall
    canonical_issues = issues or list(weaknesses)

    return {
        "summary": _truncate(summary, 1200, "Automated review generated with limited confidence."),
        "overall_judgment": _truncate(data.get("overall_judgment") or summary, 1200, ""),
        "issues": canonical_issues,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "section_feedback": section_feedback,
        "priority_issue": priority_issue,
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
        "confidence_explanation": _truncate(
            data.get("confidence_explanation")
            or data.get("grounding_summary")
            or data.get("evidence_coverage"),
            2000,
            "",
        ),
        "evidence_coverage": _truncate(data.get("evidence_coverage"), 2000, ""),
        "grounding_summary": _truncate(data.get("grounding_summary"), 2000, ""),
        "model_agreement": {
            "ml_confidence": ml_confidence,
            "llm_confidence": llm_confidence,
            "final_confidence": final_confidence,
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
    strengths = [
        converted
        for idx, item in enumerate(_as_list(data.get("strengths")))
        if (converted := _as_strength_object(item, idx)) is not None
    ]
    concerns = [
        converted
        for idx, item in enumerate(_as_list(data.get("concerns") or data.get("weaknesses")))
        if (converted := _as_issue_object(item, idx)) is not None
    ]
    weaknesses = [
        converted
        for idx, item in enumerate(_as_list(data.get("weaknesses") or data.get("concerns")))
        if (converted := _as_issue_object(item, idx)) is not None
    ]
    section_observations = [
        converted
        for idx, item in enumerate(_as_list(data.get("section_observations")))
        if (converted := _as_section_observation(item, idx)) is not None
    ]
    rubric_alignment = [
        text
        for idx, item in enumerate(_as_list(data.get("rubric_alignment")))
        if (text := _as_text_row(item, idx, prefix="Rubric alignment"))
    ]
    marking_considerations = [
        text
        for idx, item in enumerate(_as_list(data.get("marking_considerations")))
        if (text := _as_text_row(item, idx, prefix="Marking consideration"))
    ]
    action_recommendations = [
        text
        for idx, item in enumerate(_as_list(data.get("action_recommendations")))
        if (text := _as_text_row(item, idx, prefix="Action recommendation"))
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
        "summary": _truncate(data.get("summary") or feedback_explanation, 1200, ""),
        "evaluator_overview": _truncate(data.get("evaluator_overview") or data.get("summary"), 1600, ""),
        "rubric_breakdown": rubric_breakdown,
        "rubric_alignment": rubric_alignment,
        "feedback_explanation": feedback_explanation,
        "strengths": strengths,
        "concerns": concerns,
        "weaknesses": weaknesses,
        "section_observations": section_observations,
        "marking_considerations": marking_considerations,
        "moderation_notes": moderation_notes,
        "action_recommendations": action_recommendations,
        "confidence_explanation": _truncate(
            data.get("confidence_explanation")
            or data.get("grounding_summary")
            or data.get("evidence_coverage"),
            2000,
            "",
        ),
        "evidence_coverage": _truncate(data.get("evidence_coverage"), 2000, ""),
        "grounding_summary": _truncate(data.get("grounding_summary"), 2000, ""),
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
