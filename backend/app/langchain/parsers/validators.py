"""
Pydantic validators for repaired and normalised LLM payloads.

The new ``validate_*_payload`` functions expose ``ValidationResult`` for the
pipeline layer, while the older ``validate_*_report`` wrappers remain for the
existing Phase 7-compatible pipeline code paths.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app.langchain.models import ValidationResult
from app.langchain.schemas import Phase10ProfessorReport, Phase10StudentReport


def _error_messages(exc: ValidationError) -> list[str]:
    messages: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ()))
        prefix = f"{location}: " if location else ""
        messages.append(prefix + str(error.get("msg", "validation error")))
    return messages or [str(exc)]


def validate_student_payload(payload: dict[str, Any]) -> ValidationResult:
    """
    Validate a student payload against the canonical student report schema.
    """
    try:
        Phase10StudentReport.model_validate(payload or {})
        return ValidationResult.ok()
    except ValidationError as exc:
        return ValidationResult.fail(_error_messages(exc))


def validate_professor_payload(payload: dict[str, Any]) -> ValidationResult:
    """
    Validate a professor payload against the canonical professor report schema.
    """
    try:
        Phase10ProfessorReport.model_validate(payload or {})
        return ValidationResult.ok()
    except ValidationError as exc:
        return ValidationResult.fail(_error_messages(exc))


def validate_student_report(data: dict[str, Any]) -> tuple[Phase10StudentReport, bool]:
    """
    Backward-compatible wrapper used by the current student pipeline.
    """
    try:
        return Phase10StudentReport.model_validate(data or {}), True
    except ValidationError:
        fallback = Phase10StudentReport(
            summary="Automated review could not be fully validated. Manual review recommended.",
            safety={"needs_review": True, "reason": "The student payload failed schema validation."},
        )
        return fallback, False


def validate_professor_report(data: dict[str, Any]) -> tuple[Phase10ProfessorReport, bool]:
    """
    Backward-compatible wrapper used by the current professor pipeline.
    """
    try:
        return Phase10ProfessorReport.model_validate(data or {}), True
    except ValidationError:
        fallback = Phase10ProfessorReport(
            feedback_explanation="Automated moderation review could not be fully validated. Manual review recommended.",
            rubric_breakdown=[
                {
                    "criterion": "Overall academic quality",
                    "band": "Needs review",
                    "justification": "The professor payload failed schema validation.",
                }
            ],
            safety={"needs_review": True, "reason": "The professor payload failed schema validation."},
        )
        return fallback, False
