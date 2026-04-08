"""Validation nodes for Phase 12 and Phase 15/16.

This module intentionally exposes two entry points:
- ``run`` keeps the earlier Phase 12 execution-node contract intact
- ``validation_node`` validates the new Phase 15/16 structured report outputs
"""

from __future__ import annotations

from typing import Any

from app.genai.deterministic import build_professor_report, build_student_report
from app.genai.explainability import confidence_band, rag_evidence_references
from app.genai.schemas import ProfessorModerationReport, StudentReport
from app.langchain.models import ValidationResult
from app.langchain.parsers.json_repair import try_parse_json
from app.langchain.parsers.normalizer import normalize_professor_payload, normalize_student_payload
from app.langchain.parsers.validators import validate_professor_payload, validate_student_payload

from ..schemas import Phase12NodeDescriptor
from ..state import Phase12GraphState
from ._helpers import succeed_node

NODE_NAME = "validation"
NODE_SPEC = Phase12NodeDescriptor(
    name=NODE_NAME,
    description="Validate structured outputs and preserve schema-safe report payloads.",
    wrapped_modules=[
        "app.genai.schemas",
        "app.genai.deterministic",
        "app.langchain.parsers.json_repair",
        "app.langchain.parsers.normalizer",
        "app.langchain.parsers.validators",
    ],
    reads=["draft_report", "repaired_report", "pipeline_context.raw_llm_output"],
    writes=["pipeline_context.report", "pipeline_context.validation_result"],
)


def _sanitize_student_report(state: Phase12GraphState, candidate: StudentReport) -> StudentReport:
    refs = rag_evidence_references(state)
    confidence = candidate.confidence.model_copy(
        update={
            "score": candidate.confidence.score or candidate.confidence_score,
            "band": candidate.confidence.band or confidence_band(candidate.confidence_score),
        }
    )
    issues = candidate.issues or [
        {"title": item[:80], "evidence": item, "severity": "med" if idx < 2 else "low"}
        for idx, item in enumerate(candidate.weaknesses[:4])
    ]
    return candidate.model_copy(
        update={
            "evidence_references": refs or candidate.evidence_references,
            "confidence_score": confidence.score,
            "confidence": confidence,
            "issues": issues,
            "safety": candidate.safety.model_copy(
                update={
                    "needs_review": candidate.safety.needs_review or state.safe_mode or state.rag_result.weak_retrieval,
                }
            ),
        }
    )


def _sanitize_professor_report(
    state: Phase12GraphState,
    candidate: ProfessorModerationReport,
) -> ProfessorModerationReport:
    refs = rag_evidence_references(state)
    confidence = candidate.confidence.model_copy(
        update={
            "score": candidate.confidence.score or candidate.confidence_score,
            "band": candidate.confidence.band or confidence_band(candidate.confidence_score),
        }
    )
    return candidate.model_copy(
        update={
            "evidence_references": refs or candidate.evidence_references,
            "confidence_score": confidence.score,
            "confidence": confidence,
            "safety": candidate.safety.model_copy(
                update={
                    "needs_review": candidate.safety.needs_review or state.safe_mode or state.rag_result.weak_retrieval,
                }
            ),
        }
    )


async def validation_node(state: Phase12GraphState) -> Phase12GraphState:
    """Validate the new structured Phase 15/16 report payloads."""

    state.set_current_node(NODE_NAME)
    candidate_raw: dict[str, Any] = (
        dict(state.repaired_report)
        if state.repaired_report
        else dict(state.draft_report)
    )
    repaired = False
    if state.role == "student":
        try:
            report = _sanitize_student_report(state, StudentReport.model_validate(candidate_raw))
            if not (
                str(report.summary).strip()
                and (report.strengths or report.weaknesses or report.suggestions)
            ):
                raise ValueError("student_report_too_sparse")
        except Exception:
            report = build_student_report(state)
            repaired = True
    else:
        try:
            report = _sanitize_professor_report(state, ProfessorModerationReport.model_validate(candidate_raw))
            if not (
                str(report.summary).strip()
                and str(report.feedback_explanation).strip()
                and (report.strengths or report.weaknesses or report.suggestions)
            ):
                raise ValueError("professor_report_too_sparse")
        except Exception:
            report = build_professor_report(state)
            repaired = True

    if repaired:
        state.add_warning("Structured output was repaired with deterministic validation fallback.")

    state.pipeline_context.report = report.model_dump(mode="json")
    state.pipeline_context.validation_result = ValidationResult.ok(repaired=repaired)
    return succeed_node(
        state,
        node_name=NODE_NAME,
        decision_type="validation",
        branch="validated_repaired" if repaired else "validated",
        reason=(
            "Structured output was validated and lightly repaired."
            if repaired
            else "Structured output passed schema validation."
        ),
        detail={"repaired": repaired, "report_type": state.role},
        confidence=report.confidence.score,
        safe_mode_triggered=state.safe_mode,
    )


async def run(state: Phase12GraphState) -> Phase12GraphState:
    """Legacy Phase 12 validation path retained for the earlier execution graph."""

    raw_text = state.pipeline_context.raw_llm_output
    parser_result = try_parse_json(raw_text)
    if isinstance(parser_result, tuple):
        parsed, error = parser_result
    elif parser_result is None:
        parsed, error = None, "unable_to_parse_json"
    else:
        parsed, error = parser_result, None
    if parsed is None:
        state.add_failure_reason("phase12_validation_parse_failed")
        if error:
            state.add_failure_reason(f"phase12_validation_parse_failed: {error}")
        state.pipeline_context.validation_result = ValidationResult.fail([error or "unable_to_parse_json"])
        return state
    if state.role == "student":
        normalized = normalize_student_payload(parsed)
        validation_result = validate_student_payload(normalized)
    else:
        normalized = normalize_professor_payload(parsed)
        validation_result = validate_professor_payload(normalized)
    state.pipeline_context.validation_result = validation_result
    if validation_result.valid:
        state.pipeline_context.report = normalized
    else:
        state.add_failure_reason("phase12_validation_failed")
    return state
