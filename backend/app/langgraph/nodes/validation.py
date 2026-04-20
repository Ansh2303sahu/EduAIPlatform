"""Validation nodes for Phase 12 and Phase 15/16.

This module intentionally exposes two entry points:
- ``run`` keeps the earlier Phase 12 execution-node contract intact
- ``validation_node`` validates the new Phase 15/16 structured report outputs
"""

from __future__ import annotations

import re
from collections import Counter
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


def _as_report_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude={"rag_meta"})
    return dict(value or {}) if isinstance(value, dict) else {}


def _unit_float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0.0:
        return None
    return max(0.0, min(1.0, number))


def _first_unit_float(*values: Any) -> float | None:
    for value in values:
        score = _unit_float_or_none(value)
        if score is not None:
            return score
    return None


def _positive_scores(*values: Any) -> list[float]:
    return [
        score
        for value in values
        for score in [_unit_float_or_none(value)]
        if score is not None and score > 0.0
    ]


def _needs_review(state: Phase12GraphState, report: dict[str, Any]) -> bool:
    safety = report.get("safety") if isinstance(report.get("safety"), dict) else {}
    return bool(
        state.safe_mode
        or state.rag_result.weak_retrieval
        or safety.get("needs_review")
    )


_GENERIC_CONFIDENCE_PHRASES = (
    "promising work",
    "clearer support",
    "more evidence needed",
    "needs more support",
    "could be stronger",
    "shows some promising work",
)
_ANCHOR_STOPWORDS = {
    "this", "that", "with", "from", "into", "their", "there", "which", "about",
    "because", "using", "essay", "report", "student", "submission", "section",
    "paragraph", "evidence", "analysis", "argument", "introduction", "conclusion",
    "background", "context", "structure", "clarity", "support", "writing",
}


def _rag_confidence_cap(state: Phase12GraphState) -> float:
    rag_score = _unit_float_or_none(state.rag_result.confidence_score)
    if rag_score is None or rag_score <= 0.0:
        rag_score = _unit_float_or_none(state.evidence_quality_score) or 0.0
    if state.rag_result.weak_retrieval:
        rag_score = min(rag_score, 0.2)
    return min(1.0, rag_score + 0.15)


def _flatten_report_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(_flatten_report_text(item) for item in value.values()).strip()
    if isinstance(value, list):
        return " ".join(_flatten_report_text(item) for item in value).strip()
    return str(value).strip()


def _submission_anchor_terms(state: Phase12GraphState) -> set[str]:
    keywords = [
        str(item).strip().lower()
        for item in (state.rag_result.trace or {}).get("keywords_used", [])
        if str(item).strip()
    ]
    if keywords:
        return {
            token
            for keyword in keywords[:8]
            for token in re.findall(r"[a-z][a-z0-9/-]{2,}", keyword)
            if token not in _ANCHOR_STOPWORDS
        }

    text = " ".join(
        part
        for part in (
            str(state.pipeline_context.ingestion.text_content or ""),
            str(state.pipeline_context.ingestion.ocr_text or ""),
            str(state.pipeline_context.ingestion.audio_transcript or ""),
        )
        if part
    ).lower()
    counts = Counter(
        token
        for token in re.findall(r"[a-z][a-z0-9/-]{3,}", text[:8000])
        if token not in _ANCHOR_STOPWORDS
    )
    return {term for term, _count in counts.most_common(10)}


def _generic_language_penalty(state: Phase12GraphState, report: dict[str, Any]) -> float:
    text = _flatten_report_text(report).lower()
    if not text:
        return 0.0

    generic_hits = sum(1 for phrase in _GENERIC_CONFIDENCE_PHRASES if phrase in text)
    if generic_hits == 0:
        return 0.0

    anchor_terms = _submission_anchor_terms(state)
    anchor_hits = sum(1 for term in anchor_terms if term and term in text)
    quoted_markers = text.count('"') + text.count("'")

    if generic_hits >= 2 and anchor_hits == 0 and quoted_markers == 0:
        return 0.12
    if generic_hits >= 1 and anchor_hits <= 1:
        return 0.08
    return 0.0


def _calibrate_student_report_confidence(
    state: Phase12GraphState,
    report: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    """Carry graph-owned ML confidence into the student report agreement block."""

    calibrated = dict(report)
    confidence = dict(calibrated.get("confidence") or {})
    agreement = dict(calibrated.get("model_agreement") or {})

    ml_confidence = _unit_float_or_none(state.ml_confidence) or 0.0
    llm_confidence = _first_unit_float(
        agreement.get("llm_confidence"),
        confidence.get("overall"),
        confidence.get("score"),
    )
    final_confidence = _first_unit_float(
        agreement.get("final_confidence"),
        confidence.get("overall"),
        confidence.get("score"),
    )

    if ml_confidence > 0.0:
        agreement["ml_confidence"] = ml_confidence
    else:
        agreement["ml_confidence"] = _unit_float_or_none(agreement.get("ml_confidence")) or 0.0

    if llm_confidence is None or llm_confidence <= 0.0:
        fallback_scores = _positive_scores(
            confidence.get("overall"),
            confidence.get("score"),
            final_confidence,
            ml_confidence,
            state.evidence_quality_score,
        )
        llm_confidence = fallback_scores[0] if fallback_scores else (0.35 if _needs_review(state, calibrated) else 0.65)
    agreement["llm_confidence"] = llm_confidence

    rag_cap = _rag_confidence_cap(state)
    caps = [llm_confidence, rag_cap]
    if agreement["ml_confidence"] > 0.0:
        caps.append(agreement["ml_confidence"])
    bounded_confidence = min(caps) if caps else 0.0
    if final_confidence is None or final_confidence <= 0.0:
        candidates = _positive_scores(
            confidence.get("overall"),
            confidence.get("score"),
            bounded_confidence,
            state.evidence_quality_score,
        )
        final_confidence = candidates[0] if candidates else bounded_confidence
    final_confidence = min(final_confidence, bounded_confidence)

    generic_penalty = _generic_language_penalty(state, calibrated)
    if generic_penalty > 0.0:
        final_confidence = max(0.0, final_confidence - generic_penalty)

    if _needs_review(state, calibrated) and final_confidence > 0.45:
        final_confidence = 0.45

    final_confidence = round(final_confidence, 3)
    agreement["final_confidence"] = final_confidence
    overall = _first_unit_float(confidence.get("overall"), final_confidence) or final_confidence
    if overall > final_confidence:
        overall = final_confidence
    confidence["overall"] = overall

    calibrated["confidence"] = confidence
    calibrated["model_agreement"] = agreement
    return calibrated, final_confidence


def _calibrate_professor_agreement(
    state: Phase12GraphState,
    report: dict[str, Any],
) -> float:
    """Professor reports do not carry model_agreement, so store it in execution meta."""

    confidence = report.get("confidence") if isinstance(report.get("confidence"), dict) else {}
    explicit = _first_unit_float(
        report.get("confidence_score"),
        confidence.get("overall"),
        confidence.get("score"),
    )
    if explicit is not None and explicit > 0.0:
        score = explicit
    else:
        candidates = _positive_scores(
            state.ml_confidence,
            state.rag_result.confidence_score,
            state.evidence_quality_score,
        )
        if candidates:
            score = round(sum(candidates) / len(candidates), 3)
        elif report.get("rubric_breakdown") or report.get("feedback_explanation"):
            score = 0.5
        else:
            score = 0.0

    if _needs_review(state, report) and score > 0.45:
        score = 0.45
    return score


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


def _calibrate_structured_student_report(
    state: Phase12GraphState,
    report: StudentReport,
) -> tuple[StudentReport, float]:
    llm_confidence = _first_unit_float(report.confidence.score, report.confidence_score)
    if llm_confidence is None or llm_confidence <= 0.0:
        llm_confidence = _first_unit_float(state.evidence_quality_score, state.ml_confidence) or 0.35

    ml_confidence = _unit_float_or_none(state.ml_confidence)
    caps = [llm_confidence, _rag_confidence_cap(state)]
    if ml_confidence is not None and ml_confidence > 0.0:
        caps.append(ml_confidence)
    final_confidence = min(caps) if caps else 0.0
    final_confidence -= _generic_language_penalty(state, report.model_dump(mode="json"))
    final_confidence = max(0.0, round(final_confidence, 3))

    if _needs_review(state, report.model_dump(mode="json")) and final_confidence > 0.45:
        final_confidence = 0.45

    calibrated = report.model_copy(
        update={
            "confidence_score": final_confidence,
            "confidence": report.confidence.model_copy(
                update={
                    "score": final_confidence,
                    "band": confidence_band(final_confidence),
                    "rationale": (
                        report.confidence.rationale
                        or "Confidence was capped by submission evidence, ML calibration, and retrieval grounding quality."
                    ),
                }
            ),
        }
    )
    return calibrated, final_confidence


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
            report, confidence_score = _calibrate_structured_student_report(state, report)
            if not (
                str(report.summary).strip()
                and (report.strengths or report.weaknesses or report.suggestions)
            ):
                raise ValueError("student_report_too_sparse")
        except Exception:
            report = build_student_report(state)
            report, confidence_score = _calibrate_structured_student_report(state, report)
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
        confidence_score = report.confidence_score

    if repaired:
        state.add_warning("Structured output was repaired with deterministic validation fallback.")

    state.pipeline_context.report = report.model_dump(mode="json")
    state.pipeline_context.validation_result = ValidationResult.ok(repaired=repaired)
    state.pipeline_context.execution_meta.agreement_score = confidence_score
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
        confidence=confidence_score,
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
        normalized = _as_report_dict(normalize_student_payload(parsed))
        normalized, agreement_score = _calibrate_student_report_confidence(state, normalized)
        validation_result = validate_student_payload(normalized)
    else:
        normalized = _as_report_dict(normalize_professor_payload(parsed))
        agreement_score = _calibrate_professor_agreement(state, normalized)
        validation_result = validate_professor_payload(normalized)
    state.pipeline_context.validation_result = validation_result
    if validation_result.valid:
        state.pipeline_context.report = normalized
        state.pipeline_context.execution_meta.agreement_score = agreement_score
    else:
        state.add_failure_reason("phase12_validation_failed")
    return state
