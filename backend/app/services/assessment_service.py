"""Phase 14 — Assessment service layer.

Handles:
  - Building rubric context for n8n provider prompting
  - Merging and persisting multi-model assessment results
  - Creating escalation records when the gate fails

All Supabase writes are structured-log stubs with clear TODO markers for
the production insert implementation. The function signatures and return
shapes are production-stable so the router layer never needs to change.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.schemas.assessment import (
    AssessmentAuditRecord,
    AssessmentEscalateIn,
    AssessmentEscalationRecord,
    AssessmentResultIn,
    AssessmentStrength,
    AssessmentIssue,
    FigureAnalysis,
    GeminiExtractionResult,
    ImprovementAction,
    MultiModelAssessmentResult,
    ProviderUsageStats,
    RubricContextIn,
    RubricContextOut,
    RubricCriterion,
    RubricScore,
)
from app.services.assessment_audit_service import write_audit_record

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default rubric used when no professor rubric is configured
# ---------------------------------------------------------------------------

_DEFAULT_STUDENT_RUBRIC: list[RubricCriterion] = [
    RubricCriterion(
        criterion="Technical Accuracy",
        weight=0.30,
        description="Correctness of technical claims, code, and architecture decisions",
    ),
    RubricCriterion(
        criterion="Conceptual Understanding",
        weight=0.25,
        description="Depth of understanding of underlying concepts and theory",
    ),
    RubricCriterion(
        criterion="Implementation Quality",
        weight=0.25,
        description="Code quality, design patterns, error handling, and testing",
    ),
    RubricCriterion(
        criterion="Communication Clarity",
        weight=0.20,
        description="Clarity of explanations, structure, and academic writing quality",
    ),
]

_DEFAULT_PROFESSOR_RUBRIC: list[RubricCriterion] = [
    RubricCriterion(
        criterion="Pedagogical Appropriateness",
        weight=0.30,
        description="Suitability of assessment tasks for the stated learning outcomes",
    ),
    RubricCriterion(
        criterion="Rubric Completeness",
        weight=0.25,
        description="Rubric covers all learning outcomes with clear band descriptors",
    ),
    RubricCriterion(
        criterion="Marking Consistency",
        weight=0.25,
        description="Grading criteria are unambiguous and can be applied consistently",
    ),
    RubricCriterion(
        criterion="Feedback Quality",
        weight=0.20,
        description="Feedback is actionable, specific, and tied to evidence",
    ),
]


# ---------------------------------------------------------------------------
# Rubric context builder
# ---------------------------------------------------------------------------

async def build_rubric_context(
    *,
    file_id: str,
    submission_id: str,
    user_id: str,
    role: str,
    include_draft: bool = True,
) -> RubricContextOut:
    """Fetch rubric definition and Phase 12 ingestion metadata for a file.

    Production implementation should:
      1. Query Supabase ``rubrics`` table for professor-defined criteria
      2. Query Supabase ``ai_reports`` for the Phase 12 draft summary
      3. Query Supabase ``files`` for media metadata (has_images, etc.)
      4. Query Supabase ``ml_results`` for confidence / prediction label

    Currently returns a well-formed stub so n8n workflows can proceed end-to-end.
    The stub uses default rubric criteria and placeholder content values.
    """
    # TODO(phase14): query rubrics table for professor-defined criteria
    rubric = (
        _DEFAULT_STUDENT_RUBRIC if role == "student" else _DEFAULT_PROFESSOR_RUBRIC
    )

    # TODO(phase14): query ai_reports for Phase 12 draft
    draft_summary = ""
    draft_issues: list[dict[str, Any]] = []
    if include_draft:
        draft_summary = (
            "Phase 12 draft report not yet fetched from Supabase. "
            "Claude reviewer: treat this as a new assessment without baseline."
        )

    # TODO(phase14): query files table for media metadata
    has_images = False
    has_tables = False
    has_code_blocks = False

    # TODO(phase14): query ml_results for ML confidence
    ml_confidence = 0.0
    ml_prediction_label = ""

    ctx = RubricContextOut(
        file_id=file_id,
        submission_id=submission_id,
        user_id=user_id,
        role=role,  # type: ignore[arg-type]
        rubric_criteria=rubric,
        rubric_name="Default Academic Rubric",
        rubric_version="1.0",
        content_summary=f"Submission content for file {file_id} — pending Supabase fetch",
        word_count=0,
        ml_confidence=ml_confidence,
        ml_prediction_label=ml_prediction_label,
        has_images=has_images,
        has_tables=has_tables,
        has_code_blocks=has_code_blocks,
        draft_summary=draft_summary,
        draft_issues=draft_issues,
    )

    logger.debug(
        "rubric_context_built",
        extra={
            "file_id": file_id,
            "role": role,
            "criteria_count": len(rubric),
            "has_draft": bool(draft_summary),
        },
    )
    return ctx


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def _apply_claude_corrections(openai_result: Any, claude_review: Any) -> Any:
    """Apply Claude's field corrections to the OpenAI result.

    Corrections specify a ``field_path`` (e.g. ``rubric_scores[0].band``),
    the original value, and the suggested replacement. We apply them
    sequentially; on parse failure we log and skip the bad correction.
    """
    if not claude_review.corrections:
        return openai_result

    result_dict = openai_result.model_dump()

    for correction in claude_review.corrections:
        try:
            # Simple top-level field correction only (nested paths are future work)
            field = correction.field_path.strip()
            if "[" not in field and "." not in field:
                if field in result_dict:
                    result_dict[field] = correction.suggested_value
                    logger.debug(
                        "claude_correction_applied",
                        extra={"field": field, "new_value": str(correction.suggested_value)[:80]},
                    )
        except Exception as exc:
            logger.warning(
                "claude_correction_skipped",
                extra={"field_path": correction.field_path, "error": str(exc)},
            )

    # Rebuild from dict to preserve Pydantic validation
    from app.schemas.assessment import OpenAIAssessmentResult
    return OpenAIAssessmentResult.model_validate(result_dict)


def _merge_gemini_context(
    openai_result: Any,
    gemini: GeminiExtractionResult | None,
) -> tuple[list[FigureAnalysis], list[AssessmentIssue], list[AssessmentStrength], str]:
    """Extract Gemini visual content to merge into the final report."""
    if not gemini or not gemini.multimodal_used:
        return [], list(openai_result.issues), list(openai_result.strengths), ""

    merged_issues = list(openai_result.issues) + gemini.visual_issues
    merged_strengths = list(openai_result.strengths) + gemini.visual_strengths
    visual_context = gemini.additional_context
    return gemini.figures, merged_issues, merged_strengths, visual_context


def _compute_total_cost(body: AssessmentResultIn) -> float:
    total = body.openai_result.usage.cost_usd + body.claude_review.usage.cost_usd
    if body.gemini_extraction and body.gemini_extraction.multimodal_used:
        total += body.gemini_extraction.usage.cost_usd
    return round(total, 6)


async def persist_assessment_result(body: AssessmentResultIn) -> dict[str, Any]:
    """Merge provider outputs and persist the final assessment record.

    Steps:
      1. Apply Claude corrections to OpenAI result
      2. Merge Gemini visual context (if present)
      3. Build MultiModelAssessmentResult
      4. Write AssessmentAuditRecord
      5. TODO: INSERT INTO ai_assessments (Supabase)
      6. Return persisted record summary
    """
    corrected_openai = _apply_claude_corrections(body.openai_result, body.claude_review)

    figures, merged_issues, merged_strengths, visual_context = _merge_gemini_context(
        corrected_openai, body.gemini_extraction
    )

    # Build provider stats map
    provider_stats: dict[str, ProviderUsageStats] = {
        "openai": body.openai_result.usage,
        "claude": body.claude_review.usage,
    }
    if body.gemini_extraction and body.gemini_extraction.multimodal_used:
        provider_stats["gemini"] = body.gemini_extraction.usage

    total_cost = _compute_total_cost(body)

    assessment = MultiModelAssessmentResult(
        file_id=body.file_id,
        submission_id=body.submission_id,
        user_id=body.user_id,
        role=body.role,
        overall_grade=corrected_openai.overall_grade,
        overall_score=corrected_openai.overall_score,
        summary=corrected_openai.summary,
        rubric_scores=corrected_openai.rubric_scores,
        strengths=merged_strengths,
        issues=merged_issues,
        improvement_plan=corrected_openai.improvement_plan,
        figures=figures,
        visual_context=visual_context,
        gate=body.gate_decision,
        provider_stats=provider_stats,
        total_cost_usd=total_cost,
        workflow_version=body.workflow_version,
    )

    # Compute audit stats
    openai_tokens = body.openai_result.usage.total_tokens
    claude_tokens = body.claude_review.usage.total_tokens
    gemini_tokens = (
        body.gemini_extraction.usage.total_tokens
        if body.gemini_extraction and body.gemini_extraction.multimodal_used
        else 0
    )

    audit = AssessmentAuditRecord(
        assessment_id=assessment.assessment_id,
        file_id=body.file_id,
        submission_id=body.submission_id,
        user_id=body.user_id,
        role=body.role,
        openai_invoked=True,
        claude_invoked=True,
        gemini_invoked=bool(body.gemini_extraction and body.gemini_extraction.multimodal_used),
        gate_passed=body.gate_decision.pass_gate,
        hitl_triggered=False,
        final_status="completed",
        openai_latency_ms=body.openai_result.usage.latency_ms,
        claude_latency_ms=body.claude_review.usage.latency_ms,
        gemini_latency_ms=(
            body.gemini_extraction.usage.latency_ms
            if body.gemini_extraction and body.gemini_extraction.multimodal_used
            else 0
        ),
        total_latency_ms=(
            body.openai_result.usage.latency_ms
            + body.claude_review.usage.latency_ms
            + (
                body.gemini_extraction.usage.latency_ms
                if body.gemini_extraction and body.gemini_extraction.multimodal_used
                else 0
            )
        ),
        total_prompt_tokens=openai_tokens + claude_tokens + gemini_tokens,
        total_completion_tokens=(
            body.openai_result.usage.completion_tokens
            + body.claude_review.usage.completion_tokens
        ),
        total_cost_usd=total_cost,
        workflow_version=body.workflow_version,
        n8n_execution_id=body.n8n_execution_id,
        correlation_id=body.correlation_id,
    )

    await write_audit_record(audit)

    # TODO(phase14): INSERT INTO ai_assessments VALUES (assessment.model_dump())
    # via Supabase service-role client. For now: structured log.
    logger.info(
        "assessment_result_built",
        extra={
            "assessment_id": assessment.assessment_id,
            "file_id": body.file_id,
            "overall_score": assessment.overall_score,
            "total_cost_usd": total_cost,
            "corrections_applied": len(body.claude_review.corrections),
            "gemini_used": bool(figures),
        },
    )

    return {
        "assessment_id": assessment.assessment_id,
        "file_id": assessment.file_id,
        "overall_grade": assessment.overall_grade,
        "overall_score": assessment.overall_score,
        "gate_passed": assessment.gate.pass_gate,
        "total_cost_usd": total_cost,
        "created_at": assessment.created_at,
        "persisted": False,  # set True once Supabase INSERT is wired
    }


# ---------------------------------------------------------------------------
# Escalation persistence
# ---------------------------------------------------------------------------

async def persist_escalation(body: AssessmentEscalateIn) -> AssessmentEscalationRecord:
    """Create and persist a HITL escalation record.

    Production: INSERT INTO assessment_escalations (Supabase).
    Currently: structured log + in-memory object.
    """
    severity_map = {
        "critical": "critical",
        "high": "high",
        "medium": "medium",
        "low": "low",
    }
    severity = severity_map.get(body.severity, "medium")

    escalation = AssessmentEscalationRecord(
        file_id=body.file_id,
        submission_id=body.submission_id,
        user_id=body.user_id,
        role=body.role,
        reasons=body.reasons,
        openai_confidence=body.openai_confidence,
        claude_verdict=body.claude_verdict,
        severity=severity,  # type: ignore[arg-type]
        status="pending",
        n8n_resume_url=body.n8n_resume_url,
        n8n_execution_id=body.n8n_execution_id,
    )

    # TODO(phase14): INSERT INTO assessment_escalations (Supabase)
    logger.warning(
        "assessment_escalation_created",
        extra={
            "escalation_id": escalation.escalation_id,
            "file_id": body.file_id,
            "severity": severity,
            "reasons": body.reasons,
            "claude_verdict": body.claude_verdict,
            "openai_confidence": body.openai_confidence,
        },
    )
    return escalation
