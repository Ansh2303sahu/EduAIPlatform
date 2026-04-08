"""Phase 14 — Internal FastAPI router for multi-model assessment callbacks.

All endpoints are protected by ``X-Internal-Secret`` (shared secret).
No JWT auth — these routes live on the Docker-internal network only.
Do NOT expose through the public nginx/Caddy gateway.

Endpoints
---------
POST /internal/assessment/rubric-context
    n8n fetches rubric definition + file context before calling providers.

POST /internal/assessment/submit-result
    n8n submits the merged multi-model result after all providers complete
    and the gate passes.

POST /internal/assessment/escalate
    n8n reports a gate failure; backend creates an escalation record.

POST /internal/assessment/metric
    n8n increments an assessment-specific metric counter in Redis.

POST /internal/assessment/validate-result
    Optional pre-validation of a merged payload without persisting.

GET /internal/assessment/audit/{file_id}
    Admin endpoint: fetch the latest audit record for a file.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.schemas.assessment import (
    AssessmentEscalateIn,
    AssessmentMetricIn,
    AssessmentResultIn,
    AssessmentValidateIn,
    RubricContextIn,
)
from app.services.assessment_audit_service import record_audit_event
from app.services.assessment_service import (
    build_rubric_context,
    persist_assessment_result,
    persist_escalation,
)

# Reuse the shared internal-secret dependency from Phase 13
from app.events.idempotency import _verify_internal, _get_redis, _METRICS_HASH
from app.events.config import get_event_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal/assessment", tags=["internal-assessment"])

_ASSESSMENT_METRICS_PREFIX = "assessment."


# ---------------------------------------------------------------------------
# POST /internal/assessment/rubric-context
# ---------------------------------------------------------------------------

@router.post("/rubric-context")
async def rubric_context(
    body: RubricContextIn,
    _: None = Depends(_verify_internal),
) -> dict[str, Any]:
    """Return rubric definition + file context so n8n can prompt providers.

    This is the first call n8n makes at the start of the assessment workflow.
    It fetches:
      - The rubric criteria from the professor's rubric store (or defaults)
      - Extracted content summary from Phase 12 ingestion metadata
      - ML signals (confidence, prediction label) for provider context
      - Whether the submission contains non-text media (to gate Gemini)
      - The Phase 12 draft report (for Claude's baseline review)
    """
    try:
        ctx = await build_rubric_context(
            file_id=body.file_id,
            submission_id=body.submission_id,
            user_id=body.user_id,
            role=body.role,
            include_draft=body.include_draft,
        )
    except Exception as exc:
        logger.error(
            "rubric_context_build_failed",
            extra={"file_id": body.file_id, "error": str(exc)},
        )
        raise HTTPException(status_code=500, detail=f"Failed to build rubric context: {exc}")

    logger.info(
        "rubric_context_served",
        extra={
            "file_id": body.file_id,
            "role": body.role,
            "has_images": ctx.has_images,
            "criteria_count": len(ctx.rubric_criteria),
        },
    )
    return ctx.model_dump(mode="json")


# ---------------------------------------------------------------------------
# POST /internal/assessment/submit-result
# ---------------------------------------------------------------------------

@router.post("/submit-result")
async def submit_result(
    body: AssessmentResultIn,
    _: None = Depends(_verify_internal),
) -> dict[str, Any]:
    """Persist the final multi-model assessment result.

    n8n calls this after:
      1. OpenAI primary assessment completes
      2. Claude consistency review applies corrections
      3. Gemini multimodal extraction (if triggered) merges in
      4. Gate passes (no escalation required)

    The service layer:
      - Merges OpenAI + corrections from Claude + Gemini visual context
      - Writes a MultiModelAssessmentResult to ai_assessments (Supabase)
      - Writes an AssessmentAuditRecord for billing/compliance
      - Returns the persisted record id for n8n to log
    """
    try:
        result = await persist_assessment_result(body)
    except Exception as exc:
        logger.error(
            "assessment_persist_failed",
            extra={
                "file_id": body.file_id,
                "event_id": body.event_id,
                "error": str(exc),
            },
        )
        raise HTTPException(status_code=500, detail=f"Failed to persist assessment: {exc}")

    # Increment global metric
    try:
        cfg = get_event_settings()
        r = await _get_redis(cfg)
        await r.hincrby(_METRICS_HASH, f"{_ASSESSMENT_METRICS_PREFIX}results_persisted", 1)
        await r.hincrby(_METRICS_HASH, f"{_ASSESSMENT_METRICS_PREFIX}role.{body.role}", 1)
    except Exception:
        pass

    logger.info(
        "assessment_result_persisted",
        extra={
            "assessment_id": result.get("assessment_id"),
            "file_id": body.file_id,
            "role": body.role,
        },
    )
    return result


# ---------------------------------------------------------------------------
# POST /internal/assessment/escalate
# ---------------------------------------------------------------------------

@router.post("/escalate")
async def escalate_assessment(
    body: AssessmentEscalateIn,
    _: None = Depends(_verify_internal),
) -> dict[str, Any]:
    """Create an escalation record when the gate fails.

    Called by n8n when:
      - OpenAI confidence < threshold
      - Claude flags inconsistency and sets overall_verdict = "escalate"
      - OpenAI needs_human_review == True
      - Safety flags are present

    The escalation record captures the resume_url so the admin HITL
    workflow can restart the n8n execution after the admin decides.
    """
    escalation = await persist_escalation(body)

    await record_audit_event(
        file_id=body.file_id,
        submission_id=body.submission_id,
        user_id=body.user_id,
        role=body.role,
        event="escalation_created",
        metadata={
            "escalation_id": escalation.escalation_id,
            "reasons": body.reasons,
            "severity": body.severity,
        },
    )

    try:
        cfg = get_event_settings()
        r = await _get_redis(cfg)
        await r.hincrby(_METRICS_HASH, f"{_ASSESSMENT_METRICS_PREFIX}escalations.created", 1)
        await r.hincrby(
            _METRICS_HASH,
            f"{_ASSESSMENT_METRICS_PREFIX}escalations.severity.{body.severity}",
            1,
        )
    except Exception:
        pass

    logger.warning(
        "assessment_escalated",
        extra={
            "escalation_id": escalation.escalation_id,
            "file_id": body.file_id,
            "severity": body.severity,
            "reasons": body.reasons,
        },
    )
    return {
        "ok": True,
        "escalation_id": escalation.escalation_id,
        "severity": escalation.severity,
    }


# ---------------------------------------------------------------------------
# POST /internal/assessment/metric
# ---------------------------------------------------------------------------

@router.post("/metric")
async def record_metric(
    body: AssessmentMetricIn,
    _: None = Depends(_verify_internal),
) -> dict[str, Any]:
    """Increment an assessment-scoped metric counter in Redis.

    Uses the same Redis metrics hash as Phase 13 but prefixes all keys
    with ``assessment.`` to keep dashboards segmented.
    """
    scoped_metric = f"{_ASSESSMENT_METRICS_PREFIX}{body.metric}"
    cfg = get_event_settings()
    try:
        r = await _get_redis(cfg)
        new_value = await r.hincrby(_METRICS_HASH, scoped_metric, body.value)
    except Exception as exc:
        logger.warning(
            "assessment_metric_failed",
            extra={"metric": scoped_metric, "error": str(exc)},
        )
        new_value = -1

    return {"ok": True, "metric": scoped_metric, "new_value": new_value}


# ---------------------------------------------------------------------------
# POST /internal/assessment/validate-result
# ---------------------------------------------------------------------------

@router.post("/validate-result")
async def validate_result(
    body: AssessmentValidateIn,
    _: None = Depends(_verify_internal),
) -> dict[str, Any]:
    """Pre-validate a merged payload without persisting it.

    n8n can call this before submit-result to catch schema errors early
    and surface them in the workflow execution log rather than as a 500.
    Returns a list of validation warnings (non-fatal issues).
    """
    warnings: list[str] = []

    # Confidence consistency check
    if body.openai_result.confidence < 0.40 and not body.gate_decision.escalate:
        warnings.append(
            f"OpenAI confidence {body.openai_result.confidence:.2f} < 0.40 "
            "but gate.escalate is False — gate logic may be misconfigured"
        )

    # Claude correction check
    if body.claude_review.overall_verdict == "escalate" and not body.gate_decision.hitl_required:
        warnings.append(
            "Claude verdict is 'escalate' but gate.hitl_required is False — "
            "HITL will not be triggered despite Claude's recommendation"
        )

    # Rubric score completeness
    if not body.openai_result.rubric_scores:
        warnings.append("openai_result.rubric_scores is empty — no rubric-level grading")

    # Gemini check
    if body.gemini_extraction and not body.gemini_extraction.multimodal_used:
        if not body.gemini_extraction.skip_reason:
            warnings.append(
                "gemini_extraction.multimodal_used=False but skip_reason is empty"
            )

    logger.debug(
        "assessment_validate_result",
        extra={"warnings": len(warnings), "gate_passed": body.gate_decision.pass_gate},
    )
    return {"valid": True, "warnings": warnings}


# ---------------------------------------------------------------------------
# GET /internal/assessment/audit/{file_id}
# ---------------------------------------------------------------------------

@router.get("/audit/{file_id}")
async def get_audit(
    file_id: str,
    _: None = Depends(_verify_internal),
) -> dict[str, Any]:
    """Return the latest audit record for a given file_id.

    Production: SELECT * FROM assessment_audits WHERE file_id=? ORDER BY created_at DESC LIMIT 1
    Currently: structured log response (Supabase insert is a stub in the service).
    """
    logger.info("assessment_audit_read", extra={"file_id": file_id})
    # Stub — in production this queries Supabase assessment_audits table
    return {
        "file_id": file_id,
        "note": "Supabase query not yet wired — audit records are in structured logs",
    }
