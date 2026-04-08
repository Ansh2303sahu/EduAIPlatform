"""Phase 14 — Assessment audit service.

Writes immutable audit records for every multi-model assessment lifecycle event.
These records support:
  - Compliance / GDPR audit trails
  - Provider billing reconciliation (token usage, cost per file)
  - Debugging failed or escalated assessments
  - Admin dashboards showing model usage over time

Production: each write is an INSERT INTO assessment_audits (Supabase service-role).
Currently: structured JSON log entries. The log format matches the Supabase schema
so that a future migration can ingest logs into the table directly.
"""

from __future__ import annotations

import logging
from typing import Any

from app.schemas.assessment import AssessmentAuditRecord

logger = logging.getLogger(__name__)

# Structured log key used for log-drain → BigQuery / Supabase ingestion
_AUDIT_LOG_KEY = "assessment_audit"


async def write_audit_record(record: AssessmentAuditRecord) -> None:
    """Persist an immutable assessment audit record.

    Production: INSERT INTO assessment_audits using the Supabase service-role
    client (same pattern as existing audit.py service). The record is
    INSERT-only; audits are never updated or deleted.

    Currently emits a structured log line that log-drain tooling can pick up.
    """
    payload = record.model_dump(mode="json")

    # TODO(phase14): INSERT INTO assessment_audits via Supabase service-role
    # Example:
    #   url = f"{settings.supabase_url.rstrip('/')}/rest/v1/assessment_audits"
    #   headers = { "apikey": settings.supabase_service_role_key, ... }
    #   async with httpx.AsyncClient() as client:
    #       await client.post(url, headers=headers, json=payload)

    logger.info(
        _AUDIT_LOG_KEY,
        extra={
            "audit_id": record.audit_id,
            "assessment_id": record.assessment_id,
            "file_id": record.file_id,
            "user_id": record.user_id,
            "role": record.role,
            "final_status": record.final_status,
            "gate_passed": record.gate_passed,
            "hitl_triggered": record.hitl_triggered,
            "openai_invoked": record.openai_invoked,
            "claude_invoked": record.claude_invoked,
            "gemini_invoked": record.gemini_invoked,
            "openai_latency_ms": record.openai_latency_ms,
            "claude_latency_ms": record.claude_latency_ms,
            "gemini_latency_ms": record.gemini_latency_ms,
            "total_latency_ms": record.total_latency_ms,
            "total_prompt_tokens": record.total_prompt_tokens,
            "total_completion_tokens": record.total_completion_tokens,
            "total_cost_usd": record.total_cost_usd,
            "workflow_version": record.workflow_version,
            "n8n_execution_id": record.n8n_execution_id,
            "correlation_id": record.correlation_id,
            "created_at": record.created_at,
        },
    )


async def record_audit_event(
    *,
    file_id: str,
    submission_id: str,
    user_id: str,
    role: str,
    event: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Emit a lightweight lifecycle audit event (non-record form).

    Used for escalation, correction, and gate events that don't map
    directly to the full AssessmentAuditRecord schema.

    Production: INSERT INTO assessment_audit_events (Supabase).
    Currently: structured log.
    """
    # TODO(phase14): INSERT INTO assessment_audit_events
    logger.info(
        f"{_AUDIT_LOG_KEY}.event",
        extra={
            "file_id": file_id,
            "submission_id": submission_id,
            "user_id": user_id,
            "role": role,
            "event": event,
            "metadata": metadata or {},
        },
    )


async def write_provider_usage_summary(
    *,
    date_utc: str,
    openai_calls: int,
    openai_tokens: int,
    openai_cost_usd: float,
    claude_calls: int,
    claude_tokens: int,
    claude_cost_usd: float,
    gemini_calls: int,
    gemini_tokens: int,
    gemini_cost_usd: float,
    total_cost_usd: float,
) -> None:
    """Write a daily provider usage summary for the admin audit workflow.

    Called by the n8n ``admin_model_usage_audit`` workflow which aggregates
    Redis metric counters once per day and submits a summary.

    Production: INSERT INTO assessment_daily_usage (Supabase), upsert on date.
    Currently: structured log.
    """
    # TODO(phase14): UPSERT INTO assessment_daily_usage
    logger.info(
        f"{_AUDIT_LOG_KEY}.daily_usage",
        extra={
            "date_utc": date_utc,
            "openai_calls": openai_calls,
            "openai_tokens": openai_tokens,
            "openai_cost_usd": openai_cost_usd,
            "claude_calls": claude_calls,
            "claude_tokens": claude_tokens,
            "claude_cost_usd": claude_cost_usd,
            "gemini_calls": gemini_calls,
            "gemini_tokens": gemini_tokens,
            "gemini_cost_usd": gemini_cost_usd,
            "total_cost_usd": total_cost_usd,
        },
    )
