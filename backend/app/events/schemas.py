"""Typed event schemas for Phase 13 outbound event emission.

Every event emitted to n8n is an ``OutboundEvent`` envelope wrapping a
typed payload.  n8n Function nodes validate the envelope structure and
the HMAC before touching the payload.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    FILE_UPLOAD_COMPLETE = "file.upload.complete"
    AI_LOW_CONFIDENCE = "ai.output.low_confidence"
    AI_PIPELINE_FAILED = "ai.pipeline.failed"
    AI_PIPELINE_COMPLETED = "ai.pipeline.completed"
    AI_ASSESSMENT_REQUESTED = "ai.assessment.requested"  # Phase 14: triggers multi-model pass
    REPORT_PERSISTED = "report.persisted"
    MCP_TOOL_INVOKED = "mcp.tool.invoked"
    SAFE_MODE_ACTIVATED = "ai.safe_mode.activated"


# ---------------------------------------------------------------------------
# Payloads — one per event type, kept minimal and audit-safe
# (no raw file content, no extracted text, no LLM output)
# ---------------------------------------------------------------------------

class FileUploadPayload(BaseModel):
    file_id: str
    user_id: str
    role: str
    file_name: str
    file_size_bytes: int
    mime_type: str
    submission_id: str = ""
    storage_path: str = ""


class LowConfidencePayload(BaseModel):
    model_config = {"protected_namespaces": ()}

    file_id: str
    user_id: str
    role: str
    submission_id: str
    confidence_score: float
    confidence_0_to_4: int
    threshold: float
    model_used: str = ""
    pipeline: str = "phase12_langgraph"
    analysis_type: str = ""


class PipelineFailurePayload(BaseModel):
    file_id: str
    user_id: str
    role: str
    submission_id: str = ""
    failure_reasons: list[str]
    failure_code: str = ""
    pipeline: str = "phase12_langgraph"
    attempt: int = 1
    graph_name: str = ""
    execution_id: str = ""


# ---------------------------------------------------------------------------
# Envelope — wraps every payload sent to n8n
# ---------------------------------------------------------------------------

class OutboundEvent(BaseModel):
    """Wire format for every event sent from FastAPI to n8n webhooks.

    Fields:
      event_id        : UUID, used as idempotency key by n8n.
      event_type      : dot-separated string matching ``EventType`` enum values.
      timestamp       : ISO-8601 UTC emission time.
      correlation_id  : trace ID that spans the originating HTTP request.
      schema_version  : bumped when envelope shape changes.
      payload         : arbitrary dict — typed by ``event_type`` on the sender.
    """

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    correlation_id: str = ""
    schema_version: str = "1.0"
    payload: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def build(
        cls,
        event_type: EventType | str,
        payload: BaseModel | dict[str, Any],
        *,
        correlation_id: str = "",
    ) -> "OutboundEvent":
        et = event_type.value if isinstance(event_type, EventType) else event_type
        raw_payload = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
        return cls(
            event_type=et,
            payload=raw_payload,
            correlation_id=correlation_id,
        )


# ---------------------------------------------------------------------------
# Emit result — returned by the emitter so callers can log/audit
# ---------------------------------------------------------------------------

class EmitResult(BaseModel):
    success: bool
    event_id: str
    event_type: str
    http_status: int | None = None
    attempts: int = 1
    error: str | None = None
