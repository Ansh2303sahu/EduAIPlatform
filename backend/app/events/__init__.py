"""Phase 13 — event emitter public surface.

Import pattern in application code:

    from app.events import emit_file_upload, emit_low_confidence, emit_pipeline_failure
    from app.events.schemas import FileUploadPayload, LowConfidencePayload, PipelineFailurePayload
    from app.events.emitter import send_event_to_n8n   # for ad-hoc events

Lifespan cleanup (call both in FastAPI shutdown handler):

    from app.events import close_client, close_redis
"""

from .emitter import (
    close_client,
    emit_file_upload,
    emit_low_confidence,
    emit_pipeline_failure,
    send_event_to_n8n,
)
from .idempotency import close_redis, router as internal_router
from .schemas import (
    EmitResult,
    EventType,
    FileUploadPayload,
    LowConfidencePayload,
    OutboundEvent,
    PipelineFailurePayload,
)

__all__ = [
    # emitter
    "send_event_to_n8n",
    "emit_file_upload",
    "emit_low_confidence",
    "emit_pipeline_failure",
    "close_client",
    # internal router (n8n → backend callbacks)
    "internal_router",
    "close_redis",
    # schemas
    "OutboundEvent",
    "EmitResult",
    "EventType",
    "FileUploadPayload",
    "LowConfidencePayload",
    "PipelineFailurePayload",
]
