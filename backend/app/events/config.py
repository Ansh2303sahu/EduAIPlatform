"""Runtime configuration for the Phase 13 event emitter.

Reads from environment variables.  All n8n-related settings are prefixed
with N8N_ so they stay isolated from application settings.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class EventEmitterSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -- Signing ------------------------------------------------------------------
    # 32-byte hex secret shared with n8n Function nodes.
    n8n_webhook_hmac_secret: str = Field(
        default="",
        description="HMAC-SHA256 signing key for outbound events.",
    )
    n8n_signature_header: str = Field(default="X-EduAI-Signature-256")
    n8n_timestamp_header: str = Field(default="X-EduAI-Timestamp")
    n8n_idempotency_header: str = Field(default="X-EduAI-Idempotency-Key")

    # -- n8n endpoints -----------------------------------------------------------
    n8n_base_url: str = Field(default="http://n8n-main:5678")
    n8n_webhook_path_file_upload: str = Field(default="webhook/file-upload-event")
    n8n_webhook_path_low_confidence: str = Field(default="webhook/low-confidence-alert")
    n8n_webhook_path_pipeline_failure: str = Field(default="webhook/pipeline-failure")

    # -- HTTP behaviour ----------------------------------------------------------
    n8n_emit_timeout_seconds: float = Field(default=10.0, ge=1.0, le=60.0)
    n8n_emit_max_retries: int = Field(default=3, ge=0, le=10)

    # -- Clock skew tolerance when *receiving* events (not used in emitter) ------
    n8n_timestamp_tolerance_seconds: int = Field(default=300, ge=30, le=900)

    # -- HITL workflow webhook ---------------------------------------------------
    n8n_webhook_path_hitl_approval: str = Field(default="webhook/admin-approval-hitl")

    # -- Phase 14: Multi-model assessment webhooks --------------------------------
    n8n_webhook_path_assessment_student: str = Field(
        default="webhook/ai-assessment-student"
    )
    n8n_webhook_path_assessment_professor: str = Field(
        default="webhook/ai-assessment-professor"
    )

    # -- Redis (shared n8n-redis for distributed idempotency + metrics) ----------
    # Use DB 1 to isolate from BullMQ on DB 0.
    # Format: redis://:PASSWORD@HOST:PORT/DB
    redis_url: str = Field(default="redis://:@n8n-redis:6379/1")
    redis_idempotency_ttl_seconds: int = Field(default=86400, ge=60, le=604800)
    # True → 503 when Redis is unreachable (reject unknown events).
    # False (default) → fail-open: treat unavailable Redis as "fresh".
    redis_strict_idempotency: bool = Field(default=False)
    # Retry window for failed-pipeline retry state (seconds)
    redis_retry_window_seconds: int = Field(default=1800, ge=60, le=86400)

    # -- Internal API auth -------------------------------------------------------
    # Shared secret between n8n workflows and the backend /internal/* routes.
    # Generate: openssl rand -hex 32
    backend_internal_secret: str = Field(default="")
    # URL the backend is reachable at from n8n workers on eduaiplatform_internal
    backend_internal_url: str = Field(default="http://backend:8000")

    @property
    def file_upload_url(self) -> str:
        return f"{self.n8n_base_url.rstrip('/')}/{self.n8n_webhook_path_file_upload}"

    @property
    def low_confidence_url(self) -> str:
        return f"{self.n8n_base_url.rstrip('/')}/{self.n8n_webhook_path_low_confidence}"

    @property
    def pipeline_failure_url(self) -> str:
        return f"{self.n8n_base_url.rstrip('/')}/{self.n8n_webhook_path_pipeline_failure}"

    @property
    def hitl_approval_url(self) -> str:
        return f"{self.n8n_base_url.rstrip('/')}/{self.n8n_webhook_path_hitl_approval}"

    @property
    def assessment_student_url(self) -> str:
        return f"{self.n8n_base_url.rstrip('/')}/{self.n8n_webhook_path_assessment_student}"

    @property
    def assessment_professor_url(self) -> str:
        return f"{self.n8n_base_url.rstrip('/')}/{self.n8n_webhook_path_assessment_professor}"


@lru_cache(maxsize=1)
def get_event_settings() -> EventEmitterSettings:
    return EventEmitterSettings()
