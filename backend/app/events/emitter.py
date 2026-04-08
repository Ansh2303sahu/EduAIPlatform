"""Async event emitter — Phase 13 FastAPI → n8n integration.

Design
------
- All I/O is non-blocking (httpx.AsyncClient).
- Failures are soft by default: emit errors are logged and returned as
  ``EmitResult(success=False, ...)`` so the calling request is never
  failed due to an automation system being temporarily unreachable.
- Retries use exponential back-off with jitter (not a straight fixed delay).
- A module-level ``AsyncClient`` is reused across calls for connection pooling.
  Call ``close_client()`` in the FastAPI lifespan shutdown handler.

Usage
-----
    from app.events.emitter import emit_file_upload, emit_low_confidence, emit_pipeline_failure
    from app.events.schemas import FileUploadPayload

    result = await emit_file_upload(
        payload=FileUploadPayload(
            file_id=file_id,
            user_id=user_id,
            role="student",
            ...
        ),
        correlation_id=request_id,
    )
    if not result.success:
        logger.warning("n8n emit failed", extra={"event_id": result.event_id, "error": result.error})

    # Alternatively, emit any arbitrary event type:
    result = await send_event_to_n8n(
        event_type=EventType.SAFE_MODE_ACTIVATED,
        payload={"file_id": "...", "reason": "low_confidence"},
        webhook_url=settings.low_confidence_url,
        correlation_id=request_id,
    )
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

import httpx

from .config import EventEmitterSettings, get_event_settings
from .schemas import (
    EmitResult,
    EventType,
    FileUploadPayload,
    LowConfidencePayload,
    OutboundEvent,
    PipelineFailurePayload,
)
from .signing import sign_payload

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared async HTTP client (connection pooling, keep-alive)
# ---------------------------------------------------------------------------

_client: httpx.AsyncClient | None = None


def _get_client(settings: EventEmitterSettings) -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=5.0,
                read=settings.n8n_emit_timeout_seconds,
                write=5.0,
                pool=5.0,
            ),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            headers={"User-Agent": "EduAIPlatform-EventEmitter/1.0"},
            follow_redirects=False,
        )
    return _client


async def close_client() -> None:
    """Gracefully close the shared HTTP client. Call from FastAPI lifespan."""
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None


# ---------------------------------------------------------------------------
# Core send function
# ---------------------------------------------------------------------------

async def send_event_to_n8n(
    event_type: EventType | str,
    payload: dict[str, Any] | Any,
    *,
    webhook_url: str,
    correlation_id: str = "",
    settings: EventEmitterSettings | None = None,
) -> EmitResult:
    """Sign and POST an event envelope to the given n8n webhook URL.

    Args:
        event_type     : One of ``EventType`` or a custom dot-separated string.
        payload        : Pydantic model or plain dict describing the event payload.
        webhook_url    : Full n8n webhook URL (e.g. http://n8n-main:5678/webhook/...).
        correlation_id : Request-scoped trace ID for log correlation.
        settings       : Override injected settings (primarily for testing).

    Returns:
        ``EmitResult`` with success flag, HTTP status, attempt count, and any error.
    """
    cfg = settings or get_event_settings()
    event = OutboundEvent.build(event_type, payload, correlation_id=correlation_id)
    body_dict = event.model_dump(mode="json")
    raw_body, headers = sign_payload(body_dict, secret=cfg.n8n_webhook_hmac_secret)

    last_error: str | None = None
    http_status: int | None = None

    for attempt in range(1, cfg.n8n_emit_max_retries + 2):
        try:
            client = _get_client(cfg)
            response = await client.post(webhook_url, content=raw_body, headers=headers)
            http_status = response.status_code

            if response.is_success:
                logger.info(
                    "n8n event emitted",
                    extra={
                        "event_id": event.event_id,
                        "event_type": event.event_type,
                        "status": http_status,
                        "attempt": attempt,
                        "correlation_id": correlation_id,
                    },
                )
                return EmitResult(
                    success=True,
                    event_id=event.event_id,
                    event_type=event.event_type,
                    http_status=http_status,
                    attempts=attempt,
                )

            # 4xx are not retried — they indicate a bad request or auth failure.
            if 400 <= http_status < 500:
                last_error = f"HTTP {http_status}: {response.text[:200]}"
                logger.error(
                    "n8n emit rejected (non-retryable)",
                    extra={
                        "event_id": event.event_id,
                        "event_type": event.event_type,
                        "status": http_status,
                        "body": response.text[:200],
                    },
                )
                break

            # 5xx — server error, eligible for retry
            last_error = f"HTTP {http_status}: {response.text[:200]}"

        except httpx.TimeoutException as exc:
            last_error = f"Timeout: {exc}"
        except httpx.RequestError as exc:
            last_error = f"Network error: {type(exc).__name__}: {exc}"
        except Exception as exc:
            last_error = f"Unexpected: {type(exc).__name__}: {exc}"
            break  # Do not retry on unexpected errors

        if attempt <= cfg.n8n_emit_max_retries:
            # Exponential back-off with ±20% jitter: 1s, 2s, 4s (capped at 8s)
            backoff = min(8.0, (2 ** (attempt - 1))) * random.uniform(0.8, 1.2)
            logger.warning(
                "n8n emit failed, retrying",
                extra={
                    "event_id": event.event_id,
                    "attempt": attempt,
                    "backoff_seconds": round(backoff, 2),
                    "error": last_error,
                },
            )
            await asyncio.sleep(backoff)

    logger.error(
        "n8n emit permanently failed",
        extra={
            "event_id": event.event_id,
            "event_type": event.event_type,
            "attempts": cfg.n8n_emit_max_retries + 1,
            "error": last_error,
            "correlation_id": correlation_id,
        },
    )
    return EmitResult(
        success=False,
        event_id=event.event_id,
        event_type=event.event_type,
        http_status=http_status,
        attempts=cfg.n8n_emit_max_retries + 1,
        error=last_error,
    )


# ---------------------------------------------------------------------------
# Typed convenience wrappers
# ---------------------------------------------------------------------------

async def emit_file_upload(
    payload: FileUploadPayload,
    *,
    correlation_id: str = "",
    settings: EventEmitterSettings | None = None,
) -> EmitResult:
    """Emit a file.upload.complete event after successful ingestion."""
    cfg = settings or get_event_settings()
    return await send_event_to_n8n(
        EventType.FILE_UPLOAD_COMPLETE,
        payload,
        webhook_url=cfg.file_upload_url,
        correlation_id=correlation_id,
        settings=cfg,
    )


async def emit_low_confidence(
    payload: LowConfidencePayload,
    *,
    correlation_id: str = "",
    settings: EventEmitterSettings | None = None,
) -> EmitResult:
    """Emit an ai.output.low_confidence event when ML output is below threshold."""
    cfg = settings or get_event_settings()
    return await send_event_to_n8n(
        EventType.AI_LOW_CONFIDENCE,
        payload,
        webhook_url=cfg.low_confidence_url,
        correlation_id=correlation_id,
        settings=cfg,
    )


async def emit_pipeline_failure(
    payload: PipelineFailurePayload,
    *,
    correlation_id: str = "",
    settings: EventEmitterSettings | None = None,
) -> EmitResult:
    """Emit an ai.pipeline.failed event for dead-letter and retry orchestration."""
    cfg = settings or get_event_settings()
    return await send_event_to_n8n(
        EventType.AI_PIPELINE_FAILED,
        payload,
        webhook_url=cfg.pipeline_failure_url,
        correlation_id=correlation_id,
        settings=cfg,
    )
