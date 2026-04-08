"""Phase 13 — Internal FastAPI router for n8n → backend callbacks.

All endpoints are protected by ``X-Internal-Secret`` header (shared secret).
No JWT auth — these routes are on the internal Docker network only; do NOT
expose them through the public nginx/Caddy gateway.

Endpoints
---------
POST /internal/idempotency/check
    Distributed-safe event deduplication via Redis SET NX EX.
    Returns ``{fresh: true}`` on first call, ``{fresh: false}`` on duplicate.
    Fail-open when Redis is unreachable (configurable via redis_strict_idempotency).

POST /internal/retry-state/increment
    Atomic Redis INCR for per-file retry counters with sliding window expiry.
    Returns ``{attempts, max_attempts, should_retry}``.

POST /internal/events/metric
    Increment a named counter in the Redis metrics hash (HINCRBY).
    Used by n8n workflows for observability.

GET /internal/admin/workflow-metrics
    Read all accumulated counters from Redis and return as JSON dashboard.

POST /internal/escalations
    Persist a low-confidence escalation record (logs + stub for Supabase insert).

POST /internal/escalations/notify
    Receive HITL notification payload (admin email/Slack would fire here).

POST /internal/escalations/resolve
    Receive admin approval/rejection decision for HITL executions.

POST /internal/dead-letter
    Persist a dead-letter record for a permanently failed pipeline run.

POST /internal/pipeline/retry
    Queue a pipeline re-execution (logs + stub for re-queuing LangGraph run).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
from datetime import datetime, timezone
from time import monotonic
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from .config import EventEmitterSettings, get_event_settings

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import redis.asyncio as aioredis
    RedisClient = aioredis.Redis
else:
    RedisClient = Any

try:
    import redis.asyncio as aioredis
    _REDIS_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as exc:
    aioredis = None  # type: ignore[assignment]
    _REDIS_IMPORT_ERROR = exc

# ---------------------------------------------------------------------------
# Module-level Redis client (lazy init, mirrors httpx pattern in emitter.py)
# ---------------------------------------------------------------------------

_redis_client: RedisClient | None = None

_METRICS_HASH = "eduai:metrics:global"
_IDEMPOTENCY_NS = "eduai:idempotency:"
_RETRY_NS = "eduai:retry:"
_local_fallback_lock = asyncio.Lock()
_local_idempotency_fallback: dict[str, float] = {}
_local_retry_fallback: dict[str, tuple[int, float]] = {}


async def _get_redis(cfg: EventEmitterSettings) -> RedisClient:
    global _redis_client
    if aioredis is None or _REDIS_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Redis dependency is not installed. Install 'redis[hiredis]>=5.0' "
            "to enable internal idempotency and metrics endpoints."
        ) from _REDIS_IMPORT_ERROR
    if _redis_client is None:
        _redis_client = aioredis.from_url(
            cfg.redis_url,
            decode_responses=True,
            max_connections=10,
            socket_connect_timeout=3,
            socket_timeout=3,
        )
    return _redis_client


async def close_redis() -> None:
    """Gracefully close the Redis client. Call from FastAPI lifespan shutdown."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None


def _purge_expired_local_entries(now: float | None = None) -> None:
    """Remove expired local fallback keys.

    The local stores are only used when Redis is unavailable. Keeping them small
    prevents dev/test fallback state from growing without bound.
    """

    current = now if now is not None else monotonic()
    expired_idempotency = [
        key for key, expires_at in _local_idempotency_fallback.items() if expires_at <= current
    ]
    for key in expired_idempotency:
        _local_idempotency_fallback.pop(key, None)

    expired_retry = [
        key for key, (_, expires_at) in _local_retry_fallback.items() if expires_at <= current
    ]
    for key in expired_retry:
        _local_retry_fallback.pop(key, None)


async def _local_idempotency_check(*, key: str, ttl_seconds: int) -> bool:
    """Fallback idempotency gate for single-process dev/test use only."""

    async with _local_fallback_lock:
        current = monotonic()
        _purge_expired_local_entries(current)
        expires_at = _local_idempotency_fallback.get(key)
        if expires_at is not None and expires_at > current:
            return False
        _local_idempotency_fallback[key] = current + ttl_seconds
        return True


async def _local_retry_increment(*, key: str, window_seconds: int) -> int:
    """Fallback retry counter for single-process dev/test use only."""

    async with _local_fallback_lock:
        current = monotonic()
        _purge_expired_local_entries(current)
        attempts, expires_at = _local_retry_fallback.get(key, (0, current + window_seconds))
        if expires_at <= current:
            attempts = 0
            expires_at = current + window_seconds
        attempts += 1
        _local_retry_fallback[key] = (attempts, expires_at)
        return attempts


async def _reset_local_fallback_state() -> None:
    """Test helper to clear local fallback state deterministically."""

    async with _local_fallback_lock:
        _local_idempotency_fallback.clear()
        _local_retry_fallback.clear()


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

async def _verify_internal(
    x_internal_secret: str = Header(..., alias="X-Internal-Secret"),
) -> None:
    cfg = get_event_settings()
    if not cfg.backend_internal_secret:
        # Dev/test: secret not configured → skip check (log warning)
        logger.warning(
            "BACKEND_INTERNAL_SECRET not set; skipping internal auth check. "
            "Set this in production."
        )
        return
    expected = cfg.backend_internal_secret.encode()
    received = x_internal_secret.encode()
    if len(expected) != len(received) or not hmac.compare_digest(expected, received):
        raise HTTPException(status_code=403, detail="Invalid internal secret")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class IdempotencyCheckIn(BaseModel):
    event_id: str = Field(..., min_length=1, max_length=256)
    ttl_seconds: int | None = Field(default=None, ge=60, le=604800)


class RetryStateIn(BaseModel):
    file_id: str = Field(..., min_length=1)
    submission_id: str = Field(default="")
    window_seconds: int = Field(default=1800, ge=60, le=86400)
    max_attempts: int = Field(default=3, ge=1, le=10)


class MetricIn(BaseModel):
    metric: str = Field(..., min_length=1, max_length=128, pattern=r"^[a-z0-9_.]+$")
    value: int = Field(default=1, ge=1)
    labels: dict[str, str] = Field(default_factory=dict)


class EscalationIn(BaseModel):
    escalation: dict[str, Any]


class EscalationNotifyIn(BaseModel):
    escalation_id: str
    resume_url: str
    severity: str
    confidence: float | None = None
    file_id: str = ""
    user_id: str = ""
    role: str = ""


class EscalationResolveIn(BaseModel):
    escalation_id: str
    decision: str  # "approved" | "rejected"
    decided_by: str = ""
    notes: str = ""
    resumed_at: str = ""


class DeadLetterIn(BaseModel):
    file_id: str = ""
    user_id: str = ""
    role: str = ""
    submission_id: str = ""
    failure_reasons: list[str] = Field(default_factory=list)
    failure_code: str = ""
    pipeline: str = ""
    graph_name: str = ""
    execution_id: str = ""
    attempts: int = 0
    max_attempts: int = 0
    correlation_id: str = ""
    event_id: str = ""


class PipelineRetryIn(BaseModel):
    file_id: str
    user_id: str
    role: str = ""
    submission_id: str = ""
    attempt: int = 1
    correlation_id: str = ""


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/internal", tags=["internal"])


# -- Deliverable 1: Distributed idempotency ----------------------------------

@router.post("/idempotency/check")
async def idempotency_check(
    body: IdempotencyCheckIn,
    _: None = Depends(_verify_internal),
) -> dict[str, Any]:
    """Atomic Redis SET NX EX idempotency gate.

    Returns ``{"fresh": true}`` when the event is seen for the first time,
    ``{"fresh": false}`` when it has already been processed.
    If Redis is unreachable and strict mode is off, a bounded in-process
    fallback gate is used for single-instance dev/test safety.
    """
    cfg = get_event_settings()
    ttl = body.ttl_seconds or cfg.redis_idempotency_ttl_seconds
    key = f"{_IDEMPOTENCY_NS}{body.event_id}"

    try:
        r = await _get_redis(cfg)
        # SET key "1" NX EX ttl — returns True if set, None if already existed
        was_set = await r.set(key, "1", nx=True, ex=ttl)
        fresh = was_set is True
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "Redis idempotency check failed",
            extra={"event_id": body.event_id, "error": err},
        )
        if cfg.redis_strict_idempotency:
            raise HTTPException(status_code=503, detail=f"Idempotency service unavailable: {err}")
        fresh = await _local_idempotency_check(key=key, ttl_seconds=ttl)

    logger.debug(
        "idempotency_check",
        extra={"event_id": body.event_id, "fresh": fresh, "ttl": ttl},
    )
    return {"fresh": fresh, "event_id": body.event_id}


# -- Deliverable 1 (cont): Distributed retry state ---------------------------

@router.post("/retry-state/increment")
async def retry_state_increment(
    body: RetryStateIn,
    _: None = Depends(_verify_internal),
) -> dict[str, Any]:
    """Atomically increment the retry counter for a (file_id, submission_id) pair.

    Uses Redis INCR with expiry set on first increment (sliding window per event).
    Returns ``{attempts, max_attempts, should_retry}`` so n8n can branch.
    If Redis is unreachable, a bounded in-process fallback counter is used for
    single-instance dev/test safety.
    """
    cfg = get_event_settings()
    sub_key = body.submission_id or "default"
    key = f"{_RETRY_NS}{body.file_id}:{sub_key}"

    try:
        r = await _get_redis(cfg)
        attempts = await r.incr(key)
        if attempts == 1:
            # Set expiry only on first increment so the window is per-event
            await r.expire(key, body.window_seconds)
        should_retry = attempts <= body.max_attempts
    except Exception as exc:
        logger.warning(
            "Redis retry-state unavailable; using local fallback",
            extra={"file_id": body.file_id, "error": str(exc)},
        )
        attempts = await _local_retry_increment(key=key, window_seconds=body.window_seconds)
        should_retry = attempts <= body.max_attempts

    return {
        "attempts": attempts,
        "max_attempts": body.max_attempts,
        "should_retry": should_retry,
        "file_id": body.file_id,
    }


# -- Deliverable 4: Metrics --------------------------------------------------

@router.post("/events/metric")
async def record_metric(
    body: MetricIn,
    _: None = Depends(_verify_internal),
) -> dict[str, Any]:
    """Increment a named counter in the Redis metrics hash (HINCRBY).

    Metric names must match ``^[a-z0-9_.]+$``.
    n8n workflows call this at key checkpoints for real-time observability.
    """
    cfg = get_event_settings()
    try:
        r = await _get_redis(cfg)
        new_value = await r.hincrby(_METRICS_HASH, body.metric, body.value)
        # Also store last-updated timestamp for dashboard freshness display
        await r.hset(_METRICS_HASH, f"_ts:{body.metric}", datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        logger.warning(
            "Redis metric record failed",
            extra={"metric": body.metric, "error": str(exc)},
        )
        new_value = -1  # Indicates the record was not persisted

    logger.info(
        "metric_recorded",
        extra={"metric": body.metric, "value": body.value, "new_total": new_value},
    )
    return {"ok": True, "metric": body.metric, "new_value": new_value}


@router.get("/admin/workflow-metrics")
async def workflow_metrics(
    _: None = Depends(_verify_internal),
) -> dict[str, Any]:
    """Aggregate all workflow metric counters from Redis.

    Returns a flat dict of metric_name → count plus ``collected_at`` timestamp.
    Counters that could not be read (Redis down) return an empty dict.
    """
    cfg = get_event_settings()
    try:
        r = await _get_redis(cfg)
        raw = await r.hgetall(_METRICS_HASH)
    except Exception as exc:
        logger.warning("Redis metrics read failed", extra={"error": str(exc)})
        raw = {}

    # Separate counters from internal _ts: keys
    counters: dict[str, int] = {}
    for field, value in raw.items():
        if not field.startswith("_ts:"):
            try:
                counters[field] = int(value)
            except (ValueError, TypeError):
                pass

    return {
        "metrics": counters,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "redis_available": bool(raw),
    }


# -- Deliverable 4 (cont): Escalation lifecycle ------------------------------

@router.post("/escalations")
async def record_escalation(
    body: EscalationIn,
    _: None = Depends(_verify_internal),
) -> dict[str, Any]:
    """Persist a low-confidence escalation record.

    In production: INSERT INTO escalations (Supabase) via service-role client.
    Currently: structured log + 200 OK so the n8n workflow is never blocked.
    """
    esc = body.escalation
    esc_id = esc.get("id", "")
    logger.warning(
        "escalation_created",
        extra={
            "escalation_id": esc_id,
            "file_id": esc.get("file_id"),
            "user_id": esc.get("user_id"),
            "severity": esc.get("severity"),
            "confidence": esc.get("confidence"),
            "correlation_id": esc.get("correlation_id"),
        },
    )
    # Increment escalation metric
    try:
        cfg = get_event_settings()
        r = await _get_redis(cfg)
        await r.hincrby(_METRICS_HASH, "escalations.created", 1)
        severity = str(esc.get("severity", "unknown"))
        await r.hincrby(_METRICS_HASH, f"escalations.severity.{severity}", 1)
    except Exception:
        pass

    return {"ok": True, "id": esc_id}


@router.post("/escalations/notify")
async def notify_escalation(
    body: EscalationNotifyIn,
    _: None = Depends(_verify_internal),
) -> dict[str, Any]:
    """Receive HITL notification request from n8n and fan out to admin channels.

    Production: send email (SES/SendGrid) or Slack message with resume_url.
    Currently: structured log only.
    """
    logger.warning(
        "hitl_notification_sent",
        extra={
            "escalation_id": body.escalation_id,
            "severity": body.severity,
            "confidence": body.confidence,
            "file_id": body.file_id,
            "user_id": body.user_id,
            "resume_url": body.resume_url[:80] if body.resume_url else "",
        },
    )
    try:
        cfg = get_event_settings()
        r = await _get_redis(cfg)
        await r.hincrby(_METRICS_HASH, "hitl.notifications_sent", 1)
    except Exception:
        pass

    return {"ok": True, "escalation_id": body.escalation_id, "notified": True}


@router.post("/escalations/resolve")
async def resolve_escalation(
    body: EscalationResolveIn,
    _: None = Depends(_verify_internal),
) -> dict[str, Any]:
    """Record an admin's HITL approval/rejection decision.

    Production: UPDATE escalations SET status=decision, resolved_at=NOW() (Supabase).
    Currently: structured log + metric counter.
    """
    decision = body.decision.lower()
    if decision not in {"approved", "rejected"}:
        raise HTTPException(status_code=422, detail="decision must be 'approved' or 'rejected'")

    logger.info(
        "hitl_decision_recorded",
        extra={
            "escalation_id": body.escalation_id,
            "decision": decision,
            "decided_by": body.decided_by,
            "notes": body.notes[:200] if body.notes else "",
        },
    )
    try:
        cfg = get_event_settings()
        r = await _get_redis(cfg)
        await r.hincrby(_METRICS_HASH, f"hitl.decisions.{decision}", 1)
    except Exception:
        pass

    return {"ok": True, "escalation_id": body.escalation_id, "decision": decision}


# -- Dead-letter handling ----------------------------------------------------

@router.post("/dead-letter")
async def dead_letter(
    body: DeadLetterIn,
    _: None = Depends(_verify_internal),
) -> dict[str, Any]:
    """Persist a dead-letter record for a permanently failed pipeline run.

    Production: INSERT INTO dead_letter_events (Supabase) + trigger PagerDuty alert.
    Currently: structured log + metric counter.
    """
    record_id = f"dlq-{body.event_id or body.correlation_id}"
    logger.error(
        "pipeline_dead_lettered",
        extra={
            "id": record_id,
            "file_id": body.file_id,
            "user_id": body.user_id,
            "failure_code": body.failure_code,
            "attempts": body.attempts,
            "max_attempts": body.max_attempts,
            "pipeline": body.pipeline,
            "correlation_id": body.correlation_id,
        },
    )
    try:
        cfg = get_event_settings()
        r = await _get_redis(cfg)
        await r.hincrby(_METRICS_HASH, "pipeline.dead_lettered", 1)
        await r.hincrby(_METRICS_HASH, f"pipeline.failures.{body.failure_code or 'unknown'}", 1)
    except Exception:
        pass

    return {"ok": True, "id": record_id, "requires_review": True}


# -- Pipeline retry dispatch -------------------------------------------------

@router.post("/pipeline/retry")
async def pipeline_retry(
    body: PipelineRetryIn,
    _: None = Depends(_verify_internal),
) -> dict[str, Any]:
    """Accept a retry request from n8n and re-queue a LangGraph pipeline run.

    Production: emit a message to a Celery/RQ/BullMQ queue or call the
    LangGraph REST entry point directly to re-run the graph for this file.
    Currently: structured log + metric counter.
    """
    logger.info(
        "pipeline_retry_accepted",
        extra={
            "file_id": body.file_id,
            "user_id": body.user_id,
            "role": body.role,
            "attempt": body.attempt,
            "correlation_id": body.correlation_id,
        },
    )
    try:
        cfg = get_event_settings()
        r = await _get_redis(cfg)
        await r.hincrby(_METRICS_HASH, "pipeline.retries_queued", 1)
        await r.hincrby(_METRICS_HASH, f"pipeline.retry_attempt.{body.attempt}", 1)
    except Exception:
        pass

    return {"ok": True, "queued": True, "file_id": body.file_id, "attempt": body.attempt}
