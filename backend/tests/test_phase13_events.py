"""Phase 13 — Event emitter + internal API test suite.

Scenarios covered (10 total)
------------------------------
 1. HMAC signing + verification round-trip (sign → verify passes)
 2. Timestamp skew: just inside tolerance passes, just outside fails
 3. emit_file_upload success path — 202 response, EmitResult.success=True
 4. emit_low_confidence retry on 503 → success on 2nd attempt
 5. emit_pipeline_failure max retries exhausted → EmitResult(success=False)
 6. 4xx response is not retried (e.g. 403)
 7. Idempotency: first call returns fresh=True; second call returns fresh=False
 8. Idempotency: Redis unavailable → fail-open → fresh=True (strict_mode=False)
 9. Retry state: increments counter up to max; should_retry flips to False
10. Metric counter: HINCRBY called, returns new_value

Run:
    pytest backend/tests/test_phase13_events.py -v
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from app.events.config import EventEmitterSettings
from app.events.schemas import (
    EmitResult,
    EventType,
    FileUploadPayload,
    LowConfidencePayload,
    PipelineFailurePayload,
)
from app.events.signing import sign_payload, verify_signature
from app.events.emitter import send_event_to_n8n, emit_file_upload, emit_low_confidence, emit_pipeline_failure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEST_SECRET = "deadbeef" * 8  # 64-hex = 32 bytes


def _make_settings(**overrides) -> EventEmitterSettings:
    """Build EventEmitterSettings with test values."""
    defaults = dict(
        n8n_webhook_hmac_secret=TEST_SECRET,
        n8n_base_url="http://n8n-test:5678",
        n8n_emit_max_retries=2,
        n8n_emit_timeout_seconds=5.0,
        backend_internal_secret="test-internal-secret",
        redis_url="redis://:testpass@localhost:6379/15",
        redis_strict_idempotency=False,
    )
    defaults.update(overrides)
    return EventEmitterSettings.model_construct(**defaults)


def _mock_response(status: int, body: dict | str = "") -> httpx.Response:
    content = json.dumps(body).encode() if isinstance(body, dict) else body.encode()
    return httpx.Response(status, content=content)


# ---------------------------------------------------------------------------
# Scenario 1 — HMAC sign → verify round-trip
# ---------------------------------------------------------------------------

class TestHMACSignVerify:
    def test_sign_then_verify_succeeds(self):
        payload = {"event_id": "evt-001", "event_type": "file.upload.complete"}
        raw_body, headers = sign_payload(payload, secret=TEST_SECRET)

        # Should not raise
        verify_signature(raw_body, headers=headers, secret=TEST_SECRET)

    def test_tampered_body_fails(self):
        payload = {"event_id": "evt-002"}
        raw_body, headers = sign_payload(payload, secret=TEST_SECRET)
        tampered = raw_body[:-1] + b"X"

        with pytest.raises(ValueError, match="Signature"):
            verify_signature(tampered, headers=headers, secret=TEST_SECRET)

    def test_wrong_secret_fails(self):
        payload = {"event_id": "evt-003"}
        raw_body, headers = sign_payload(payload, secret=TEST_SECRET)

        with pytest.raises(ValueError, match="Signature"):
            verify_signature(raw_body, headers=headers, secret="wrong-secret" * 3)


# ---------------------------------------------------------------------------
# Scenario 2 — Timestamp skew tolerance
# ---------------------------------------------------------------------------

class TestTimestampTolerance:
    def test_within_tolerance_passes(self):
        payload = {"event_id": "evt-ts-1"}
        raw_body, headers = sign_payload(payload, secret=TEST_SECRET)
        # Within tolerance — should pass
        verify_signature(raw_body, headers=headers, secret=TEST_SECRET, tolerance_seconds=300)

    def test_stale_timestamp_fails(self):
        from app.events.signing import _canonical_json

        payload = {"event_id": "evt-ts-2"}
        raw = _canonical_json(payload)
        # Craft a header with a timestamp 400 seconds in the past
        stale_ts = str(int(time.time()) - 400)
        mac = hmac.new(TEST_SECRET.encode(), raw, hashlib.sha256)
        headers = {
            "X-EduAI-Signature-256": f"sha256={mac.hexdigest()}",
            "X-EduAI-Timestamp": stale_ts,
            "X-EduAI-Idempotency-Key": "key-stale",
        }
        with pytest.raises(ValueError, match="[Tt]imestamp"):
            verify_signature(raw, headers=headers, secret=TEST_SECRET, tolerance_seconds=300)


# ---------------------------------------------------------------------------
# Scenario 3 — emit_file_upload success
# ---------------------------------------------------------------------------

class TestEmitFileUpload:
    @pytest.mark.asyncio
    async def test_success_returns_emit_result(self):
        settings = _make_settings()
        payload = FileUploadPayload(
            file_id="file-abc",
            user_id="user-1",
            role="student",
            file_name="essay.pdf",
            file_type="application/pdf",
            file_size_bytes=102400,
        )

        mock_resp = _mock_response(202, {"status": "accepted"})

        with patch("app.events.emitter._get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_get_client.return_value = mock_client

            result = await emit_file_upload(payload, settings=settings)

        assert result.success is True
        assert result.http_status == 202
        assert result.attempts == 1
        assert result.error is None


# ---------------------------------------------------------------------------
# Scenario 4 — emit_low_confidence retries on 503 → success on 2nd attempt
# ---------------------------------------------------------------------------

class TestRetryBehavior:
    @pytest.mark.asyncio
    async def test_retry_on_503_then_success(self):
        settings = _make_settings(n8n_emit_max_retries=2)
        payload = LowConfidencePayload(
            file_id="file-low",
            user_id="user-2",
            role="professor",
            confidence_score=0.21,
            confidence_threshold=0.45,
            model_id="model-v1",
            safe_mode=True,
        )

        resp_fail = _mock_response(503, {"error": "unavailable"})
        resp_ok   = _mock_response(202, {"status": "accepted"})

        call_count = 0

        async def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return resp_fail if call_count == 1 else resp_ok

        with patch("app.events.emitter._get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=_side_effect)
            mock_get_client.return_value = mock_client
            with patch("app.events.emitter.asyncio.sleep", new_callable=AsyncMock):
                result = await emit_low_confidence(payload, settings=settings)

        assert result.success is True
        assert result.attempts == 2
        assert call_count == 2


# ---------------------------------------------------------------------------
# Scenario 5 — emit_pipeline_failure exhausts retries → EmitResult(success=False)
# ---------------------------------------------------------------------------

class TestMaxRetriesExhausted:
    @pytest.mark.asyncio
    async def test_all_retries_fail(self):
        settings = _make_settings(n8n_emit_max_retries=2)
        payload = PipelineFailurePayload(
            file_id="file-fail",
            user_id="user-3",
            role="student",
            failure_reasons=["timeout"],
            failure_code="phase12.ingestion.error",
            pipeline="phase12_langgraph",
        )

        resp_fail = _mock_response(503, {"error": "service down"})

        with patch("app.events.emitter._get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=resp_fail)
            mock_get_client.return_value = mock_client
            with patch("app.events.emitter.asyncio.sleep", new_callable=AsyncMock):
                result = await emit_pipeline_failure(payload, settings=settings)

        assert result.success is False
        assert result.attempts == 3          # 1 initial + 2 retries
        assert result.error is not None


# ---------------------------------------------------------------------------
# Scenario 6 — 4xx response is NOT retried
# ---------------------------------------------------------------------------

class TestNoRetryOn4xx:
    @pytest.mark.asyncio
    async def test_403_not_retried(self):
        settings = _make_settings(n8n_emit_max_retries=3)
        payload = FileUploadPayload(
            file_id="file-403",
            user_id="user-4",
            role="student",
            file_name="test.pdf",
            file_type="application/pdf",
            file_size_bytes=1024,
        )

        resp_forbidden = _mock_response(403, {"error": "forbidden"})

        call_count = 0

        async def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return resp_forbidden

        with patch("app.events.emitter._get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=_side_effect)
            mock_get_client.return_value = mock_client

            result = await emit_file_upload(payload, settings=settings)

        # Must stop after first 4xx — no retry
        assert call_count == 1
        assert result.success is False
        assert result.http_status == 403


# ---------------------------------------------------------------------------
# Scenario 7 — Idempotency: first call fresh=True, second call fresh=False
# ---------------------------------------------------------------------------

class TestIdempotencyCheck:
    @pytest.mark.asyncio
    async def test_first_call_fresh_second_call_duplicate(self):
        from app.events.idempotency import idempotency_check, IdempotencyCheckIn

        mock_redis = AsyncMock()
        # First SET NX → returns True (key was new)
        # Second SET NX → returns None (key already exists)
        mock_redis.set = AsyncMock(side_effect=[True, None])

        settings = _make_settings()

        with patch("app.events.idempotency._get_redis", new_callable=AsyncMock, return_value=mock_redis):
            body = IdempotencyCheckIn(event_id="evt-idem-1")

            result1 = await idempotency_check(body, _=None)
            result2 = await idempotency_check(body, _=None)

        assert result1["fresh"] is True
        assert result2["fresh"] is False

    @pytest.mark.asyncio
    async def test_idempotency_key_uses_correct_namespace(self):
        from app.events.idempotency import idempotency_check, IdempotencyCheckIn

        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(return_value=True)
        settings = _make_settings()

        with patch("app.events.idempotency._get_redis", new_callable=AsyncMock, return_value=mock_redis):
            await idempotency_check(IdempotencyCheckIn(event_id="evt-ns-test"), _=None)

        call_args = mock_redis.set.call_args
        key_used = call_args[0][0]
        assert key_used.startswith("eduai:idempotency:")
        assert "evt-ns-test" in key_used


# ---------------------------------------------------------------------------
# Scenario 8 — Idempotency: Redis unavailable → fail-open → fresh=True
# ---------------------------------------------------------------------------

class TestIdempotencyFailOpen:
    @pytest.mark.asyncio
    async def test_redis_unavailable_fail_open(self):
        from app.events.idempotency import _reset_local_fallback_state, idempotency_check, IdempotencyCheckIn

        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(side_effect=ConnectionError("Redis unreachable"))
        settings = _make_settings(redis_strict_idempotency=False)

        await _reset_local_fallback_state()
        with patch("app.events.idempotency._get_redis", new_callable=AsyncMock, return_value=mock_redis):
            with patch("app.events.idempotency.get_event_settings", return_value=settings):
                result = await idempotency_check(
                    IdempotencyCheckIn(event_id="evt-fail-open"), _=None
                )

        # First fallback attempt should still be fresh.
        assert result["fresh"] is True

    @pytest.mark.asyncio
    async def test_redis_unavailable_uses_local_duplicate_detection(self):
        from app.events.idempotency import _reset_local_fallback_state, idempotency_check, IdempotencyCheckIn

        settings = _make_settings(redis_strict_idempotency=False)

        await _reset_local_fallback_state()
        with patch(
            "app.events.idempotency._get_redis",
            new_callable=AsyncMock,
            side_effect=ConnectionError("Redis unreachable"),
        ):
            with patch("app.events.idempotency.get_event_settings", return_value=settings):
                body = IdempotencyCheckIn(event_id="evt-fallback-duplicate", ttl_seconds=300)
                result1 = await idempotency_check(body, _=None)
                result2 = await idempotency_check(body, _=None)

        assert result1["fresh"] is True
        assert result2["fresh"] is False

    @pytest.mark.asyncio
    async def test_missing_redis_dependency_fail_open(self):
        from app.events import idempotency as idempotency_module
        from app.events.idempotency import _reset_local_fallback_state, IdempotencyCheckIn, idempotency_check

        settings = _make_settings(redis_strict_idempotency=False)

        await _reset_local_fallback_state()
        with patch.object(idempotency_module, "aioredis", None):
            with patch.object(
                idempotency_module,
                "_REDIS_IMPORT_ERROR",
                ModuleNotFoundError("No module named 'redis'"),
            ):
                with patch("app.events.idempotency.get_event_settings", return_value=settings):
                    result = await idempotency_check(
                        IdempotencyCheckIn(event_id="evt-missing-redis"),
                        _=None,
                    )

        assert result["fresh"] is True

    @pytest.mark.asyncio
    async def test_redis_unavailable_strict_mode_raises_503(self):
        from app.events.idempotency import idempotency_check, IdempotencyCheckIn
        from fastapi import HTTPException

        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(side_effect=ConnectionError("Redis down"))
        settings = _make_settings(redis_strict_idempotency=True)

        with patch("app.events.idempotency._get_redis", new_callable=AsyncMock, return_value=mock_redis):
            with patch("app.events.idempotency.get_event_settings", return_value=settings):
                with pytest.raises(HTTPException) as exc_info:
                    await idempotency_check(
                        IdempotencyCheckIn(event_id="evt-strict"), _=None
                    )

        assert exc_info.value.status_code == 503


# ---------------------------------------------------------------------------
# Scenario 9 — Retry state: counter increments, should_retry flips at max
# ---------------------------------------------------------------------------

class TestRetryState:
    @pytest.mark.asyncio
    async def test_increments_up_to_max_then_blocks(self):
        from app.events.idempotency import retry_state_increment, RetryStateIn
        from unittest.mock import call as mock_call

        # Simulate Redis INCR returning 1, 2, 3, 4 on successive calls
        incr_values = [1, 2, 3, 4]
        incr_call_index = 0

        async def _incr(key):
            nonlocal incr_call_index
            val = incr_values[incr_call_index]
            incr_call_index += 1
            return val

        mock_redis = AsyncMock()
        mock_redis.incr = AsyncMock(side_effect=_incr)
        mock_redis.expire = AsyncMock(return_value=True)

        settings = _make_settings()
        body = RetryStateIn(file_id="file-retry", submission_id="sub-1", max_attempts=3)

        results = []
        with patch("app.events.idempotency._get_redis", new_callable=AsyncMock, return_value=mock_redis):
            with patch("app.events.idempotency.get_event_settings", return_value=settings):
                for _ in range(4):
                    r = await retry_state_increment(body, _=None)
                    results.append(r)

        assert results[0]["should_retry"] is True   # attempt 1/3
        assert results[1]["should_retry"] is True   # attempt 2/3
        assert results[2]["should_retry"] is True   # attempt 3/3
        assert results[3]["should_retry"] is False  # attempt 4 > max

    @pytest.mark.asyncio
    async def test_redis_unavailable_uses_local_retry_counter(self):
        from app.events.idempotency import _reset_local_fallback_state, retry_state_increment, RetryStateIn

        settings = _make_settings()
        body = RetryStateIn(file_id="file-local-retry", submission_id="sub-local", max_attempts=2)

        await _reset_local_fallback_state()
        results = []
        with patch(
            "app.events.idempotency._get_redis",
            new_callable=AsyncMock,
            side_effect=ConnectionError("Redis down"),
        ):
            with patch("app.events.idempotency.get_event_settings", return_value=settings):
                for _ in range(3):
                    results.append(await retry_state_increment(body, _=None))

        assert results[0]["attempts"] == 1
        assert results[0]["should_retry"] is True
        assert results[1]["attempts"] == 2
        assert results[1]["should_retry"] is True
        assert results[2]["attempts"] == 3
        assert results[2]["should_retry"] is False


# ---------------------------------------------------------------------------
# Scenario 10 — Metric counter: HINCRBY called, returns new_value
# ---------------------------------------------------------------------------

class TestMetricCounter:
    @pytest.mark.asyncio
    async def test_hincrby_called_returns_new_value(self):
        from app.events.idempotency import record_metric, MetricIn

        mock_redis = AsyncMock()
        mock_redis.hincrby = AsyncMock(return_value=42)
        mock_redis.hset    = AsyncMock(return_value=1)

        settings = _make_settings()

        with patch("app.events.idempotency._get_redis", new_callable=AsyncMock, return_value=mock_redis):
            with patch("app.events.idempotency.get_event_settings", return_value=settings):
                result = await record_metric(MetricIn(metric="file_upload.processed"), _=None)

        assert result["ok"] is True
        assert result["new_value"] == 42
        assert result["metric"] == "file_upload.processed"

        # Verify HINCRBY was called with correct hash key + field
        call_args = mock_redis.hincrby.call_args
        assert call_args[0][1] == "file_upload.processed"

    @pytest.mark.asyncio
    async def test_metric_redis_failure_returns_negative_one(self):
        from app.events.idempotency import record_metric, MetricIn

        mock_redis = AsyncMock()
        mock_redis.hincrby = AsyncMock(side_effect=ConnectionError("Redis down"))

        settings = _make_settings()

        with patch("app.events.idempotency._get_redis", new_callable=AsyncMock, return_value=mock_redis):
            with patch("app.events.idempotency.get_event_settings", return_value=settings):
                result = await record_metric(MetricIn(metric="file_upload.processed"), _=None)

        # Soft failure: ok=True but new_value=-1 signals the counter was not stored
        assert result["ok"] is True
        assert result["new_value"] == -1
