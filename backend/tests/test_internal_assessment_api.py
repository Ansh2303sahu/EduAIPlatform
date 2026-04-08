"""Phase 14 — Integration tests for /internal/assessment/* endpoints.

Uses the FastAPI ASGI test client (httpx + ASGITransport) exactly as the
existing test suite does, with Redis and Supabase calls mocked out.

Test scenarios:
  1. rubric-context: happy path returns expected fields
  2. rubric-context: missing user_id → 422
  3. submit-result: happy path returns assessment_id
  4. submit-result: invalid role → 422
  5. escalate: happy path returns escalation_id + severity
  6. escalate: invalid severity → 422
  7. metric: valid metric incremented, returns ok=True
  8. metric: invalid metric name pattern → 422
  9. validate-result: passes, returns warnings list
  10. validate-result: low-confidence + gate not escalating → warning surfaced
  11. audit/{file_id}: returns file_id echo
  12. all endpoints: missing X-Internal-Secret → 403 (when secret configured)
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.schemas.assessment import (
    AssessmentGateDecision,
    ClaudeReviewResult,
    GeminiExtractionResult,
    OpenAIAssessmentResult,
    ProviderUsageStats,
    RubricScore,
)

# ---------------------------------------------------------------------------
# Shared test data builders
# ---------------------------------------------------------------------------

_INTERNAL_SECRET = "test-internal-secret-32-bytes-xx"
_INTERNAL_HEADERS = {
    "X-Internal-Secret": _INTERNAL_SECRET,
    "Content-Type": "application/json",
}


def _make_openai_payload() -> dict:
    return {
        "model_id": "gpt-4o",
        "rubric_scores": [
            {
                "criterion": "Technical Accuracy",
                "band": "Merit",
                "score": 72.0,
                "justification": "Mostly correct.",
                "evidence_quotes": [],
            }
        ],
        "overall_grade": "Merit",
        "overall_score": 72.0,
        "summary": "A solid submission.",
        "strengths": [],
        "issues": [],
        "improvement_plan": [],
        "confidence": 0.82,
        "needs_human_review": False,
        "safety_flags": [],
        "usage": {
            "model": "gpt-4o",
            "prompt_tokens": 1500,
            "completion_tokens": 400,
            "total_tokens": 1900,
            "latency_ms": 2300,
            "cost_usd": 0.0057,
        },
        "raw_response_hash": "",
        "assessed_at": "2026-04-08T10:00:00+00:00",
    }


def _make_claude_payload() -> dict:
    return {
        "model_id": "claude-sonnet-4-6",
        "consistent": True,
        "reviewer_confidence": 0.91,
        "concerns": [],
        "corrections": [],
        "flagged_for_hitl": False,
        "hitl_reason": "",
        "overall_verdict": "approved",
        "usage": {
            "model": "claude-sonnet-4-6",
            "prompt_tokens": 2000,
            "completion_tokens": 300,
            "total_tokens": 2300,
            "latency_ms": 1800,
            "cost_usd": 0.0069,
        },
        "reviewed_at": "2026-04-08T10:01:00+00:00",
    }


def _make_gate_payload(pass_gate: bool = True, escalate: bool = False) -> dict:
    return {
        "pass_gate": pass_gate,
        "escalate": escalate,
        "hitl_required": escalate,
        "escalation_reasons": [],
        "final_confidence": 0.82,
        "confidence_sources": {"openai": 0.82, "claude": 0.91},
        "decided_at": "2026-04-08T10:02:00+00:00",
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture(autouse=True)
def patch_internal_secret(monkeypatch):
    """Make the internal secret match _INTERNAL_SECRET for all tests."""
    import app.events.idempotency as idempotency_mod
    import app.events.config as config_mod

    original_get = config_mod.get_event_settings

    class _FakeSettings:
        backend_internal_secret = _INTERNAL_SECRET
        redis_url = "redis://localhost:6379/1"
        redis_idempotency_ttl_seconds = 86400
        redis_strict_idempotency = False
        redis_retry_window_seconds = 1800

    def fake_get_settings():
        return _FakeSettings()

    monkeypatch.setattr(config_mod, "get_event_settings", fake_get_settings)
    # Also patch the reference in idempotency module
    monkeypatch.setattr(
        "app.events.idempotency.get_event_settings", fake_get_settings
    )
    yield


@pytest.fixture(autouse=True)
def patch_redis(monkeypatch):
    """Stub out Redis so tests don't require a running Redis instance."""

    class _FakeRedis:
        async def hincrby(self, *a, **kw):
            return 1

        async def hgetall(self, *a, **kw):
            return {}

        async def set(self, *a, **kw):
            return True

        async def incr(self, *a, **kw):
            return 1

        async def expire(self, *a, **kw):
            return True

    async def fake_get_redis(cfg):
        return _FakeRedis()

    monkeypatch.setattr("app.events.idempotency._get_redis", fake_get_redis)
    monkeypatch.setattr("app.api.internal.assessment._get_redis", fake_get_redis)
    yield


# ---------------------------------------------------------------------------
# 1 & 2: rubric-context
# ---------------------------------------------------------------------------

class TestRubricContext:
    @pytest.mark.asyncio
    async def test_happy_path(self, client):
        resp = await client.post(
            "/internal/assessment/rubric-context",
            headers=_INTERNAL_HEADERS,
            json={
                "file_id": "file-abc",
                "user_id": "user-123",
                "role": "student",
                "include_draft": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["file_id"] == "file-abc"
        assert data["role"] == "student"
        assert isinstance(data["rubric_criteria"], list)
        assert len(data["rubric_criteria"]) > 0
        assert data["schema_version"] == "14.1"

    @pytest.mark.asyncio
    async def test_professor_rubric(self, client):
        resp = await client.post(
            "/internal/assessment/rubric-context",
            headers=_INTERNAL_HEADERS,
            json={"file_id": "f1", "user_id": "u1", "role": "professor"},
        )
        assert resp.status_code == 200
        assert resp.json()["role"] == "professor"

    @pytest.mark.asyncio
    async def test_missing_user_id(self, client):
        resp = await client.post(
            "/internal/assessment/rubric-context",
            headers=_INTERNAL_HEADERS,
            json={"file_id": "f1", "role": "student"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 3 & 4: submit-result
# ---------------------------------------------------------------------------

class TestSubmitResult:
    @pytest.mark.asyncio
    async def test_happy_path(self, client):
        resp = await client.post(
            "/internal/assessment/submit-result",
            headers=_INTERNAL_HEADERS,
            json={
                "event_id": "evt-001",
                "file_id": "file-xyz",
                "user_id": "user-999",
                "role": "student",
                "openai_result": _make_openai_payload(),
                "claude_review": _make_claude_payload(),
                "gate_decision": _make_gate_payload(),
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "assessment_id" in data
        assert data["file_id"] == "file-xyz"
        assert data["gate_passed"] is True
        assert data["overall_score"] == pytest.approx(72.0)

    @pytest.mark.asyncio
    async def test_invalid_role(self, client):
        resp = await client.post(
            "/internal/assessment/submit-result",
            headers=_INTERNAL_HEADERS,
            json={
                "event_id": "evt-002",
                "file_id": "f1",
                "user_id": "u1",
                "role": "admin",  # not valid for assessment
                "openai_result": _make_openai_payload(),
                "claude_review": _make_claude_payload(),
                "gate_decision": _make_gate_payload(),
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_with_gemini(self, client):
        resp = await client.post(
            "/internal/assessment/submit-result",
            headers=_INTERNAL_HEADERS,
            json={
                "event_id": "evt-003",
                "file_id": "f2",
                "user_id": "u2",
                "role": "student",
                "openai_result": _make_openai_payload(),
                "claude_review": _make_claude_payload(),
                "gemini_extraction": {
                    "model_id": "gemini-1.5-pro",
                    "multimodal_used": True,
                    "figures": [
                        {
                            "figure_id": "fig-1",
                            "figure_type": "diagram",
                            "description": "UML diagram",
                            "quality_score": 0.9,
                        }
                    ],
                    "usage": {
                        "model": "gemini-1.5-pro",
                        "prompt_tokens": 500,
                        "completion_tokens": 200,
                        "total_tokens": 700,
                        "latency_ms": 3200,
                        "cost_usd": 0.0021,
                    },
                },
                "gate_decision": _make_gate_payload(),
            },
        )
        assert resp.status_code == 200
        assert resp.json()["assessment_id"] != ""


# ---------------------------------------------------------------------------
# 5 & 6: escalate
# ---------------------------------------------------------------------------

class TestEscalate:
    @pytest.mark.asyncio
    async def test_happy_path(self, client):
        resp = await client.post(
            "/internal/assessment/escalate",
            headers=_INTERNAL_HEADERS,
            json={
                "event_id": "evt-esc-1",
                "file_id": "file-esc",
                "user_id": "u1",
                "reasons": ["confidence < 0.40"],
                "openai_confidence": 0.32,
                "severity": "high",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "escalation_id" in data
        assert data["severity"] == "high"

    @pytest.mark.asyncio
    async def test_critical_severity(self, client):
        resp = await client.post(
            "/internal/assessment/escalate",
            headers=_INTERNAL_HEADERS,
            json={
                "event_id": "evt-esc-2",
                "file_id": "f1",
                "user_id": "u1",
                "reasons": ["safety_flags", "claude_escalate"],
                "severity": "critical",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["severity"] == "critical"

    @pytest.mark.asyncio
    async def test_invalid_severity(self, client):
        resp = await client.post(
            "/internal/assessment/escalate",
            headers=_INTERNAL_HEADERS,
            json={
                "event_id": "e1",
                "file_id": "f1",
                "user_id": "u1",
                "severity": "extreme",
            },
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 7 & 8: metric
# ---------------------------------------------------------------------------

class TestMetric:
    @pytest.mark.asyncio
    async def test_valid_metric(self, client):
        resp = await client.post(
            "/internal/assessment/metric",
            headers=_INTERNAL_HEADERS,
            json={"metric": "openai.calls", "value": 1},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["metric"] == "assessment.openai.calls"

    @pytest.mark.asyncio
    async def test_invalid_metric_pattern(self, client):
        resp = await client.post(
            "/internal/assessment/metric",
            headers=_INTERNAL_HEADERS,
            json={"metric": "openai calls", "value": 1},  # space not allowed
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_zero_value_rejected(self, client):
        resp = await client.post(
            "/internal/assessment/metric",
            headers=_INTERNAL_HEADERS,
            json={"metric": "openai.calls", "value": 0},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 9 & 10: validate-result
# ---------------------------------------------------------------------------

class TestValidateResult:
    @pytest.mark.asyncio
    async def test_passes_no_warnings(self, client):
        resp = await client.post(
            "/internal/assessment/validate-result",
            headers=_INTERNAL_HEADERS,
            json={
                "openai_result": _make_openai_payload(),
                "claude_review": _make_claude_payload(),
                "gate_decision": _make_gate_payload(pass_gate=True, escalate=False),
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True
        assert data["warnings"] == []

    @pytest.mark.asyncio
    async def test_low_confidence_not_escalating_surfaces_warning(self, client):
        openai_low = _make_openai_payload()
        openai_low["confidence"] = 0.30  # below 0.40 threshold

        resp = await client.post(
            "/internal/assessment/validate-result",
            headers=_INTERNAL_HEADERS,
            json={
                "openai_result": openai_low,
                "claude_review": _make_claude_payload(),
                "gate_decision": _make_gate_payload(pass_gate=True, escalate=False),
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True
        assert any("confidence" in w for w in data["warnings"])

    @pytest.mark.asyncio
    async def test_empty_rubric_scores_warns(self, client):
        openai_no_rubric = _make_openai_payload()
        openai_no_rubric["rubric_scores"] = []

        resp = await client.post(
            "/internal/assessment/validate-result",
            headers=_INTERNAL_HEADERS,
            json={
                "openai_result": openai_no_rubric,
                "claude_review": _make_claude_payload(),
                "gate_decision": _make_gate_payload(),
            },
        )
        assert resp.status_code == 200
        assert any("rubric_scores" in w for w in resp.json()["warnings"])


# ---------------------------------------------------------------------------
# 11: audit/{file_id}
# ---------------------------------------------------------------------------

class TestAuditEndpoint:
    @pytest.mark.asyncio
    async def test_returns_file_id(self, client):
        resp = await client.get(
            "/internal/assessment/audit/file-xyz",
            headers=_INTERNAL_HEADERS,
        )
        assert resp.status_code == 200
        assert resp.json()["file_id"] == "file-xyz"


# ---------------------------------------------------------------------------
# 12: missing X-Internal-Secret → 403
# ---------------------------------------------------------------------------

class TestAuthRequired:
    @pytest.mark.asyncio
    async def test_rubric_context_no_secret(self, client):
        resp = await client.post(
            "/internal/assessment/rubric-context",
            headers={"Content-Type": "application/json"},
            json={"file_id": "f1", "user_id": "u1", "role": "student"},
        )
        # 422 (missing header field) or 403 (wrong secret) — either means rejected
        assert resp.status_code in (403, 422)

    @pytest.mark.asyncio
    async def test_submit_result_wrong_secret(self, client):
        resp = await client.post(
            "/internal/assessment/submit-result",
            headers={
                "X-Internal-Secret": "wrong-secret",
                "Content-Type": "application/json",
            },
            json={
                "event_id": "e1",
                "file_id": "f1",
                "user_id": "u1",
                "role": "student",
                "openai_result": _make_openai_payload(),
                "claude_review": _make_claude_payload(),
                "gate_decision": _make_gate_payload(),
            },
        )
        assert resp.status_code == 403
