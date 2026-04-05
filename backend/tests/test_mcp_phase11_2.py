"""
Phase 11.2 MCP tests.

Covers:
- LLM success path (mocked llm-service call)
- Deterministic fallback when LLM is disabled
- Invalid LLM JSON → fallback
- Empty LLM response → fallback
- Ownership denial when student provides file_id belonging to another user
- Cache hit / idempotency behaviour
- ExplainabilityMeta presence and content
- Rubric evaluator grounding shape (evidence_points)
- Consistency checker contradiction detection
- Metrics recording
- Metrics endpoint (admin only)
- Cache invalidation endpoint (admin only)

All unit tests disable audit, ownership, and metrics by default to avoid
external I/O.  LLM calls are mocked at the ``app.mcp.llm_client`` level.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.mcp.schemas import MCPFailureEnvelope, MCPSuccessEnvelope

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import app.mcp  # noqa: F401 — triggers tool registration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(role: str = "student", user_id: str = "u1", file_id: str | None = None):
    from app.mcp.schemas import ToolExecutionContext
    return ToolExecutionContext(
        user_id=user_id,
        role=role,
        correlation_id="test-corr",
        file_id=file_id,
    )


def _disable_externals(mcp_settings):
    """Disable audit, ownership, and metrics for unit tests."""
    mcp_settings.audit_enabled = False
    mcp_settings.ownership_check_enabled = False
    mcp_settings.metrics_enabled = False
    mcp_settings.cache_ttl_seconds = 0  # disable caching by default
    return mcp_settings


def _mock_llm_json(json_str: str):
    """Return a mock call_llm that produces a specific JSON string."""
    from app.mcp.llm_client import LLMCallResult
    return AsyncMock(return_value=LLMCallResult(ok=True, text=json_str, model_used="mock-model"))


def _mock_llm_fail(reason: str = "service unavailable"):
    """Return a mock call_llm that fails."""
    from app.mcp.llm_client import LLMCallResult
    return AsyncMock(return_value=LLMCallResult(ok=False, error_reason=reason))


def _assert_success(
    envelope: MCPSuccessEnvelope | MCPFailureEnvelope,
) -> MCPSuccessEnvelope:
    assert envelope.ok is True
    return cast(MCPSuccessEnvelope, envelope)


def _assert_failure(
    envelope: MCPSuccessEnvelope | MCPFailureEnvelope,
) -> MCPFailureEnvelope:
    assert envelope.ok is False
    return cast(MCPFailureEnvelope, envelope)


# ===========================================================================
# Summariser — LLM path
# ===========================================================================

class TestSummariserLLM:

    @pytest.mark.asyncio
    async def test_llm_success_uses_llm_summary(self, monkeypatch):
        from app.mcp import config as mcp_config
        _disable_externals(mcp_config.mcp_settings)
        mcp_config.mcp_settings.llm_enabled = True

        llm_json = '{"summary": "This is a test summary.", "key_points": ["Testing", "Summary"]}'
        with patch("app.mcp.tools.student_summariser.call_llm", _mock_llm_json(llm_json)):
            from app.mcp.executor import execute_tool
            from app.mcp.schemas import MCPExecuteRequest
            r = await execute_tool(MCPExecuteRequest(
                tool_name="student.summariser.v1",
                payload={"text": "First sentence. Second sentence."},
                context=_make_ctx(),
            ))

        r = _assert_success(r)
        assert r.result["summary"] == "This is a test summary."
        assert "Testing" in r.result["key_points"]
        assert r.meta.llm_used is True
        assert r.meta.deterministic_fallback is False
        assert r.meta.model_used == "mock-model"

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_extractive(self, monkeypatch):
        from app.mcp import config as mcp_config
        _disable_externals(mcp_config.mcp_settings)
        mcp_config.mcp_settings.llm_enabled = True

        with patch("app.mcp.tools.student_summariser.call_llm", _mock_llm_fail()):
            from app.mcp.executor import execute_tool
            from app.mcp.schemas import MCPExecuteRequest
            r = await execute_tool(MCPExecuteRequest(
                tool_name="student.summariser.v1",
                payload={"text": "Alpha. Beta. Gamma."},
                context=_make_ctx(),
            ))

        r = _assert_success(r)
        assert r.meta.deterministic_fallback is True
        assert r.meta.llm_used is False
        # fallback warning must be present
        assert any("LLM unavailable" in w for w in r.result["warnings"])

    @pytest.mark.asyncio
    async def test_invalid_llm_json_falls_back(self, monkeypatch):
        from app.mcp import config as mcp_config
        _disable_externals(mcp_config.mcp_settings)
        mcp_config.mcp_settings.llm_enabled = True

        # LLM returns garbage (non-JSON)
        from app.mcp.llm_client import LLMCallResult
        bad_result = AsyncMock(return_value=LLMCallResult(ok=True, text="NOT JSON AT ALL", model_used="m"))
        with patch("app.mcp.tools.student_summariser.call_llm", bad_result):
            from app.mcp.executor import execute_tool
            from app.mcp.schemas import MCPExecuteRequest
            r = await execute_tool(MCPExecuteRequest(
                tool_name="student.summariser.v1",
                payload={"text": "Alpha. Beta.", "max_sentences": 1},
                context=_make_ctx(),
            ))

        r = _assert_success(r)
        assert r.meta.deterministic_fallback is True

    @pytest.mark.asyncio
    async def test_short_text_warning(self, monkeypatch):
        from app.mcp import config as mcp_config
        _disable_externals(mcp_config.mcp_settings)
        mcp_config.mcp_settings.llm_enabled = False

        from app.mcp.executor import execute_tool
        from app.mcp.schemas import MCPExecuteRequest
        r = await execute_tool(MCPExecuteRequest(
            tool_name="student.summariser.v1",
            payload={"text": "Short."},
            context=_make_ctx(),
        ))
        r = _assert_success(r)
        assert any("short" in w.lower() for w in r.result["warnings"])


# ===========================================================================
# Structure improver — LLM path
# ===========================================================================

class TestStructureImproverLLM:

    @pytest.mark.asyncio
    async def test_llm_success_returns_llm_sections(self, monkeypatch):
        from app.mcp import config as mcp_config
        _disable_externals(mcp_config.mcp_settings)
        mcp_config.mcp_settings.llm_enabled = True

        llm_json = (
            '{"detected_sections": ["Introduction", "Conclusion"], '
            '"missing_sections": ["References"], '
            '"suggested_order": ["Introduction", "Conclusion", "References"], '
            '"improvement_notes": ["Add a References section."]}'
        )
        with patch("app.mcp.tools.student_structure_improver.call_llm", _mock_llm_json(llm_json)):
            from app.mcp.executor import execute_tool
            from app.mcp.schemas import MCPExecuteRequest
            r = await execute_tool(MCPExecuteRequest(
                tool_name="student.structure_improver.v1",
                payload={"text": "Introduction: hi. Conclusion: bye.", "submission_type": "essay"},
                context=_make_ctx(),
            ))

        r = _assert_success(r)
        assert "Introduction" in r.result["detected_sections"]
        assert "References" in r.result["missing_sections"]
        assert r.meta.llm_used is True

    @pytest.mark.asyncio
    async def test_deterministic_fallback_detects_headings(self, monkeypatch):
        from app.mcp import config as mcp_config
        _disable_externals(mcp_config.mcp_settings)
        mcp_config.mcp_settings.llm_enabled = False

        from app.mcp.executor import execute_tool
        from app.mcp.schemas import MCPExecuteRequest
        r = await execute_tool(MCPExecuteRequest(
            tool_name="student.structure_improver.v1",
            payload={"text": "Introduction: test.\nMethodology: approach.\nConclusion: done."},
            context=_make_ctx(),
        ))
        r = _assert_success(r)
        assert "Introduction" in r.result["detected_sections"]
        assert r.meta.deterministic_fallback is True


# ===========================================================================
# Rubric evaluator — grounding shape
# ===========================================================================

class TestRubricEvaluatorGrounding:

    @pytest.mark.asyncio
    async def test_llm_success_populates_evidence_points(self, monkeypatch):
        from app.mcp import config as mcp_config
        _disable_externals(mcp_config.mcp_settings)
        mcp_config.mcp_settings.llm_enabled = True

        llm_json = (
            '{"evaluations": [{"criterion": "Clarity", "provisional_band": "First (70%+)", '
            '"justification": "Clear writing throughout.", '
            '"evidence_points": ["The introduction states..."]}, '
            '{"criterion": "Depth", "provisional_band": "Upper Second (60-69%)", '
            '"justification": "Good depth.", "evidence_points": []}], '
            '"overall_assessment": "Strong submission.", "warnings": []}'
        )
        with patch("app.mcp.tools.professor_rubric_evaluator.call_llm", _mock_llm_json(llm_json)):
            from app.mcp.executor import execute_tool
            from app.mcp.schemas import MCPExecuteRequest
            r = await execute_tool(MCPExecuteRequest(
                tool_name="professor.rubric_evaluator.v1",
                payload={
                    "submission_text": "A well-written introduction states the objective clearly.",
                    "rubric_criteria": ["Clarity", "Depth"],
                },
                context=_make_ctx(role="professor"),
            ))

        r = _assert_success(r)
        evals = r.result["evaluations"]
        assert len(evals) == 2
        clarity = next(e for e in evals if e["criterion"] == "Clarity")
        assert len(clarity["evidence_points"]) == 1
        assert r.meta.llm_used is True

    @pytest.mark.asyncio
    async def test_deterministic_fallback_heuristic_band(self, monkeypatch):
        from app.mcp import config as mcp_config
        _disable_externals(mcp_config.mcp_settings)
        mcp_config.mcp_settings.llm_enabled = False

        from app.mcp.executor import execute_tool
        from app.mcp.schemas import MCPExecuteRequest
        long_text = " ".join(["word"] * 1_200)
        r = await execute_tool(MCPExecuteRequest(
            tool_name="professor.rubric_evaluator.v1",
            payload={"submission_text": long_text, "rubric_criteria": ["Structure"]},
            context=_make_ctx(role="professor"),
        ))
        r = _assert_success(r)
        assert r.meta.deterministic_fallback is True
        assert r.result["evaluations"][0]["criterion"] == "Structure"
        # 1200 words >= standard threshold (1000) → Upper Second or above
        band = r.result["evaluations"][0]["provisional_band"]
        assert band in {"First (70%+)", "Upper Second (60-69%)"}


# ===========================================================================
# Consistency checker — contradiction detection
# ===========================================================================

class TestConsistencyCheckerDetection:

    @pytest.mark.asyncio
    async def test_positive_feedback_low_score_detected(self, monkeypatch):
        from app.mcp import config as mcp_config
        _disable_externals(mcp_config.mcp_settings)
        mcp_config.mcp_settings.llm_enabled = False

        from app.mcp.executor import execute_tool
        from app.mcp.schemas import MCPExecuteRequest
        r = await execute_tool(MCPExecuteRequest(
            tool_name="professor.consistency_checker.v1",
            payload={
                "feedback_items": ["Excellent work throughout!"],
                "scores": [25.0],
            },
            context=_make_ctx(role="professor"),
        ))
        r = _assert_success(r)
        assert len(r.result["contradictions"]) >= 1
        assert r.result["consistency_score"] < 1.0

    @pytest.mark.asyncio
    async def test_no_contradictions_when_consistent(self, monkeypatch):
        from app.mcp import config as mcp_config
        _disable_externals(mcp_config.mcp_settings)
        mcp_config.mcp_settings.llm_enabled = False

        from app.mcp.executor import execute_tool
        from app.mcp.schemas import MCPExecuteRequest
        r = await execute_tool(MCPExecuteRequest(
            tool_name="professor.consistency_checker.v1",
            payload={
                "feedback_items": ["Good work here.", "Reasonable effort."],
                "scores": [75.0, 70.0],
            },
            context=_make_ctx(role="professor"),
        ))
        r = _assert_success(r)
        assert r.result["consistency_score"] == 1.0

    @pytest.mark.asyncio
    async def test_count_mismatch_produces_warning(self, monkeypatch):
        from app.mcp import config as mcp_config
        _disable_externals(mcp_config.mcp_settings)
        mcp_config.mcp_settings.llm_enabled = False

        from app.mcp.executor import execute_tool
        from app.mcp.schemas import MCPExecuteRequest
        r = await execute_tool(MCPExecuteRequest(
            tool_name="professor.consistency_checker.v1",
            payload={
                "feedback_items": ["Good.", "Average."],
                "scores": [75.0],
            },
            context=_make_ctx(role="professor"),
        ))
        r = _assert_success(r)
        assert any("mismatch" in w.lower() for w in r.result["warnings"])

    @pytest.mark.asyncio
    async def test_final_summary_mismatch_detected(self, monkeypatch):
        from app.mcp import config as mcp_config
        _disable_externals(mcp_config.mcp_settings)
        mcp_config.mcp_settings.llm_enabled = False

        from app.mcp.executor import execute_tool
        from app.mcp.schemas import MCPExecuteRequest
        r = await execute_tool(MCPExecuteRequest(
            tool_name="professor.consistency_checker.v1",
            payload={
                "feedback_items": ["Reasonable."],
                "scores": [85.0],
                "final_summary": "This is a very poor and weak submission.",
            },
            context=_make_ctx(role="professor"),
        ))
        r = _assert_success(r)
        # Final summary says negative (poor/weak) but scores are high → contradiction
        assert len(r.result["contradictions"]) >= 1

    @pytest.mark.asyncio
    async def test_llm_merges_with_deterministic(self, monkeypatch):
        from app.mcp import config as mcp_config
        _disable_externals(mcp_config.mcp_settings)
        mcp_config.mcp_settings.llm_enabled = True

        llm_json = (
            '{"contradictions": ["Tone mismatch in item 2."], '
            '"suggested_fixes": ["Revise item 2 tone."], '
            '"warnings": ["Score variance high."]}'
        )
        with patch("app.mcp.tools.professor_consistency_checker.call_llm", _mock_llm_json(llm_json)):
            from app.mcp.executor import execute_tool
            from app.mcp.schemas import MCPExecuteRequest
            r = await execute_tool(MCPExecuteRequest(
                tool_name="professor.consistency_checker.v1",
                payload={
                    "feedback_items": ["Excellent!", "Poor quality."],
                    "scores": [85.0, 20.0],
                },
                context=_make_ctx(role="professor"),
            ))

        r = _assert_success(r)
        assert r.meta.llm_used is True
        # Both deterministic + LLM contradictions should be present
        assert len(r.result["contradictions"]) >= 1


# ===========================================================================
# Ownership enforcement
# ===========================================================================

class TestOwnershipEnforcement:

    @pytest.mark.asyncio
    async def test_student_denied_when_file_owned_by_other(self, monkeypatch):
        from app.mcp import config as mcp_config
        _disable_externals(mcp_config.mcp_settings)
        mcp_config.mcp_settings.llm_enabled = False
        mcp_config.mcp_settings.ownership_check_enabled = True
        mcp_config.mcp_settings.audit_enabled = False

        # Mock ownership check: file belongs to a different user.
        from app.mcp.ownership import OwnershipResult
        mock_check = AsyncMock(return_value=OwnershipResult(
            allowed=False,
            denial_reason="Student u1 does not own file_id='file-xyz'",
        ))

        with patch("app.mcp.policies.check_resource_ownership", mock_check):
            from app.mcp.executor import execute_tool
            from app.mcp.schemas import MCPExecuteRequest
            r = await execute_tool(MCPExecuteRequest(
                tool_name="student.summariser.v1",
                payload={"text": "Hello."},
                context=_make_ctx(role="student", file_id="file-xyz"),
            ))

        r = _assert_failure(r)
        assert r.error_code == "mcp.policy_denied"
        assert "does not own" in r.message.lower()

    @pytest.mark.asyncio
    async def test_professor_passes_ownership_check(self, monkeypatch):
        """Professors bypass ownership — check is never called for professor role."""
        from app.mcp import config as mcp_config
        _disable_externals(mcp_config.mcp_settings)
        mcp_config.mcp_settings.llm_enabled = False
        mcp_config.mcp_settings.ownership_check_enabled = True

        mock_check = AsyncMock(return_value=None)  # Should not be called.

        with patch("app.mcp.policies.check_resource_ownership", mock_check):
            from app.mcp.executor import execute_tool
            from app.mcp.schemas import MCPExecuteRequest
            r = await execute_tool(MCPExecuteRequest(
                tool_name="professor.rubric_evaluator.v1",
                payload={
                    "submission_text": "Test submission.",
                    "rubric_criteria": ["Clarity"],
                },
                context=_make_ctx(role="professor", file_id="file-abc"),
            ))

        r = _assert_success(r)
        # Professor skips DB ownership call.
        mock_check.assert_not_called()


# ===========================================================================
# Caching / idempotency
# ===========================================================================

class TestCaching:

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_result(self, monkeypatch):
        from app.mcp import config as mcp_config
        _disable_externals(mcp_config.mcp_settings)
        mcp_config.mcp_settings.llm_enabled = False
        mcp_config.mcp_settings.cache_ttl_seconds = 300.0  # enable cache

        from app.mcp.executor import execute_tool
        from app.mcp.schemas import MCPExecuteRequest
        from app.mcp import cache

        payload = {"text": "Cache test sentence.", "max_sentences": 1}
        ctx = _make_ctx()

        # First call — populates cache.
        r1 = await execute_tool(MCPExecuteRequest(
            tool_name="student.summariser.v1",
            payload=payload,
            context=ctx,
        ))
        r1 = _assert_success(r1)
        assert r1.meta.cache_hit is False

        # Second call — should hit cache.
        r2 = await execute_tool(MCPExecuteRequest(
            tool_name="student.summariser.v1",
            payload=payload,
            context=ctx,
        ))
        r2 = _assert_success(r2)
        assert r2.meta.cache_hit is True
        assert r2.result == r1.result

        # Cleanup
        cache.invalidate("student.summariser.v1")

    @pytest.mark.asyncio
    async def test_different_payloads_produce_different_cache_keys(self, monkeypatch):
        from app.mcp import config as mcp_config
        _disable_externals(mcp_config.mcp_settings)
        mcp_config.mcp_settings.llm_enabled = False
        mcp_config.mcp_settings.cache_ttl_seconds = 300.0

        from app.mcp.executor import execute_tool
        from app.mcp.schemas import MCPExecuteRequest
        from app.mcp import cache

        ctx = _make_ctx()
        r1 = await execute_tool(MCPExecuteRequest(
            tool_name="student.summariser.v1",
            payload={"text": "Text A.", "max_sentences": 1},
            context=ctx,
        ))
        r2 = await execute_tool(MCPExecuteRequest(
            tool_name="student.summariser.v1",
            payload={"text": "Text B.", "max_sentences": 1},
            context=ctx,
        ))

        # Both miss cache — different payloads.
        r1 = _assert_success(r1)
        r2 = _assert_success(r2)
        assert r1.meta.cache_hit is False
        assert r2.meta.cache_hit is False
        assert r1.result["summary"] != r2.result["summary"]

        cache.invalidate("student.summariser.v1")


# ===========================================================================
# ExplainabilityMeta presence
# ===========================================================================

class TestExplainabilityMeta:

    @pytest.mark.asyncio
    async def test_meta_always_present_on_success(self, monkeypatch):
        from app.mcp import config as mcp_config
        _disable_externals(mcp_config.mcp_settings)
        mcp_config.mcp_settings.llm_enabled = False

        from app.mcp.executor import execute_tool
        from app.mcp.schemas import MCPExecuteRequest

        r = await execute_tool(MCPExecuteRequest(
            tool_name="student.summariser.v1",
            payload={"text": "A test sentence."},
            context=_make_ctx(),
        ))
        r = _assert_success(r)
        assert r.meta is not None
        assert isinstance(r.meta.llm_used, bool)
        assert isinstance(r.meta.deterministic_fallback, bool)
        assert isinstance(r.meta.confidence_note, str)

    @pytest.mark.asyncio
    async def test_meta_deterministic_flag_true_when_llm_disabled(self, monkeypatch):
        from app.mcp import config as mcp_config
        _disable_externals(mcp_config.mcp_settings)
        mcp_config.mcp_settings.llm_enabled = False

        from app.mcp.executor import execute_tool
        from app.mcp.schemas import MCPExecuteRequest

        r = await execute_tool(MCPExecuteRequest(
            tool_name="student.structure_improver.v1",
            payload={"text": "Introduction here."},
            context=_make_ctx(),
        ))
        r = _assert_success(r)
        assert r.meta.deterministic_fallback is True
        assert r.meta.llm_used is False


# ===========================================================================
# Metrics
# ===========================================================================

class TestMetrics:

    def test_metrics_snapshot_structure(self):
        from app.mcp.metrics import snapshot
        s = snapshot()
        assert "tools" in s
        assert "totals" in s
        assert "success" in s["totals"]
        assert "failure" in s["totals"]

    def test_record_success_increments_counter(self):
        from app.mcp.metrics import record_success, snapshot
        before = snapshot()["totals"]["success"]
        record_success("test.tool.v1")
        after = snapshot()["totals"]["success"]
        assert after == before + 1

    def test_record_failure_increments_counter(self):
        from app.mcp.metrics import record_failure, snapshot
        before = snapshot()["totals"]["failure"]
        record_failure("test.tool.v1", error_code="mcp.timeout", timeout=True)
        after = snapshot()["totals"]["failure"]
        assert after == before + 1
        assert after >= snapshot()["totals"]["timeout"]


# ===========================================================================
# HTTP integration — Phase 11.2 specific
# ===========================================================================

@pytest.mark.asyncio
async def test_http_success_includes_meta(async_client):
    async with async_client as ac:
        resp = await ac.post(
            "/api/mcp/execute",
            json={
                "tool_name": "student.summariser.v1",
                "payload": {"text": "Hello world. Second sentence."},
            },
            headers={"Authorization": "Bearer token-student-a"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert "meta" in body
    meta = body["meta"]
    assert "llm_used" in meta
    assert "deterministic_fallback" in meta
    assert "confidence_note" in meta
    assert "cache_hit" in meta


@pytest.mark.asyncio
async def test_http_metrics_endpoint_admin_only(async_client):
    # Admin can access.
    async with async_client as ac:
        resp = await ac.get(
            "/api/mcp/metrics",
            headers={"Authorization": "Bearer token-admin"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert "metrics" in body


@pytest.mark.asyncio
async def test_http_metrics_endpoint_student_forbidden(async_client):
    async with async_client as ac:
        resp = await ac.get(
            "/api/mcp/metrics",
            headers={"Authorization": "Bearer token-student-a"},
        )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_http_cache_invalidate_admin_only(async_client):
    async with async_client as ac:
        resp = await ac.delete(
            "/api/mcp/cache/student.summariser.v1",
            headers={"Authorization": "Bearer token-admin"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["tool_name"] == "student.summariser.v1"
