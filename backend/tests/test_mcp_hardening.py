from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import app.mcp  # noqa: F401


def _make_ctx(
    *,
    role: str = "student",
    user_id: str = "user-student-a",
    file_id: str | None = None,
    submission_id: str | None = None,
):
    from app.mcp.schemas import ToolExecutionContext

    return ToolExecutionContext(
        user_id=user_id,
        role=role,
        correlation_id="hardening-test-corr",
        file_id=file_id,
        submission_id=submission_id,
    )


def _disable_externals(mcp_settings):
    mcp_settings.audit_enabled = False
    mcp_settings.ownership_check_enabled = False
    mcp_settings.metrics_enabled = False
    mcp_settings.cache_ttl_seconds = 0
    return mcp_settings


def _mock_llm_json(json_str: str):
    from app.mcp.llm_client import LLMCallResult

    return AsyncMock(return_value=LLMCallResult(ok=True, text=json_str, model_used="mock-model"))


def _mock_llm_fail(reason: str = "service unavailable"):
    from app.mcp.llm_client import LLMCallResult

    return AsyncMock(return_value=LLMCallResult(ok=False, error_reason=reason))


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        json_body=None,
        text: str = "",
        headers: dict[str, str] | None = None,
    ):
        self.status_code = status_code
        self._json_body = json_body
        self.text = text
        self.headers = headers or {}

    def json(self):
        if isinstance(self._json_body, Exception):
            raise self._json_body
        return self._json_body


def _fake_async_client_factory(*, response=None, exc: Exception | None = None):
    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            if exc is not None:
                raise exc
            return response

    return _FakeAsyncClient


class TestOwnershipHardening:
    @pytest.mark.asyncio
    async def test_own_resource_allowed(self, monkeypatch):
        from app.mcp import ownership

        monkeypatch.setattr(ownership, "_supabase_configured", lambda: True)

        async def fake_fetch_row(table: str, id_value: str):
            return {"id": id_value, "user_id": "user-student-a"}

        monkeypatch.setattr(ownership, "_fetch_row", fake_fetch_row)

        result = await ownership.check_resource_ownership(
            user_id="user-student-a",
            role="student",
            file_id="file-1",
            submission_id=None,
        )

        assert result.allowed is True
        assert result.denial_reason == ""

    @pytest.mark.asyncio
    async def test_foreign_resource_denied(self, monkeypatch):
        from app.mcp import ownership

        monkeypatch.setattr(ownership, "_supabase_configured", lambda: True)

        async def fake_fetch_row(table: str, id_value: str):
            return {"id": id_value, "user_id": "user-student-b"}

        monkeypatch.setattr(ownership, "_fetch_row", fake_fetch_row)

        result = await ownership.check_resource_ownership(
            user_id="user-student-a",
            role="student",
            file_id="file-foreign",
            submission_id=None,
        )

        assert result.allowed is False
        assert "does not own" in result.denial_reason

    @pytest.mark.asyncio
    async def test_missing_resource_denied(self, monkeypatch):
        from app.mcp import ownership

        monkeypatch.setattr(ownership, "_supabase_configured", lambda: True)

        async def fake_fetch_row(table: str, id_value: str):
            return None

        monkeypatch.setattr(ownership, "_fetch_row", fake_fetch_row)

        result = await ownership.check_resource_ownership(
            user_id="user-student-a",
            role="student",
            file_id="missing-file",
            submission_id=None,
        )

        assert result.allowed is False
        assert "Resource not found" in result.denial_reason

    @pytest.mark.asyncio
    async def test_professor_bypass(self, monkeypatch):
        from app.mcp import ownership

        fetch_mock = AsyncMock()
        monkeypatch.setattr(ownership, "_fetch_row", fetch_mock)

        result = await ownership.check_resource_ownership(
            user_id="user-professor-x",
            role="professor",
            file_id="file-1",
            submission_id="sub-1",
        )

        assert result.allowed is True
        fetch_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_admin_bypass(self, monkeypatch):
        from app.mcp import ownership

        fetch_mock = AsyncMock()
        monkeypatch.setattr(ownership, "_fetch_row", fetch_mock)

        result = await ownership.check_resource_ownership(
            user_id="user-admin",
            role="admin",
            file_id="file-1",
            submission_id="sub-1",
        )

        assert result.allowed is True
        fetch_mock.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("tool_name", "role", "payload"),
        [
            (
                "professor.rubric_evaluator.v1",
                "student",
                {"submission_text": "Some text", "rubric_criteria": ["Clarity"]},
            ),
            (
                "student.summariser.v1",
                "professor",
                {"text": "Some text"},
            ),
        ],
    )
    async def test_cross_role_denial_happens_before_ownership(
        self,
        monkeypatch,
        tool_name: str,
        role: str,
        payload: dict[str, object],
    ):
        from app.mcp import config as mcp_config
        from app.mcp.executor import execute_tool
        from app.mcp.schemas import MCPExecuteRequest

        _disable_externals(mcp_config.mcp_settings)
        mcp_config.mcp_settings.ownership_check_enabled = True

        ownership_probe = AsyncMock()
        monkeypatch.setattr("app.mcp.executor.enforce_ownership_policy", ownership_probe)

        result = await execute_tool(
            MCPExecuteRequest(
                tool_name=tool_name,
                payload=payload,
                context=_make_ctx(role=role, file_id="file-1"),
            )
        )

        assert result.ok is False
        assert result.error_code == "mcp.policy_denied"
        ownership_probe.assert_not_awaited()


class TestLLMClientHardening:
    @pytest.mark.asyncio
    async def test_call_llm_successful_response(self, monkeypatch):
        from app.mcp import llm_client
        from app.core.config import settings

        monkeypatch.setattr(settings, "llm_service_url", "http://llm-service:8030")
        monkeypatch.setattr(settings, "llm_service_secret", "test-secret")
        monkeypatch.setattr(
            llm_client.httpx,
            "AsyncClient",
            _fake_async_client_factory(
                response=_FakeResponse(
                    json_body={"response": '{"summary":"ok"}', "model_used": "gemma3:4b"},
                    headers={"x-llm-fallback": "true"},
                )
            ),
        )

        result = await llm_client.call_llm("hello", tool_name="student.summariser.v1")

        assert result.ok is True
        assert result.text == '{"summary":"ok"}'
        assert result.model_used == "gemma3:4b"
        assert result.used_fallback is True

    def test_parse_json_response_rejects_invalid_json(self):
        from app.mcp.llm_client import LLMCallResult, parse_json_response

        parsed = parse_json_response(
            LLMCallResult(ok=True, text="NOT JSON", model_used="gemma3:4b")
        )

        assert parsed.ok is False
        assert "JSON decode failed" in parsed.error_reason

    def test_parse_json_response_rejects_non_dict_shape(self):
        from app.mcp.llm_client import LLMCallResult, parse_json_response

        parsed = parse_json_response(
            LLMCallResult(ok=True, text='["not","a","dict"]', model_used="gemma3:4b")
        )

        assert parsed.ok is False
        assert "non-dict JSON" in parsed.error_reason

    @pytest.mark.asyncio
    async def test_call_llm_timeout(self, monkeypatch):
        from app.mcp import llm_client
        from app.core.config import settings

        monkeypatch.setattr(settings, "llm_service_url", "http://llm-service:8030")
        monkeypatch.setattr(settings, "llm_service_secret", "test-secret")
        monkeypatch.setattr(
            llm_client.httpx,
            "AsyncClient",
            _fake_async_client_factory(exc=httpx.TimeoutException("slow")),
        )

        result = await llm_client.call_llm("hello", tool_name="student.summariser.v1")

        assert result.ok is False
        assert "timed out" in result.error_reason

    @pytest.mark.asyncio
    async def test_call_llm_service_error(self, monkeypatch):
        from app.mcp import llm_client
        from app.core.config import settings

        monkeypatch.setattr(settings, "llm_service_url", "http://llm-service:8030")
        monkeypatch.setattr(settings, "llm_service_secret", "test-secret")
        monkeypatch.setattr(
            llm_client.httpx,
            "AsyncClient",
            _fake_async_client_factory(
                response=_FakeResponse(
                    status_code=503,
                    json_body={"detail": "service unavailable"},
                )
            ),
        )

        result = await llm_client.call_llm("hello", tool_name="student.summariser.v1")

        assert result.ok is False
        assert "HTTP 503" in result.error_reason
        assert "service unavailable" in result.error_reason

    @pytest.mark.asyncio
    async def test_deterministic_fallback_used_correctly(self, monkeypatch):
        from app.mcp import config as mcp_config
        from app.mcp.executor import execute_tool
        from app.mcp.schemas import MCPExecuteRequest

        _disable_externals(mcp_config.mcp_settings)
        mcp_config.mcp_settings.llm_enabled = True

        with patch("app.mcp.tools.student_summariser.call_llm", _mock_llm_fail("service down")):
            result = await execute_tool(
                MCPExecuteRequest(
                    tool_name="student.summariser.v1",
                    payload={"text": "Alpha. Beta. Gamma."},
                    context=_make_ctx(),
                )
            )

        assert result.ok is True
        assert result.meta.deterministic_fallback is True
        assert result.meta.llm_used is False
        assert any("LLM unavailable" in warning for warning in result.result["warnings"])


class TestCacheHardening:
    @pytest.mark.asyncio
    async def test_same_request_cache_hit(self):
        from app.mcp import cache
        from app.mcp import config as mcp_config
        from app.mcp.executor import execute_tool
        from app.mcp.schemas import MCPExecuteRequest

        _disable_externals(mcp_config.mcp_settings)
        mcp_config.mcp_settings.llm_enabled = False
        mcp_config.mcp_settings.cache_ttl_seconds = 300.0
        cache.invalidate("student.summariser.v1")

        req = MCPExecuteRequest(
            tool_name="student.summariser.v1",
            payload={"text": "Cache me once. Cache me twice.", "max_sentences": 1},
            context=_make_ctx(user_id="user-student-a", file_id="file-a"),
        )
        first = await execute_tool(req)
        second = await execute_tool(req)

        assert first.ok is True
        assert second.ok is True
        assert first.meta.cache_hit is False
        assert second.meta.cache_hit is True

        cache.invalidate("student.summariser.v1")

    @pytest.mark.asyncio
    async def test_changed_role_cache_miss(self):
        from app.mcp import cache
        from app.mcp import config as mcp_config
        from app.mcp.executor import execute_tool
        from app.mcp.schemas import MCPExecuteRequest

        _disable_externals(mcp_config.mcp_settings)
        mcp_config.mcp_settings.llm_enabled = False
        mcp_config.mcp_settings.cache_ttl_seconds = 300.0
        cache.invalidate("student.summariser.v1")

        student = await execute_tool(
            MCPExecuteRequest(
                tool_name="student.summariser.v1",
                payload={"text": "Same payload for role test."},
                context=_make_ctx(role="student", user_id="user-student-a"),
            )
        )
        admin = await execute_tool(
            MCPExecuteRequest(
                tool_name="student.summariser.v1",
                payload={"text": "Same payload for role test."},
                context=_make_ctx(role="admin", user_id="user-admin"),
            )
        )

        assert student.ok is True
        assert admin.ok is True
        assert student.meta.cache_hit is False
        assert admin.meta.cache_hit is False

        cache.invalidate("student.summariser.v1")

    @pytest.mark.asyncio
    async def test_failed_results_are_not_cached(self, monkeypatch):
        from app.mcp import cache
        from app.mcp import config as mcp_config
        from app.mcp.executor import execute_tool
        from app.mcp.models import ToolDefinition
        from app.mcp.registry import _REGISTRY
        from app.mcp.schemas import MCPExecuteRequest

        _disable_externals(mcp_config.mcp_settings)
        mcp_config.mcp_settings.cache_ttl_seconds = 300.0

        original = _REGISTRY["student.summariser.v1"]

        async def bad_handler(_inp, _ctx):
            return {"unexpected": True}

        bad_defn = ToolDefinition(
            tool_name="student.cache_failure_probe.v1",
            namespace=original.namespace,
            version=original.version,
            description="Temporary failing cache probe.",
            allowed_roles=original.allowed_roles,
            risk_level=original.risk_level,
            enabled=True,
            timeout_seconds=original.timeout_seconds,
            supports_idempotency=True,
            safe_for_multi_step=original.safe_for_multi_step,
            input_model=original.input_model,
            output_model=original.output_model,
            handler=bad_handler,
        )
        monkeypatch.setitem(_REGISTRY, bad_defn.tool_name, bad_defn)
        cache.invalidate(bad_defn.tool_name)

        req = MCPExecuteRequest(
            tool_name=bad_defn.tool_name,
            payload={"text": "This should fail validation."},
            context=_make_ctx(),
        )
        result1 = await execute_tool(req)
        result2 = await execute_tool(req)

        assert result1.ok is False
        assert result2.ok is False
        assert cache.invalidate(bad_defn.tool_name) == 0

    @pytest.mark.asyncio
    async def test_no_ownership_leakage_via_cached_responses(self):
        from app.mcp import cache
        from app.mcp import config as mcp_config
        from app.mcp.executor import execute_tool
        from app.mcp.schemas import MCPExecuteRequest

        _disable_externals(mcp_config.mcp_settings)
        mcp_config.mcp_settings.llm_enabled = False
        mcp_config.mcp_settings.cache_ttl_seconds = 300.0
        cache.invalidate("student.summariser.v1")

        payload = {"text": "Shared payload but isolated cache scope.", "max_sentences": 1}

        first = await execute_tool(
            MCPExecuteRequest(
                tool_name="student.summariser.v1",
                payload=payload,
                context=_make_ctx(user_id="user-student-a", file_id="file-a"),
            )
        )
        second = await execute_tool(
            MCPExecuteRequest(
                tool_name="student.summariser.v1",
                payload=payload,
                context=_make_ctx(user_id="user-student-b", file_id="file-b"),
            )
        )
        third = await execute_tool(
            MCPExecuteRequest(
                tool_name="student.summariser.v1",
                payload=payload,
                context=_make_ctx(user_id="user-student-a", file_id="file-a"),
            )
        )

        assert first.ok is True
        assert second.ok is True
        assert third.ok is True
        assert first.meta.cache_hit is False
        assert second.meta.cache_hit is False
        assert third.meta.cache_hit is True

        cache.invalidate("student.summariser.v1")


class TestMetricsHardening:
    @pytest.mark.asyncio
    async def test_success_llm_used_and_cache_hit_counters(self):
        from app.mcp import cache
        from app.mcp import config as mcp_config
        from app.mcp.executor import execute_tool
        from app.mcp.metrics import reset, snapshot
        from app.mcp.schemas import MCPExecuteRequest

        reset()
        _disable_externals(mcp_config.mcp_settings)
        mcp_config.mcp_settings.metrics_enabled = True
        mcp_config.mcp_settings.llm_enabled = True
        mcp_config.mcp_settings.cache_ttl_seconds = 300.0
        cache.invalidate("student.summariser.v1")

        llm_json = '{"summary": "This is a test summary.", "key_points": ["Testing", "Summary"]}'
        with patch("app.mcp.tools.student_summariser.call_llm", _mock_llm_json(llm_json)):
            req = MCPExecuteRequest(
                tool_name="student.summariser.v1",
                payload={"text": "First sentence. Second sentence."},
                context=_make_ctx(file_id="file-a"),
            )
            first = await execute_tool(req)
            second = await execute_tool(req)

        stats = snapshot()
        tool_stats = stats["tools"]["student.summariser.v1"]
        assert first.ok is True
        assert second.ok is True
        assert stats["totals"]["success"] == 2
        assert stats["totals"]["llm_used"] == 1
        assert stats["totals"]["cache_hit"] == 1
        assert tool_stats["cache_hit"] == 1
        assert tool_stats["llm_used"] == 1

        cache.invalidate("student.summariser.v1")

    @pytest.mark.asyncio
    async def test_policy_denial_and_timeout_counters(self, monkeypatch):
        import asyncio

        from app.mcp import config as mcp_config
        from app.mcp.executor import execute_tool
        from app.mcp.metrics import reset, snapshot
        from app.mcp.models import ToolDefinition
        from app.mcp.registry import _REGISTRY
        from app.mcp.schemas import MCPExecuteRequest

        reset()
        _disable_externals(mcp_config.mcp_settings)
        mcp_config.mcp_settings.metrics_enabled = True

        denied = await execute_tool(
            MCPExecuteRequest(
                tool_name="professor.rubric_evaluator.v1",
                payload={"submission_text": "Text", "rubric_criteria": ["Clarity"]},
                context=_make_ctx(role="student"),
            )
        )

        original = _REGISTRY["student.summariser.v1"]

        async def slow_handler(_inp, _ctx):
            await asyncio.sleep(999)

        slow_defn = ToolDefinition(
            tool_name=original.tool_name,
            namespace=original.namespace,
            version=original.version,
            description=original.description,
            allowed_roles=original.allowed_roles,
            risk_level=original.risk_level,
            enabled=original.enabled,
            timeout_seconds=0.01,
            supports_idempotency=original.supports_idempotency,
            safe_for_multi_step=original.safe_for_multi_step,
            input_model=original.input_model,
            output_model=original.output_model,
            handler=slow_handler,
        )
        monkeypatch.setitem(_REGISTRY, original.tool_name, slow_defn)
        try:
            timed_out = await execute_tool(
                MCPExecuteRequest(
                    tool_name="student.summariser.v1",
                    payload={"text": "Hello world."},
                    context=_make_ctx(),
                )
            )
        finally:
            monkeypatch.setitem(_REGISTRY, original.tool_name, original)

        stats = snapshot()
        assert denied.ok is False
        assert timed_out.ok is False
        assert stats["totals"]["failure"] == 2
        assert stats["totals"]["policy_denied"] == 1
        assert stats["totals"]["timeout"] == 1

    @pytest.mark.asyncio
    async def test_fallback_used_counter(self):
        from app.mcp import config as mcp_config
        from app.mcp.executor import execute_tool
        from app.mcp.metrics import reset, snapshot
        from app.mcp.schemas import MCPExecuteRequest

        reset()
        _disable_externals(mcp_config.mcp_settings)
        mcp_config.mcp_settings.metrics_enabled = True
        mcp_config.mcp_settings.llm_enabled = False

        result = await execute_tool(
            MCPExecuteRequest(
                tool_name="student.summariser.v1",
                payload={"text": "Alpha. Beta. Gamma."},
                context=_make_ctx(),
            )
        )

        stats = snapshot()
        assert result.ok is True
        assert result.meta.deterministic_fallback is True
        assert stats["totals"]["fallback_used"] == 1
        assert stats["tools"]["student.summariser.v1"]["fallback_used"] == 1
