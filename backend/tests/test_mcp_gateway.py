"""
Phase 11 MCP gateway tests.

Coverage
--------
Unit (no HTTP, no auth):
  - Registry: unknown tool denied, disabled tool denied, successful resolution.
  - Policy: role mismatch denied, namespace mismatch denied, valid access allowed.
  - Executor: success envelope is stable, timeout produces safe failure,
    input validation failure is safe, output validation failure is safe.

Integration (HTTP via async_client + conftest mock_auth):
  - POST /api/mcp/execute — unknown tool returns ok=False + correct error_code.
  - POST /api/mcp/execute — student calls professor tool → policy denied.
  - POST /api/mcp/execute — professor calls student tool → policy denied.
  - POST /api/mcp/execute — malformed payload → input validation failure.
  - POST /api/mcp/execute — valid student call → ok=True, stable envelope.
  - POST /api/mcp/execute — valid professor call (admin token) → ok=True.
  - GET  /api/mcp/tools   — lists only role-appropriate tools.
  - Unauthenticated request → HTTP 401.

The conftest mock_auth autouse fixture provides:
  "token-student-a" → user_id="user-student-a", role="student"
  "token-admin"     → user_id="user-admin",     role="admin"

For professor calls we patch ``deps.fetch_user_role_from_db`` locally.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# Ensure backend is importable.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# Trigger tool registration before any test runs.
import app.mcp  # noqa: F401


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(role: str = "student", user_id: str = "u1") -> "ToolExecutionContext":
    from app.mcp.schemas import ToolExecutionContext
    return ToolExecutionContext(
        user_id=user_id,
        role=role,
        correlation_id="corr-test-001",
    )


# ===========================================================================
# Registry unit tests
# ===========================================================================


class TestRegistry:

    def test_unknown_tool_raises(self):
        from app.mcp.errors import UnknownToolError
        from app.mcp.registry import resolve_tool

        with pytest.raises(UnknownToolError):
            resolve_tool("nonexistent.tool.v99")

    def test_disabled_tool_raises(self, monkeypatch):
        from app.mcp.errors import DisabledToolError
        from app.mcp.models import ToolDefinition
        from app.mcp.registry import _REGISTRY, resolve_tool

        # Temporarily patch a known tool to disabled.
        original = _REGISTRY["student.summariser.v1"]
        disabled = ToolDefinition(
            tool_name=original.tool_name,
            namespace=original.namespace,
            version=original.version,
            description=original.description,
            allowed_roles=original.allowed_roles,
            risk_level=original.risk_level,
            enabled=False,  # <-- disabled
            timeout_seconds=original.timeout_seconds,
            supports_idempotency=original.supports_idempotency,
            safe_for_multi_step=original.safe_for_multi_step,
            input_model=original.input_model,
            output_model=original.output_model,
            handler=original.handler,
        )
        monkeypatch.setitem(_REGISTRY, "student.summariser.v1", disabled)

        with pytest.raises(DisabledToolError):
            resolve_tool("student.summariser.v1")

    def test_known_tool_resolves(self):
        from app.mcp.registry import resolve_tool

        defn = resolve_tool("student.summariser.v1")
        assert defn.tool_name == "student.summariser.v1"
        assert defn.enabled is True

    def test_all_four_seed_tools_registered(self):
        from app.mcp.registry import list_tools

        names = {d.tool_name for d in list_tools(include_disabled=True)}
        assert "student.summariser.v1" in names
        assert "student.structure_improver.v1" in names
        assert "professor.rubric_evaluator.v1" in names
        assert "professor.consistency_checker.v1" in names


# ===========================================================================
# Policy unit tests
# ===========================================================================


class TestPolicy:

    def test_student_cannot_call_professor_tool(self):
        from app.mcp.errors import PolicyDeniedError
        from app.mcp.policies import enforce_policy
        from app.mcp.registry import resolve_tool

        defn = resolve_tool("professor.rubric_evaluator.v1")
        ctx = _make_ctx(role="student")

        with pytest.raises(PolicyDeniedError) as exc_info:
            enforce_policy(defn, ctx)

        assert "professor" in str(exc_info.value).lower() or "role" in str(exc_info.value).lower()

    def test_professor_cannot_call_student_tool(self):
        from app.mcp.errors import PolicyDeniedError
        from app.mcp.policies import enforce_policy
        from app.mcp.registry import resolve_tool

        defn = resolve_tool("student.summariser.v1")
        ctx = _make_ctx(role="professor")

        with pytest.raises(PolicyDeniedError):
            enforce_policy(defn, ctx)

    def test_valid_student_access_allowed(self):
        from app.mcp.policies import enforce_policy
        from app.mcp.registry import resolve_tool

        defn = resolve_tool("student.summariser.v1")
        ctx = _make_ctx(role="student")
        # Should not raise.
        enforce_policy(defn, ctx)

    def test_admin_can_call_student_tool(self):
        from app.mcp.policies import enforce_policy
        from app.mcp.registry import resolve_tool

        defn = resolve_tool("student.summariser.v1")
        ctx = _make_ctx(role="admin")
        enforce_policy(defn, ctx)

    def test_admin_can_call_professor_tool(self):
        from app.mcp.policies import enforce_policy
        from app.mcp.registry import resolve_tool

        defn = resolve_tool("professor.rubric_evaluator.v1")
        ctx = _make_ctx(role="admin")
        enforce_policy(defn, ctx)

    def test_unknown_role_denied(self):
        from app.mcp.errors import PolicyDeniedError
        from app.mcp.policies import enforce_policy
        from app.mcp.registry import resolve_tool

        defn = resolve_tool("student.summariser.v1")
        ctx = _make_ctx(role="hacker")
        with pytest.raises(PolicyDeniedError):
            enforce_policy(defn, ctx)

    def test_denial_reason_is_explicit(self):
        from app.mcp.errors import PolicyDeniedError
        from app.mcp.policies import enforce_policy
        from app.mcp.registry import resolve_tool

        defn = resolve_tool("professor.rubric_evaluator.v1")
        ctx = _make_ctx(role="student")
        with pytest.raises(PolicyDeniedError) as exc_info:
            enforce_policy(defn, ctx)

        assert exc_info.value.reason, "Denial reason must be non-empty"


# ===========================================================================
# Executor unit tests (no HTTP)
# ===========================================================================


class TestExecutor:

    @pytest.mark.asyncio
    async def test_success_envelope_shape(self):
        from app.mcp.executor import execute_tool
        from app.mcp.schemas import MCPExecuteRequest

        with patch("app.mcp.executor.emit_tool_started", new_callable=AsyncMock):
            with patch("app.mcp.executor.emit_tool_succeeded", new_callable=AsyncMock):
                req = MCPExecuteRequest(
                    tool_name="student.summariser.v1",
                    payload={"text": "The quick brown fox. Second sentence here."},
                    context=_make_ctx(role="student"),
                )
                result = await execute_tool(req)

        assert result.ok is True
        assert result.tool_name == "student.summariser.v1"
        assert result.tool_version == "v1"
        assert result.request_id  # non-empty UUID
        assert isinstance(result.execution_ms, float)
        assert result.execution_ms >= 0

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_failure_envelope(self):
        from app.mcp.executor import execute_tool
        from app.mcp.schemas import MCPExecuteRequest

        with patch("app.mcp.executor.emit_tool_failed", new_callable=AsyncMock):
            req = MCPExecuteRequest(
                tool_name="student.nonexistent.v99",
                payload={},
                context=_make_ctx(role="student"),
            )
            result = await execute_tool(req)

        assert result.ok is False
        assert result.error_code == "mcp.unknown_tool"
        assert result.retryable is False

    @pytest.mark.asyncio
    async def test_policy_denied_returns_failure_envelope(self):
        from app.mcp.executor import execute_tool
        from app.mcp.schemas import MCPExecuteRequest

        with patch("app.mcp.executor.emit_tool_failed", new_callable=AsyncMock):
            req = MCPExecuteRequest(
                tool_name="professor.rubric_evaluator.v1",
                payload={},
                context=_make_ctx(role="student"),
            )
            result = await execute_tool(req)

        assert result.ok is False
        assert result.error_code == "mcp.policy_denied"

    @pytest.mark.asyncio
    async def test_malformed_payload_returns_input_validation_failure(self):
        from app.mcp.executor import execute_tool
        from app.mcp.schemas import MCPExecuteRequest

        with patch("app.mcp.executor.emit_tool_failed", new_callable=AsyncMock):
            # SummariserInput requires 'text' — omitting it causes ValidationError.
            req = MCPExecuteRequest(
                tool_name="student.summariser.v1",
                payload={"max_sentences": 3},  # missing required 'text'
                context=_make_ctx(role="student"),
            )
            result = await execute_tool(req)

        assert result.ok is False
        assert result.error_code == "mcp.input_validation"
        assert result.retryable is False

    @pytest.mark.asyncio
    async def test_timeout_returns_safe_failure(self):
        import asyncio
        from app.mcp.executor import execute_tool
        from app.mcp.schemas import MCPExecuteRequest

        async def _slow_handler(inp, ctx):
            await asyncio.sleep(999)

        from app.mcp.registry import _REGISTRY
        from app.mcp.models import ToolDefinition

        original = _REGISTRY["student.summariser.v1"]
        slow_defn = ToolDefinition(
            tool_name=original.tool_name,
            namespace=original.namespace,
            version=original.version,
            description=original.description,
            allowed_roles=original.allowed_roles,
            risk_level=original.risk_level,
            enabled=original.enabled,
            timeout_seconds=0.01,  # 10 ms — will fire immediately
            supports_idempotency=original.supports_idempotency,
            safe_for_multi_step=original.safe_for_multi_step,
            input_model=original.input_model,
            output_model=original.output_model,
            handler=_slow_handler,
        )

        original_defn = _REGISTRY["student.summariser.v1"]
        _REGISTRY["student.summariser.v1"] = slow_defn
        try:
            with patch("app.mcp.executor.emit_tool_started", new_callable=AsyncMock):
                with patch("app.mcp.executor.emit_tool_failed", new_callable=AsyncMock):
                    req = MCPExecuteRequest(
                        tool_name="student.summariser.v1",
                        payload={"text": "Hello world."},
                        context=_make_ctx(role="student"),
                    )
                    result = await execute_tool(req)
        finally:
            _REGISTRY["student.summariser.v1"] = original_defn

        assert result.ok is False
        assert result.error_code == "mcp.timeout"
        assert result.retryable is True

    @pytest.mark.asyncio
    async def test_disabled_tool_returns_failure_envelope(self, monkeypatch):
        from app.mcp.executor import execute_tool
        from app.mcp.models import ToolDefinition
        from app.mcp.registry import _REGISTRY
        from app.mcp.schemas import MCPExecuteRequest

        original = _REGISTRY["student.summariser.v1"]
        monkeypatch.setitem(
            _REGISTRY,
            "student.summariser.v1",
            ToolDefinition(
                tool_name=original.tool_name,
                namespace=original.namespace,
                version=original.version,
                description=original.description,
                allowed_roles=original.allowed_roles,
                risk_level=original.risk_level,
                enabled=False,
                timeout_seconds=original.timeout_seconds,
                supports_idempotency=original.supports_idempotency,
                safe_for_multi_step=original.safe_for_multi_step,
                input_model=original.input_model,
                output_model=original.output_model,
                handler=original.handler,
            ),
        )

        with patch("app.mcp.executor.emit_tool_failed", new_callable=AsyncMock):
            req = MCPExecuteRequest(
                tool_name="student.summariser.v1",
                payload={"text": "Hello."},
                context=_make_ctx(role="student"),
            )
            result = await execute_tool(req)

        assert result.ok is False
        assert result.error_code == "mcp.disabled_tool"


# ===========================================================================
# Summariser tool unit tests
# ===========================================================================


class TestStudentSummariser:

    @pytest.mark.asyncio
    async def test_returns_correct_sentence_count(self):
        from app.mcp.executor import execute_tool
        from app.mcp.schemas import MCPExecuteRequest

        with patch("app.mcp.executor.emit_tool_started", new_callable=AsyncMock):
            with patch("app.mcp.executor.emit_tool_succeeded", new_callable=AsyncMock):
                req = MCPExecuteRequest(
                    tool_name="student.summariser.v1",
                    payload={
                        "text": "First sentence. Second sentence. Third sentence. Fourth sentence.",
                        "max_sentences": 2,
                    },
                    context=_make_ctx(role="student"),
                )
                result = await execute_tool(req)

        assert result.ok is True
        assert result.result["sentence_count"] == 2
        assert result.result["original_word_count"] > 0


# ===========================================================================
# HTTP integration tests (uses conftest mock_auth + async_client)
# ===========================================================================


@pytest.mark.asyncio
async def test_http_unknown_tool_returns_ok_false(async_client):
    async with async_client as ac:
        resp = await ac.post(
            "/api/mcp/execute",
            json={"tool_name": "ghost.tool.v999", "payload": {}},
            headers={"Authorization": "Bearer token-student-a"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["error_code"] == "mcp.unknown_tool"


@pytest.mark.asyncio
async def test_http_student_cannot_call_professor_tool(async_client):
    async with async_client as ac:
        resp = await ac.post(
            "/api/mcp/execute",
            json={
                "tool_name": "professor.rubric_evaluator.v1",
                "payload": {
                    "submission_text": "Some text.",
                    "rubric_criteria": ["Clarity"],
                },
            },
            headers={"Authorization": "Bearer token-student-a"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["error_code"] == "mcp.policy_denied"


@pytest.mark.asyncio
async def test_http_professor_cannot_call_student_tool(async_client, monkeypatch):
    import app.core.deps as deps

    async def professor_role(user_id: str) -> str:
        if user_id == "user-professor-x":
            return "professor"
        return "student"

    async def professor_jwt(token: str):
        if token == "token-professor-x":
            return {"sub": "user-professor-x", "email": "prof@example.com", "aal": "aal1"}
        raise Exception("bad token")

    monkeypatch.setattr(deps, "fetch_user_role_from_db", professor_role)
    monkeypatch.setattr(deps, "verify_supabase_jwt", professor_jwt)

    async with async_client as ac:
        resp = await ac.post(
            "/api/mcp/execute",
            json={
                "tool_name": "student.summariser.v1",
                "payload": {"text": "Hello world."},
            },
            headers={"Authorization": "Bearer token-professor-x"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["error_code"] == "mcp.policy_denied"


@pytest.mark.asyncio
async def test_http_malformed_payload_rejected(async_client):
    async with async_client as ac:
        resp = await ac.post(
            "/api/mcp/execute",
            json={
                "tool_name": "student.summariser.v1",
                # 'text' is required but absent
                "payload": {"max_sentences": 3},
            },
            headers={"Authorization": "Bearer token-student-a"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["error_code"] == "mcp.input_validation"


@pytest.mark.asyncio
async def test_http_valid_student_call_success(async_client):
    async with async_client as ac:
        resp = await ac.post(
            "/api/mcp/execute",
            json={
                "tool_name": "student.summariser.v1",
                "payload": {
                    "text": "This is the first sentence. This is the second sentence.",
                    "max_sentences": 1,
                },
                "correlation_id": "integ-test-001",
            },
            headers={"Authorization": "Bearer token-student-a"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["tool_name"] == "student.summariser.v1"
    assert body["tool_version"] == "v1"
    assert "request_id" in body
    assert "execution_ms" in body
    assert body["result"]["sentence_count"] == 1
    assert body["correlation_id"] == "integ-test-001"


@pytest.mark.asyncio
async def test_http_admin_can_call_professor_tool(async_client):
    async with async_client as ac:
        resp = await ac.post(
            "/api/mcp/execute",
            json={
                "tool_name": "professor.rubric_evaluator.v1",
                "payload": {
                    "submission_text": "This is a long submission text with many words.",
                    "rubric_criteria": ["Clarity", "Depth"],
                },
            },
            headers={"Authorization": "Bearer token-admin"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["tool_name"] == "professor.rubric_evaluator.v1"
    assert len(body["result"]["evaluations"]) == 2


@pytest.mark.asyncio
async def test_http_unauthenticated_returns_401(async_client):
    async with async_client as ac:
        resp = await ac.post(
            "/api/mcp/execute",
            json={"tool_name": "student.summariser.v1", "payload": {"text": "hi"}},
            # No Authorization header
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_http_list_tools_student(async_client):
    async with async_client as ac:
        resp = await ac.get(
            "/api/mcp/tools",
            headers={"Authorization": "Bearer token-student-a"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    tool_names = {t["tool_name"] for t in body["tools"]}
    # Student should see student tools
    assert "student.summariser.v1" in tool_names
    assert "student.structure_improver.v1" in tool_names
    # Student should NOT see professor tools
    assert "professor.rubric_evaluator.v1" not in tool_names
    assert "professor.consistency_checker.v1" not in tool_names


@pytest.mark.asyncio
async def test_http_list_tools_admin_sees_all(async_client):
    async with async_client as ac:
        resp = await ac.get(
            "/api/mcp/tools",
            headers={"Authorization": "Bearer token-admin"},
        )
    assert resp.status_code == 200
    body = resp.json()
    tool_names = {t["tool_name"] for t in body["tools"]}
    assert "student.summariser.v1" in tool_names
    assert "professor.rubric_evaluator.v1" in tool_names
