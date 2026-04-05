from __future__ import annotations

import sys
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import app.mcp  # noqa: F401

from app.mcp.schemas import MCPFailureEnvelope, MCPSuccessEnvelope


def _make_ctx(role: str = "student", user_id: str = "u1"):
    from app.mcp.schemas import ToolExecutionContext

    return ToolExecutionContext(
        user_id=user_id,
        role=role,
        correlation_id="citation-test-corr",
    )


def _assert_success(
    envelope: MCPSuccessEnvelope | MCPFailureEnvelope,
) -> MCPSuccessEnvelope:
    assert envelope.ok is True
    return cast(MCPSuccessEnvelope, envelope)


@pytest.fixture(autouse=True)
def _configure_mcp_for_tests():
    from app.mcp.cache import invalidate
    from app.mcp.config import mcp_settings

    mcp_settings.audit_enabled = False
    mcp_settings.ownership_check_enabled = False
    mcp_settings.metrics_enabled = False
    mcp_settings.cache_ttl_seconds = 0
    mcp_settings.llm_enabled = False
    invalidate("student.citation_helper.v1")


class TestCitationHelper:
    @pytest.mark.asyncio
    async def test_student_can_execute_tool_http(self, async_client):
        async with async_client as ac:
            resp = await ac.post(
                "/api/mcp/execute",
                json={
                    "tool_name": "student.citation_helper.v1",
                    "payload": {
                        "text": "Research shows that 42% of participants improved in 2023 after the intervention.",
                        "citation_style": "apa",
                    },
                },
                headers={"Authorization": "Bearer token-student-a"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["tool_name"] == "student.citation_helper.v1"
        assert body["meta"]["deterministic_fallback"] is True
        assert body["meta"]["llm_used"] is False

    @pytest.mark.asyncio
    async def test_professor_cannot_execute_tool_http(self, async_client, monkeypatch):
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
                    "tool_name": "student.citation_helper.v1",
                    "payload": {"text": "The protocol was introduced in 2021."},
                },
                headers={"Authorization": "Bearer token-professor-x"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is False
        assert body["error_code"] == "mcp.policy_denied"

    @pytest.mark.asyncio
    async def test_structured_success_envelope_shape(self):
        from app.mcp.executor import execute_tool
        from app.mcp.schemas import MCPExecuteRequest

        result = await execute_tool(
            MCPExecuteRequest(
                tool_name="student.citation_helper.v1",
                payload={
                    "text": "Studies indicate that completion rates increased by 18% in 2024.",
                    "citation_style": "harvard",
                    "max_flags": 3,
                },
                context=_make_ctx(),
            )
        )

        result = _assert_success(result)
        assert set(result.result) == {
            "flagged_segments",
            "citation_density_note",
            "style_warnings",
            "warnings",
            "confidence_note",
        }
        assert isinstance(result.result["flagged_segments"], list)

    @pytest.mark.asyncio
    async def test_numeric_statistical_claim_gets_flagged(self):
        from app.mcp.executor import execute_tool
        from app.mcp.schemas import MCPExecuteRequest

        result = await execute_tool(
            MCPExecuteRequest(
                tool_name="student.citation_helper.v1",
                payload={
                    "text": (
                        "Research shows that 42% of respondents reported improved outcomes in 2023 "
                        "after the redesigned service launched."
                    ),
                    "citation_style": "apa",
                },
                context=_make_ctx(),
            )
        )

        result = _assert_success(result)
        assert result.result["flagged_segments"]
        assert any(
            flag["severity"] in {"medium", "high"}
            and ("42%" in flag["text_excerpt"] or "2023" in flag["text_excerpt"])
            for flag in result.result["flagged_segments"]
        )

    @pytest.mark.asyncio
    async def test_existing_citation_markers_reduce_false_positives(self):
        from app.mcp.executor import execute_tool
        from app.mcp.schemas import MCPExecuteRequest

        result = await execute_tool(
            MCPExecuteRequest(
                tool_name="student.citation_helper.v1",
                payload={
                    "text": "According to Smith (2020), the intervention reduced risk by 25%.",
                    "citation_style": "apa",
                },
                context=_make_ctx(),
            )
        )

        result = _assert_success(result)
        assert result.result["flagged_segments"] == []

    @pytest.mark.asyncio
    async def test_long_uncited_paragraph_gets_flagged(self):
        from app.mcp.executor import execute_tool
        from app.mcp.schemas import MCPExecuteRequest

        paragraph = " ".join(
            [
                "Cybersecurity policy has changed substantially over the last decade and organisations now operate under complex governance expectations."
            ]
            * 12
        )
        result = await execute_tool(
            MCPExecuteRequest(
                tool_name="student.citation_helper.v1",
                payload={
                    "text": paragraph,
                    "citation_style": "generic",
                    "sensitivity": "medium",
                },
                context=_make_ctx(),
            )
        )

        result = _assert_success(result)
        assert any(
            "long paragraph" in flag["reason"].lower()
            for flag in result.result["flagged_segments"]
        )

    @pytest.mark.asyncio
    async def test_style_warnings_returned_correctly(self):
        from app.mcp.executor import execute_tool
        from app.mcp.schemas import MCPExecuteRequest

        result = await execute_tool(
            MCPExecuteRequest(
                tool_name="student.citation_helper.v1",
                payload={
                    "text": "Smith (2020) argues that reflective practice improves retention.",
                    "citation_style": "ieee",
                },
                context=_make_ctx(),
            )
        )

        result = _assert_success(result)
        assert any("IEEE" in warning for warning in result.result["style_warnings"])

    @pytest.mark.asyncio
    async def test_deterministic_fallback_when_llm_unavailable(self):
        from app.mcp.executor import execute_tool
        from app.mcp.llm_client import LLMCallResult
        from app.mcp.schemas import MCPExecuteRequest
        from app.mcp.config import mcp_settings

        mcp_settings.llm_enabled = True
        with patch(
            "app.mcp.tools.student_citation_helper.call_llm",
            AsyncMock(return_value=LLMCallResult(ok=False, error_reason="service unavailable")),
        ):
            result = await execute_tool(
                MCPExecuteRequest(
                    tool_name="student.citation_helper.v1",
                    payload={
                        "text": "Studies indicate that 61% of teams improved release quality in 2024.",
                        "citation_style": "harvard",
                    },
                    context=_make_ctx(),
                )
            )

        result = _assert_success(result)
        assert result.meta.deterministic_fallback is True
        assert result.meta.llm_used is False
        assert any("LLM refinement unavailable" in warning for warning in result.result["warnings"])

    @pytest.mark.asyncio
    async def test_no_fabricated_references_appear_in_output(self):
        from app.mcp.executor import execute_tool
        from app.mcp.schemas import MCPExecuteRequest

        result = await execute_tool(
            MCPExecuteRequest(
                tool_name="student.citation_helper.v1",
                payload={
                    "text": "It is widely known that the protocol was introduced in 2019 and adoption rose by 30%.",
                    "citation_style": "apa",
                },
                context=_make_ctx(),
            )
        )

        result = _assert_success(result)
        combined = str(result.result)
        assert "http://" not in combined
        assert "https://" not in combined
        assert "doi" not in combined.lower()
        assert "retrieved from" not in combined.lower()

    @pytest.mark.asyncio
    async def test_http_response_schema_stable(self, async_client):
        async with async_client as ac:
            resp = await ac.post(
                "/api/mcp/execute",
                json={
                    "tool_name": "student.citation_helper.v1",
                    "payload": {
                        "text": "Experts say that 67% of users preferred the revised workflow in 2022.",
                        "citation_style": "generic",
                        "max_flags": 4,
                    },
                    "correlation_id": "citation-http-001",
                },
                headers={"Authorization": "Bearer token-student-a"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["tool_name"] == "student.citation_helper.v1"
        assert body["tool_version"] == "v1"
        assert body["correlation_id"] == "citation-http-001"
        assert "request_id" in body
        assert set(body["result"]) == {
            "flagged_segments",
            "citation_density_note",
            "style_warnings",
            "warnings",
            "confidence_note",
        }

    def test_citation_helper_is_manual_only(self):
        from app.mcp.planner import list_visible_workflows
        from app.mcp.registry import resolve_tool

        defn = resolve_tool("student.citation_helper.v1")
        assert defn.safe_for_multi_step is False
        assert all(
            step.tool_name != "student.citation_helper.v1"
            for workflow in list_visible_workflows("student")
            for step in workflow.steps
        )
