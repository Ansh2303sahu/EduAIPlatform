from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import app.mcp  # noqa: F401


def _disable_externals(mcp_settings):
    mcp_settings.audit_enabled = False
    mcp_settings.ownership_check_enabled = False
    mcp_settings.llm_enabled = False
    mcp_settings.cache_ttl_seconds = 0
    return mcp_settings


def _student_request(**overrides):
    from app.mcp.orchestration_schemas import StudentReviewAssistRequest

    body = {
        "workflow_name": "student_review_assist",
        "payload": {
            "text": (
                "Introduction. Background. The thesis is that modular design improves "
                "maintainability. Discussion of evidence. Conclusion. References."
            ),
            "max_sentences": 2,
            "focus_mode": "overview",
            "preserve_key_terms": True,
            "submission_type": "essay",
            "expected_sections": ["Introduction", "Conclusion", "References"],
        },
    }
    body.update(overrides)
    return StudentReviewAssistRequest.model_validate(body)


def _professor_request(**overrides):
    from app.mcp.orchestration_schemas import ProfessorReviewAssistRequest

    body = {
        "workflow_name": "professor_review_assist",
        "payload": {
            "submission_text": (
                "This dissertation evaluates a web platform with clear methodology, "
                "results, and discussion sections."
            ),
            "rubric_criteria": ["Clarity", "Depth"],
            "grading_scale": "uk_honours",
            "strictness": "standard",
            "max_evidence_quotes": 1,
            "scores": [68.0, 72.0],
            "feedback_items": [],
            "expected_band_labels": ["Upper Second (60-69%)", "First (70%+)"],
            "final_summary": None,
        },
    }
    body.update(overrides)
    return ProfessorReviewAssistRequest.model_validate(body)


def _success_envelope(tool_name: str, *, request_id: str = "step-ok", result=None):
    from app.mcp.schemas import ExplainabilityMeta, MCPSuccessEnvelope

    return MCPSuccessEnvelope(
        tool_name=tool_name,
        tool_version="v1",
        request_id=request_id,
        correlation_id="orch-corr",
        result=result or {"ok": True},
        warnings=[],
        execution_ms=1.23,
        meta=ExplainabilityMeta(),
    )


def _failure_envelope(
    tool_name: str,
    *,
    request_id: str = "step-fail",
    error_code: str = "mcp.internal_error",
    message: str = "step failed",
    retryable: bool = False,
):
    from app.mcp.schemas import MCPFailureEnvelope

    return MCPFailureEnvelope(
        tool_name=tool_name,
        tool_version="v1",
        request_id=request_id,
        correlation_id="orch-corr",
        error_code=error_code,
        message=message,
        retryable=retryable,
        execution_ms=1.23,
    )


class TestWorkflowPlannerAndExecution:
    @pytest.mark.asyncio
    async def test_valid_student_workflow(self):
        from app.mcp import config as mcp_config
        from app.mcp.orchestrator import orchestrate_workflow

        _disable_externals(mcp_config.mcp_settings)

        result = await orchestrate_workflow(
            _student_request(),
            user_id="user-student-a",
            role="student",
            correlation_id="orch-student-1",
        )

        assert result.ok is True
        assert result.final_status == "completed"
        assert result.executed_steps == ["summarise_submission", "improve_structure"]
        assert [step.tool_name for step in result.step_results] == [
            "student.summariser.v1",
            "student.structure_improver.v1",
        ]
        assert all(step.envelope.ok for step in result.step_results)
        assert result.meta.executed_tool_order == [
            "student.summariser.v1",
            "student.structure_improver.v1",
        ]

    @pytest.mark.asyncio
    async def test_valid_professor_workflow(self):
        from app.mcp import config as mcp_config
        from app.mcp.orchestrator import orchestrate_workflow

        _disable_externals(mcp_config.mcp_settings)

        result = await orchestrate_workflow(
            _professor_request(),
            user_id="user-professor-x",
            role="professor",
            correlation_id="orch-professor-1",
        )

        assert result.ok is True
        assert result.final_status == "completed"
        assert result.executed_steps == ["evaluate_rubric", "check_consistency"]
        assert [step.tool_name for step in result.step_results] == [
            "professor.rubric_evaluator.v1",
            "professor.consistency_checker.v1",
        ]
        assert all(step.envelope.ok for step in result.step_results)

    @pytest.mark.asyncio
    async def test_role_mismatch_denial(self):
        from app.mcp import config as mcp_config
        from app.mcp.orchestrator import orchestrate_workflow

        _disable_externals(mcp_config.mcp_settings)

        result = await orchestrate_workflow(
            _student_request(),
            user_id="user-professor-x",
            role="professor",
            correlation_id="orch-role-mismatch",
        )

        assert result.ok is False
        assert result.final_status == "blocked"
        assert result.error_code == "mcp.policy_denied"
        assert result.step_results == []

    @pytest.mark.asyncio
    async def test_blocked_unsafe_tool_in_workflow(self, monkeypatch):
        from app.mcp import config as mcp_config
        from app.mcp.models import ToolDefinition
        from app.mcp.orchestrator import orchestrate_workflow
        from app.mcp.registry import _REGISTRY

        _disable_externals(mcp_config.mcp_settings)

        original = _REGISTRY["professor.rubric_evaluator.v1"]
        monkeypatch.setitem(
            _REGISTRY,
            "professor.rubric_evaluator.v1",
            ToolDefinition(
                tool_name=original.tool_name,
                namespace=original.namespace,
                version=original.version,
                description=original.description,
                allowed_roles=original.allowed_roles,
                risk_level=original.risk_level,
                enabled=original.enabled,
                timeout_seconds=original.timeout_seconds,
                supports_idempotency=original.supports_idempotency,
                safe_for_multi_step=False,
                input_model=original.input_model,
                output_model=original.output_model,
                handler=original.handler,
            ),
        )

        result = await orchestrate_workflow(
            _professor_request(),
            user_id="user-professor-x",
            role="professor",
            correlation_id="orch-unsafe",
        )

        assert result.ok is False
        assert result.final_status == "blocked"
        assert result.error_code == "mcp.policy_denied"
        assert result.message
        assert result.step_results == []

    @pytest.mark.asyncio
    async def test_max_step_enforcement(self):
        from app.mcp import config as mcp_config
        from app.mcp.orchestrator import orchestrate_workflow

        _disable_externals(mcp_config.mcp_settings)
        mcp_config.mcp_settings.orchestration_max_steps = 4

        result = await orchestrate_workflow(
            _student_request(max_steps=1),
            user_id="user-student-a",
            role="student",
            correlation_id="orch-max-steps",
        )

        assert result.ok is True
        assert result.final_status == "partial"
        assert result.meta.max_steps_applied == 1
        assert result.executed_steps == ["summarise_submission"]
        assert len(result.skipped_steps) == 1
        assert result.skipped_steps[0].reason == "max_steps_reached"

    @pytest.mark.asyncio
    async def test_step_failure_handling(self, monkeypatch):
        from app.mcp import config as mcp_config
        from app.mcp.orchestrator import orchestrate_workflow

        _disable_externals(mcp_config.mcp_settings)

        async def fake_execute_tool(req):
            return _failure_envelope(
                req.tool_name,
                error_code="mcp.output_validation",
                message="critical step failed",
            )

        monkeypatch.setattr("app.mcp.orchestrator.execute_tool", fake_execute_tool)

        result = await orchestrate_workflow(
            _student_request(),
            user_id="user-student-a",
            role="student",
            correlation_id="orch-critical-failure",
        )

        assert result.ok is False
        assert result.final_status == "failed"
        assert len(result.step_results) == 1
        assert result.step_results[0].envelope.ok is False
        assert result.skipped_steps[0].reason == "stopped_after_critical_failure"

    @pytest.mark.asyncio
    async def test_partial_completion_behavior(self, monkeypatch):
        from app.mcp import config as mcp_config
        from app.mcp.orchestrator import orchestrate_workflow

        _disable_externals(mcp_config.mcp_settings)

        async def fake_execute_tool(req):
            if req.tool_name == "student.summariser.v1":
                return _success_envelope(
                    req.tool_name,
                    result={"summary": "ok", "warnings": []},
                )
            return _failure_envelope(
                req.tool_name,
                error_code="mcp.timeout",
                message="non-critical step failed",
                retryable=True,
            )

        monkeypatch.setattr("app.mcp.orchestrator.execute_tool", fake_execute_tool)

        result = await orchestrate_workflow(
            _student_request(),
            user_id="user-student-a",
            role="student",
            correlation_id="orch-partial",
        )

        assert result.ok is True
        assert result.final_status == "partial"
        assert len(result.step_results) == 2
        assert result.step_results[0].envelope.ok is True
        assert result.step_results[1].envelope.ok is False
        assert result.meta.partial_completion is True

    @pytest.mark.asyncio
    async def test_policy_denial_propagation(self, monkeypatch):
        from app.mcp import config as mcp_config
        from app.mcp.orchestrator import orchestrate_workflow

        _disable_externals(mcp_config.mcp_settings)

        async def fake_execute_tool(req):
            if req.tool_name == "student.summariser.v1":
                return _success_envelope(req.tool_name, result={"summary": "ok"})
            return _failure_envelope(
                req.tool_name,
                error_code="mcp.policy_denied",
                message="ownership denied",
            )

        monkeypatch.setattr("app.mcp.orchestrator.execute_tool", fake_execute_tool)

        result = await orchestrate_workflow(
            _student_request(),
            user_id="user-student-a",
            role="student",
            correlation_id="orch-policy-denied",
        )

        assert result.ok is False
        assert result.final_status == "blocked"
        assert result.error_code == "mcp.policy_denied"
        assert result.step_results[-1].envelope.ok is False

    @pytest.mark.asyncio
    async def test_ownership_sensitive_workflow_execution(self, monkeypatch):
        from app.mcp import config as mcp_config
        from app.mcp.orchestrator import orchestrate_workflow
        from app.mcp.ownership import OwnershipResult

        _disable_externals(mcp_config.mcp_settings)
        mcp_config.mcp_settings.ownership_check_enabled = True

        monkeypatch.setattr(
            "app.mcp.policies.check_resource_ownership",
            AsyncMock(
                return_value=OwnershipResult(
                    allowed=False,
                    denial_reason="Student does not own file_id='file-foreign'",
                )
            ),
        )

        result = await orchestrate_workflow(
            _student_request(),
            user_id="user-student-a",
            role="student",
            correlation_id="orch-ownership",
            file_id="file-foreign",
        )

        assert result.ok is False
        assert result.final_status == "blocked"
        assert result.error_code == "mcp.policy_denied"
        assert len(result.step_results) == 1
        assert result.step_results[0].envelope.ok is False

    @pytest.mark.asyncio
    async def test_workflow_metrics_record_runs_failures_partials_and_order(
        self, monkeypatch
    ):
        from app.mcp import config as mcp_config
        from app.mcp.metrics import reset, snapshot
        from app.mcp.orchestrator import orchestrate_workflow

        _disable_externals(mcp_config.mcp_settings)
        mcp_config.mcp_settings.metrics_enabled = True
        reset()

        async def fake_execute_tool(req):
            if req.tool_name == "student.summariser.v1":
                return _success_envelope(req.tool_name)
            return _failure_envelope(req.tool_name, error_code="mcp.timeout")

        monkeypatch.setattr("app.mcp.orchestrator.execute_tool", fake_execute_tool)

        result = await orchestrate_workflow(
            _student_request(),
            user_id="user-student-a",
            role="student",
            correlation_id="orch-metrics",
        )

        metrics = snapshot()
        workflow_metrics = metrics["workflows"]["student_review_assist"]

        assert result.final_status == "partial"
        assert workflow_metrics["runs"] == 1
        assert workflow_metrics["partial_completions"] == 1
        assert workflow_metrics["step_failures"] == 1
        assert (
            workflow_metrics["executed_orders"][
                "student.summariser.v1 > student.structure_improver.v1"
            ]
            == 1
        )


class TestWorkflowHTTP:
    @pytest.mark.asyncio
    async def test_http_orchestrate_student_stable_response(self, async_client):
        from app.mcp import config as mcp_config

        _disable_externals(mcp_config.mcp_settings)

        async with async_client as ac:
            resp = await ac.post(
                "/api/mcp/orchestrate",
                json=_student_request().model_dump(),
                headers={"Authorization": "Bearer token-student-a"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert set(body) >= {
            "ok",
            "workflow_name",
            "request_id",
            "correlation_id",
            "executed_steps",
            "skipped_steps",
            "final_status",
            "step_results",
            "warnings",
            "meta",
        }
        assert body["workflow_name"] == "student_review_assist"

    @pytest.mark.asyncio
    async def test_http_list_workflows_student(self, async_client):
        async with async_client as ac:
            resp = await ac.get(
                "/api/mcp/workflows",
                headers={"Authorization": "Bearer token-student-a"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        names = {workflow["workflow_name"] for workflow in body["workflows"]}
        assert "student_review_assist" in names
        assert "professor_review_assist" not in names
