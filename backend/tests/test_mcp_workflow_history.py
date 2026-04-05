from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import app.mcp  # noqa: F401


def _student_request():
    from app.mcp.orchestration_schemas import StudentReviewAssistRequest

    return StudentReviewAssistRequest.model_validate(
        {
            "workflow_name": "student_review_assist",
            "payload": {
                "text": (
                    "This essay argues that modular architecture improves maintainability "
                    "and supports long-term system evolution."
                ),
                "max_sentences": 2,
                "focus_mode": "overview",
                "preserve_key_terms": True,
                "submission_type": "essay",
                "expected_sections": ["Introduction", "Conclusion", "References"],
            },
        }
    )


def _success_envelope(tool_name: str, *, result=None, cache_hit=False, llm_used=False, deterministic_fallback=False):
    from app.mcp.schemas import ExplainabilityMeta, MCPSuccessEnvelope

    return MCPSuccessEnvelope(
        tool_name=tool_name,
        tool_version="v1",
        request_id="req-step",
        correlation_id="corr-step",
        result=result or {"ok": True},
        warnings=["warn"] if cache_hit else [],
        execution_ms=12.5,
        meta=ExplainabilityMeta(
            cache_hit=cache_hit,
            llm_used=llm_used,
            deterministic_fallback=deterministic_fallback,
        ),
    )


def _failure_envelope(tool_name: str, *, error_code="mcp.timeout", message="step failed"):
    from app.mcp.schemas import MCPFailureEnvelope

    return MCPFailureEnvelope(
        tool_name=tool_name,
        tool_version="v1",
        request_id="req-step-fail",
        correlation_id="corr-step",
        error_code=error_code,
        message=message,
        retryable=(error_code == "mcp.timeout"),
        execution_ms=18.0,
    )


def _workflow_response(*, final_status: str):
    from app.mcp.orchestration_schemas import (
        WorkflowExplainabilityMeta,
        WorkflowResponse,
        WorkflowSkippedStep,
        WorkflowStepResult,
    )

    if final_status == "completed":
        steps = [
            WorkflowStepResult(
                step_name="summarise_submission",
                tool_name="student.summariser.v1",
                index=1,
                critical=True,
                envelope=_success_envelope(
                    "student.summariser.v1",
                    llm_used=True,
                ),
            ),
            WorkflowStepResult(
                step_name="improve_structure",
                tool_name="student.structure_improver.v1",
                index=2,
                critical=False,
                envelope=_success_envelope(
                    "student.structure_improver.v1",
                    cache_hit=True,
                    deterministic_fallback=True,
                ),
            ),
        ]
        skipped = []
        warnings = []
        meta = WorkflowExplainabilityMeta(
            max_steps_applied=2,
            continue_on_non_critical_failure=True,
            executed_tool_order=[
                "student.summariser.v1",
                "student.structure_improver.v1",
            ],
            cache_hit_steps=1,
            llm_used_steps=1,
            fallback_steps=1,
            partial_completion=False,
            step_failures=0,
            stopped_reason="completed",
        )
        error_code = None
        message = None
    elif final_status == "partial":
        steps = [
            WorkflowStepResult(
                step_name="summarise_submission",
                tool_name="student.summariser.v1",
                index=1,
                critical=True,
                envelope=_success_envelope("student.summariser.v1", llm_used=True),
            ),
            WorkflowStepResult(
                step_name="improve_structure",
                tool_name="student.structure_improver.v1",
                index=2,
                critical=False,
                envelope=_failure_envelope("student.structure_improver.v1"),
            ),
        ]
        skipped = []
        warnings = ["Non-critical step failed."]
        meta = WorkflowExplainabilityMeta(
            max_steps_applied=2,
            continue_on_non_critical_failure=True,
            executed_tool_order=[
                "student.summariser.v1",
                "student.structure_improver.v1",
            ],
            cache_hit_steps=0,
            llm_used_steps=1,
            fallback_steps=0,
            partial_completion=True,
            step_failures=1,
            stopped_reason="partial_completion",
        )
        error_code = "mcp.timeout"
        message = "step failed"
    else:
        steps = [
            WorkflowStepResult(
                step_name="summarise_submission",
                tool_name="student.summariser.v1",
                index=1,
                critical=True,
                envelope=_failure_envelope(
                    "student.summariser.v1",
                    error_code="mcp.policy_denied",
                    message="ownership denied",
                ),
            )
        ]
        skipped = [
            WorkflowSkippedStep(
                step_name="improve_structure",
                tool_name="student.structure_improver.v1",
                reason="stopped_after_policy_denial",
            )
        ]
        warnings = ["Workflow stopped by policy."]
        meta = WorkflowExplainabilityMeta(
            max_steps_applied=2,
            continue_on_non_critical_failure=True,
            executed_tool_order=["student.summariser.v1"],
            cache_hit_steps=0,
            llm_used_steps=0,
            fallback_steps=0,
            partial_completion=False,
            step_failures=1,
            stopped_reason="policy_denied:summarise_submission",
        )
        error_code = "mcp.policy_denied"
        message = "ownership denied"

    return WorkflowResponse(
        ok=final_status in {"completed", "partial"},
        workflow_name="student_review_assist",
        request_id="workflow-request-id",
        correlation_id="workflow-correlation-id",
        final_status=final_status,
        executed_steps=[step.step_name for step in steps],
        skipped_steps=skipped,
        step_results=steps,
        warnings=warnings,
        meta=meta,
        error_code=error_code,
        message=message,
    )


class _FakeRepo:
    def __init__(self):
        self.run_row = None
        self.step_rows = None
        self.list_kwargs = None
        self.detail_row = None
        self.detail_steps = None
        self.failed_steps = []

    async def insert_run(self, row):
        self.run_row = row
        return row

    async def insert_steps(self, rows):
        self.step_rows = rows

    async def list_runs(self, **kwargs):
        self.list_kwargs = kwargs
        if self.detail_row is None:
            return [], 0
        return [self.detail_row], 1

    async def get_run(self, workflow_run_id):
        if self.detail_row and self.detail_row["workflow_run_id"] == workflow_run_id:
            return self.detail_row
        return None

    async def list_steps(self, workflow_run_id):
        if self.detail_row and self.detail_row["workflow_run_id"] == workflow_run_id:
            return self.detail_steps or []
        return []

    async def list_failed_steps(self, *, limit=500):
        return self.failed_steps


class TestWorkflowHistoryService:
    @pytest.mark.asyncio
    async def test_successful_workflow_history_write(self):
        from app.mcp.workflow_history_service import WorkflowHistoryService

        repo = _FakeRepo()
        service = WorkflowHistoryService(repo=repo)
        req = _student_request()
        response = _workflow_response(final_status="completed")

        await service.record_workflow_run(
            workflow_run_id="11111111-1111-1111-1111-111111111111",
            workflow_req=req,
            workflow_response=response,
            user_id="user-student-a",
            role="student",
            correlation_id="workflow-correlation-id",
            request_id="workflow-request-id",
            file_id="file-1",
            submission_id=None,
            started_at=datetime(2026, 4, 5, 12, 0, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 4, 5, 12, 0, 1, tzinfo=timezone.utc),
            duration_ms=1000.0,
        )

        assert repo.run_row["final_status"] == "completed"
        assert repo.run_row["step_count"] == 2
        assert repo.run_row["cache_hits_count"] == 1
        assert repo.run_row["llm_steps_count"] == 1
        assert repo.run_row["fallback_steps_count"] == 1
        assert repo.run_row["ownership_context_present"] is True
        assert repo.step_rows[0]["step_status"] == "completed"
        assert repo.step_rows[1]["step_status"] == "completed"
        dumped = json.dumps({"run": repo.run_row, "steps": repo.step_rows})
        assert "modular architecture improves maintainability" not in dumped
        assert '"text":' not in dumped
        assert '"submission_text":' not in dumped

    @pytest.mark.asyncio
    async def test_partial_workflow_history_write(self):
        from app.mcp.workflow_history_service import WorkflowHistoryService

        repo = _FakeRepo()
        service = WorkflowHistoryService(repo=repo)

        await service.record_workflow_run(
            workflow_run_id="22222222-2222-2222-2222-222222222222",
            workflow_req=_student_request(),
            workflow_response=_workflow_response(final_status="partial"),
            user_id="user-student-a",
            role="student",
            correlation_id="workflow-correlation-id",
            request_id="workflow-request-id",
            file_id=None,
            submission_id=None,
            started_at=datetime(2026, 4, 5, 12, 0, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 4, 5, 12, 0, 1, tzinfo=timezone.utc),
            duration_ms=1000.0,
        )

        assert repo.run_row["final_status"] == "partial"
        assert repo.run_row["partial_reason"] == "partial_completion"
        assert repo.step_rows[0]["step_status"] == "completed"
        assert repo.step_rows[1]["step_status"] == "failed"
        assert repo.step_rows[1]["error_code"] == "mcp.timeout"

    @pytest.mark.asyncio
    async def test_blocked_workflow_history_write(self):
        from app.mcp.workflow_history_service import WorkflowHistoryService

        repo = _FakeRepo()
        service = WorkflowHistoryService(repo=repo)

        await service.record_workflow_run(
            workflow_run_id="33333333-3333-3333-3333-333333333333",
            workflow_req=_student_request(),
            workflow_response=_workflow_response(final_status="blocked"),
            user_id="user-student-a",
            role="student",
            correlation_id="workflow-correlation-id",
            request_id="workflow-request-id",
            file_id="file-1",
            submission_id=None,
            started_at=datetime(2026, 4, 5, 12, 0, 0, tzinfo=timezone.utc),
            finished_at=datetime(2026, 4, 5, 12, 0, 1, tzinfo=timezone.utc),
            duration_ms=1000.0,
        )

        assert repo.run_row["final_status"] == "blocked"
        assert repo.run_row["blocked_reason"] == "policy_denied:summarise_submission"
        assert repo.run_row["error_code"] == "mcp.policy_denied"
        assert repo.step_rows[0]["step_status"] == "failed"

    @pytest.mark.asyncio
    async def test_filters_and_pagination(self):
        from app.mcp.workflow_history_service import WorkflowHistoryService

        repo = _FakeRepo()
        repo.detail_row = {
            "workflow_run_id": "44444444-4444-4444-4444-444444444444",
            "workflow_name": "student_review_assist",
            "user_id": "user-student-a",
            "role": "student",
            "correlation_id": "corr",
            "request_id": "req",
            "final_status": "completed",
            "blocked_reason": None,
            "partial_reason": None,
            "executed_steps": ["summarise_submission"],
            "skipped_steps": [],
            "step_count": 1,
            "started_at": "2026-04-05T12:00:00+00:00",
            "finished_at": "2026-04-05T12:00:01+00:00",
            "duration_ms": 1000.0,
            "tool_order": ["student.summariser.v1"],
            "cache_hits_count": 0,
            "llm_steps_count": 1,
            "fallback_steps_count": 0,
            "ownership_context_present": False,
            "request_fingerprint": "abc",
            "request_meta": {"text_char_count": 123},
            "warnings": [],
            "error_code": None,
            "message": None,
        }
        service = WorkflowHistoryService(repo=repo)

        result = await service.list_workflow_runs(
            limit=10,
            offset=20,
            workflow_name="student_review_assist",
            role="student",
            final_status="completed",
            user_id="user-student-a",
            partial_only=False,
            failed_only=False,
            date_from=datetime(2026, 4, 1, tzinfo=timezone.utc),
            date_to=datetime(2026, 4, 6, tzinfo=timezone.utc),
        )

        assert result.total == 1
        assert result.limit == 10
        assert result.offset == 20
        assert repo.list_kwargs["workflow_name"] == "student_review_assist"
        assert repo.list_kwargs["role"] == "student"
        assert repo.list_kwargs["final_status"] == "completed"
        assert repo.list_kwargs["user_id"] == "user-student-a"

    @pytest.mark.asyncio
    async def test_workflow_detail_retrieval_and_no_sensitive_leakage(self):
        from app.mcp.workflow_history_service import WorkflowHistoryService

        repo = _FakeRepo()
        repo.detail_row = {
            "workflow_run_id": "55555555-5555-5555-5555-555555555555",
            "workflow_name": "student_review_assist",
            "user_id": "user-student-a",
            "role": "student",
            "correlation_id": "corr",
            "request_id": "req",
            "final_status": "completed",
            "blocked_reason": None,
            "partial_reason": None,
            "executed_steps": ["summarise_submission"],
            "skipped_steps": [],
            "step_count": 1,
            "started_at": "2026-04-05T12:00:00+00:00",
            "finished_at": "2026-04-05T12:00:01+00:00",
            "duration_ms": 1000.0,
            "tool_order": ["student.summariser.v1"],
            "cache_hits_count": 0,
            "llm_steps_count": 1,
            "fallback_steps_count": 0,
            "ownership_context_present": False,
            "request_fingerprint": "abc",
            "request_meta": {"text_char_count": 123, "payload_keys": ["text"]},
            "warnings": [],
            "error_code": None,
            "message": None,
        }
        repo.detail_steps = [
            {
                "step_index": 1,
                "step_name": "summarise_submission",
                "tool_name": "student.summariser.v1",
                "tool_version": "v1",
                "step_status": "completed",
                "execution_ms": 12.5,
                "cache_hit": False,
                "llm_used": True,
                "deterministic_fallback": False,
                "error_code": None,
                "warning_count": 0,
            }
        ]
        service = WorkflowHistoryService(repo=repo)

        detail = await service.get_workflow_run_detail(
            "55555555-5555-5555-5555-555555555555"
        )

        assert detail is not None
        dumped = json.dumps(detail.model_dump())
        assert '"text":' not in dumped
        assert '"submission_text":' not in dumped

    @pytest.mark.asyncio
    async def test_summary_builds_expected_fields(self):
        from app.mcp.workflow_history_service import WorkflowHistoryService

        repo = _FakeRepo()
        repo.detail_row = {
            "workflow_run_id": "66666666-6666-6666-6666-666666666666",
            "workflow_name": "student_review_assist",
            "user_id": "user-student-a",
            "role": "student",
            "correlation_id": "corr",
            "request_id": "req",
            "final_status": "completed",
            "blocked_reason": None,
            "partial_reason": None,
            "executed_steps": ["summarise_submission"],
            "skipped_steps": [],
            "step_count": 1,
            "started_at": "2026-04-05T12:00:00+00:00",
            "finished_at": "2026-04-05T12:00:01+00:00",
            "duration_ms": 1000.0,
            "tool_order": ["student.summariser.v1"],
            "cache_hits_count": 0,
            "llm_steps_count": 1,
            "fallback_steps_count": 0,
            "ownership_context_present": False,
            "request_fingerprint": "abc",
            "request_meta": {"text_char_count": 123},
            "warnings": [],
            "error_code": None,
            "message": None,
        }
        repo.failed_steps = [
            {
                "tool_name": "student.structure_improver.v1",
                "step_name": "improve_structure",
                "error_code": "mcp.timeout",
            }
        ]
        service = WorkflowHistoryService(repo=repo)

        summary = await service.get_workflow_summary()

        assert summary.total_runs == 1
        assert summary.success_count == 1
        assert summary.average_duration_ms == 1000.0
        assert summary.most_used_workflows[0]["workflow_name"] == "student_review_assist"
        assert summary.most_common_failed_step["tool_name"] == "student.structure_improver.v1"


class TestWorkflowHistoryIntegration:
    @pytest.mark.asyncio
    async def test_history_persistence_failure_does_not_break_execution(self, monkeypatch):
        from app.mcp import config as mcp_config
        from app.mcp.orchestrator import orchestrate_workflow

        mcp_config.mcp_settings.audit_enabled = False
        mcp_config.mcp_settings.ownership_check_enabled = False
        mcp_config.mcp_settings.llm_enabled = False
        mcp_config.mcp_settings.cache_ttl_seconds = 0
        mcp_config.mcp_settings.metrics_enabled = False

        monkeypatch.setattr(
            "app.mcp.orchestrator.workflow_history_service.record_workflow_run",
            AsyncMock(side_effect=RuntimeError("history db down")),
        )

        result = await orchestrate_workflow(
            _student_request(),
            user_id="user-student-a",
            role="student",
            correlation_id="corr-history-fail",
        )

        assert result.ok is True
        assert result.final_status == "completed"

    @pytest.mark.asyncio
    async def test_admin_can_list_workflow_history(self, async_client, monkeypatch):
        from app.mcp.orchestration_schemas import WorkflowHistoryListOut, WorkflowHistoryRunOut

        monkeypatch.setattr(
            "app.api.mcp.workflow_history_service.list_workflow_runs",
            AsyncMock(
                return_value=WorkflowHistoryListOut(
                    items=[
                        WorkflowHistoryRunOut(
                            workflow_run_id="77777777-7777-7777-7777-777777777777",
                            workflow_name="student_review_assist",
                            user_id="user-student-a",
                            role="student",
                            correlation_id="corr",
                            request_id="req",
                            final_status="completed",
                            started_at="2026-04-05T12:00:00+00:00",
                            finished_at="2026-04-05T12:00:01+00:00",
                            duration_ms=1000.0,
                        )
                    ],
                    total=1,
                    limit=50,
                    offset=0,
                )
            ),
        )

        async with async_client as ac:
            resp = await ac.get(
                "/api/mcp/admin/workflows",
                headers={"Authorization": "Bearer token-admin"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["total"] == 1
        assert body["items"][0]["workflow_name"] == "student_review_assist"

    @pytest.mark.asyncio
    async def test_non_admin_denied_workflow_history(self, async_client):
        async with async_client as ac:
            resp = await ac.get(
                "/api/mcp/admin/workflows",
                headers={"Authorization": "Bearer token-student-a"},
            )

        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_admin_workflow_detail_retrieval(self, async_client, monkeypatch):
        from app.mcp.orchestration_schemas import (
            WorkflowHistoryDetailOut,
            WorkflowHistoryRunOut,
            WorkflowHistoryStepOut,
        )

        monkeypatch.setattr(
            "app.api.mcp.workflow_history_service.get_workflow_run_detail",
            AsyncMock(
                return_value=WorkflowHistoryDetailOut(
                    run=WorkflowHistoryRunOut(
                        workflow_run_id="88888888-8888-8888-8888-888888888888",
                        workflow_name="student_review_assist",
                        user_id="user-student-a",
                        role="student",
                        correlation_id="corr",
                        request_id="req",
                        final_status="completed",
                        started_at="2026-04-05T12:00:00+00:00",
                        finished_at="2026-04-05T12:00:01+00:00",
                        duration_ms=1000.0,
                    ),
                    steps=[
                        WorkflowHistoryStepOut(
                            step_index=1,
                            step_name="summarise_submission",
                            tool_name="student.summariser.v1",
                            tool_version="v1",
                            step_status="completed",
                            execution_ms=12.5,
                        )
                    ],
                )
            ),
        )

        async with async_client as ac:
            resp = await ac.get(
                "/api/mcp/admin/workflows/88888888-8888-8888-8888-888888888888",
                headers={"Authorization": "Bearer token-admin"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["run"]["workflow_run_id"] == "88888888-8888-8888-8888-888888888888"
        assert body["steps"][0]["tool_name"] == "student.summariser.v1"

    @pytest.mark.asyncio
    async def test_admin_workflow_summary(self, async_client, monkeypatch):
        from app.mcp.orchestration_schemas import WorkflowHistorySummaryOut

        monkeypatch.setattr(
            "app.api.mcp.workflow_history_service.get_workflow_summary",
            AsyncMock(
                return_value=WorkflowHistorySummaryOut(
                    total_runs=4,
                    success_count=2,
                    partial_count=1,
                    failed_count=1,
                    blocked_count=0,
                    average_duration_ms=321.5,
                    most_used_workflows=[
                        {"workflow_name": "student_review_assist", "count": 3}
                    ],
                    most_common_failed_step={
                        "tool_name": "student.structure_improver.v1",
                        "step_name": "improve_structure",
                        "count": 1,
                    },
                )
            ),
        )

        async with async_client as ac:
            resp = await ac.get(
                "/api/mcp/admin/workflows/summary",
                headers={"Authorization": "Bearer token-admin"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["total_runs"] == 4
        assert body["most_used_workflows"][0]["workflow_name"] == "student_review_assist"
