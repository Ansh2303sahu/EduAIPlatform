"""
Phase 11.4 MCP workflow history service.

This service is responsible for:
- building redacted persistence rows from workflow execution metadata
- writing workflow/step history through the repository
- serving admin-only history list/detail/summary views

Raw submission text and full tool payloads are intentionally excluded.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime
from typing import Any

from app.mcp.orchestration_schemas import (
    ProfessorReviewAssistRequest,
    StudentReviewAssistRequest,
    WorkflowHistoryDetailOut,
    WorkflowHistoryListOut,
    WorkflowHistoryRunOut,
    WorkflowHistoryStepOut,
    WorkflowHistorySummaryOut,
    WorkflowRequest,
    WorkflowResponse,
)
from app.mcp.workflow_history_repo import WorkflowHistoryRepo


def _stable_hash(value: object) -> str:
    dumped = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def _safe_request_meta(workflow_req: WorkflowRequest) -> dict[str, object]:
    base: dict[str, object] = {
        "workflow_name": workflow_req.workflow_name,
        "payload_keys": sorted(workflow_req.payload.model_dump().keys()),
        "max_steps_requested": workflow_req.max_steps,
        "continue_on_non_critical_failure": workflow_req.continue_on_non_critical_failure,
    }

    if isinstance(workflow_req, StudentReviewAssistRequest):
        base.update(
            {
                "text_char_count": len(workflow_req.payload.text),
                "submission_type": workflow_req.payload.submission_type,
                "expected_sections_count": len(workflow_req.payload.expected_sections),
                "focus_mode": workflow_req.payload.focus_mode,
            }
        )
    elif isinstance(workflow_req, ProfessorReviewAssistRequest):
        base.update(
            {
                "submission_text_char_count": len(workflow_req.payload.submission_text),
                "rubric_criteria_count": len(workflow_req.payload.rubric_criteria),
                "scores_count": len(workflow_req.payload.scores),
                "feedback_items_count": len(workflow_req.payload.feedback_items),
                "expected_band_labels_count": len(workflow_req.payload.expected_band_labels),
                "grading_scale": workflow_req.payload.grading_scale,
                "strictness": workflow_req.payload.strictness,
            }
        )

    return base


def _as_run_out(row: dict[str, Any]) -> WorkflowHistoryRunOut:
    return WorkflowHistoryRunOut(
        workflow_run_id=str(row.get("workflow_run_id") or ""),
        workflow_name=str(row.get("workflow_name") or ""),
        user_id=str(row.get("user_id") or ""),
        role=str(row.get("role") or ""),
        correlation_id=str(row.get("correlation_id") or ""),
        request_id=str(row.get("request_id") or ""),
        final_status=str(row.get("final_status") or ""),
        blocked_reason=row.get("blocked_reason"),
        partial_reason=row.get("partial_reason"),
        executed_steps=list(row.get("executed_steps") or []),
        skipped_steps=list(row.get("skipped_steps") or []),
        step_count=int(row.get("step_count") or 0),
        started_at=str(row.get("started_at") or ""),
        finished_at=str(row.get("finished_at") or ""),
        duration_ms=float(row.get("duration_ms") or 0.0),
        tool_order=list(row.get("tool_order") or []),
        cache_hits_count=int(row.get("cache_hits_count") or 0),
        llm_steps_count=int(row.get("llm_steps_count") or 0),
        fallback_steps_count=int(row.get("fallback_steps_count") or 0),
        ownership_context_present=bool(row.get("ownership_context_present", False)),
        request_fingerprint=str(row.get("request_fingerprint") or ""),
        request_meta=dict(row.get("request_meta") or {}),
        warnings=list(row.get("warnings") or []),
        error_code=row.get("error_code"),
        message=row.get("message"),
    )


def _as_step_out(row: dict[str, Any]) -> WorkflowHistoryStepOut:
    return WorkflowHistoryStepOut(
        step_index=int(row.get("step_index") or 0),
        step_name=str(row.get("step_name") or ""),
        tool_name=str(row.get("tool_name") or ""),
        tool_version=row.get("tool_version"),
        step_status=str(row.get("step_status") or "failed"),
        execution_ms=float(row.get("execution_ms") or 0.0),
        cache_hit=bool(row.get("cache_hit", False)),
        llm_used=bool(row.get("llm_used", False)),
        deterministic_fallback=bool(row.get("deterministic_fallback", False)),
        error_code=row.get("error_code"),
        warning_count=int(row.get("warning_count") or 0),
    )


class WorkflowHistoryService:
    def __init__(self, repo: WorkflowHistoryRepo | None = None) -> None:
        self.repo = repo or WorkflowHistoryRepo()

    async def record_workflow_run(
        self,
        *,
        workflow_run_id: str,
        workflow_req: WorkflowRequest,
        workflow_response: WorkflowResponse,
        user_id: str,
        role: str,
        correlation_id: str,
        request_id: str,
        file_id: str | None,
        submission_id: str | None,
        started_at: datetime,
        finished_at: datetime,
        duration_ms: float,
    ) -> None:
        request_meta = _safe_request_meta(workflow_req)
        run_row = {
            "workflow_run_id": workflow_run_id,
            "workflow_name": workflow_req.workflow_name,
            "user_id": user_id,
            "role": role,
            "correlation_id": correlation_id,
            "request_id": request_id,
            "final_status": workflow_response.final_status,
            "blocked_reason": (
                workflow_response.meta.stopped_reason
                if workflow_response.final_status == "blocked"
                else None
            ),
            "partial_reason": (
                workflow_response.meta.stopped_reason
                if workflow_response.final_status == "partial"
                else None
            ),
            "executed_steps": workflow_response.executed_steps,
            "skipped_steps": [step.model_dump() for step in workflow_response.skipped_steps],
            "step_count": len(workflow_response.step_results),
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_ms": round(duration_ms, 2),
            "tool_order": workflow_response.meta.executed_tool_order,
            "cache_hits_count": workflow_response.meta.cache_hit_steps,
            "llm_steps_count": workflow_response.meta.llm_used_steps,
            "fallback_steps_count": workflow_response.meta.fallback_steps,
            "ownership_context_present": bool(file_id or submission_id),
            "request_fingerprint": _stable_hash(
                {
                    "workflow_name": workflow_req.workflow_name,
                    "payload": workflow_req.payload.model_dump(),
                }
            ),
            "request_meta": request_meta,
            "warnings": workflow_response.warnings,
            "error_code": workflow_response.error_code,
            "message": workflow_response.message,
        }
        await self.repo.insert_run(run_row)

        step_rows = []
        for step in workflow_response.step_results:
            if step.envelope.ok:
                step_rows.append(
                    {
                        "workflow_run_id": workflow_run_id,
                        "step_index": step.index,
                        "step_name": step.step_name,
                        "tool_name": step.tool_name,
                        "tool_version": step.envelope.tool_version,
                        "step_status": "completed",
                        "execution_ms": step.envelope.execution_ms,
                        "cache_hit": step.envelope.meta.cache_hit,
                        "llm_used": step.envelope.meta.llm_used,
                        "deterministic_fallback": step.envelope.meta.deterministic_fallback,
                        "error_code": None,
                        "warning_count": len(step.envelope.warnings),
                    }
                )
            else:
                step_rows.append(
                    {
                        "workflow_run_id": workflow_run_id,
                        "step_index": step.index,
                        "step_name": step.step_name,
                        "tool_name": step.tool_name,
                        "tool_version": step.envelope.tool_version,
                        "step_status": "failed",
                        "execution_ms": step.envelope.execution_ms,
                        "cache_hit": False,
                        "llm_used": False,
                        "deterministic_fallback": False,
                        "error_code": step.envelope.error_code,
                        "warning_count": 0,
                    }
                )
        await self.repo.insert_steps(step_rows)

    async def list_workflow_runs(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        workflow_name: str | None = None,
        role: str | None = None,
        final_status: str | None = None,
        user_id: str | None = None,
        partial_only: bool = False,
        failed_only: bool = False,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> WorkflowHistoryListOut:
        rows, total = await self.repo.list_runs(
            limit=limit,
            offset=offset,
            workflow_name=workflow_name,
            role=role,
            final_status=final_status,
            user_id=user_id,
            partial_only=partial_only,
            failed_only=failed_only,
            date_from=date_from,
            date_to=date_to,
        )
        return WorkflowHistoryListOut(
            items=[_as_run_out(row) for row in rows],
            total=total,
            limit=limit,
            offset=offset,
        )

    async def get_workflow_run_detail(
        self,
        workflow_run_id: str,
    ) -> WorkflowHistoryDetailOut | None:
        run = await self.repo.get_run(workflow_run_id)
        if run is None:
            return None
        steps = await self.repo.list_steps(workflow_run_id)
        return WorkflowHistoryDetailOut(
            run=_as_run_out(run),
            steps=[_as_step_out(step) for step in steps],
        )

    async def get_workflow_summary(self) -> WorkflowHistorySummaryOut:
        runs = await self.list_workflow_runs(limit=500, offset=0)
        failed_steps = await self.repo.list_failed_steps(limit=500)

        counts = Counter(item.final_status for item in runs.items)
        workflow_counts = Counter(item.workflow_name for item in runs.items)
        failed_step_counts = Counter(
            f"{step.get('tool_name') or ''}:{step.get('step_name') or ''}"
            for step in failed_steps
            if step.get("tool_name") or step.get("step_name")
        )
        avg_duration = round(
            sum(item.duration_ms for item in runs.items) / len(runs.items),
            2,
        ) if runs.items else 0.0

        most_used = [
            {"workflow_name": name, "count": count}
            for name, count in workflow_counts.most_common(5)
        ]
        most_common_failed_step: dict[str, object] = {}
        if failed_step_counts:
            top_key, top_count = failed_step_counts.most_common(1)[0]
            tool_name, _, step_name = top_key.partition(":")
            most_common_failed_step = {
                "tool_name": tool_name,
                "step_name": step_name,
                "count": top_count,
            }

        return WorkflowHistorySummaryOut(
            total_runs=runs.total,
            success_count=counts.get("completed", 0),
            partial_count=counts.get("partial", 0),
            failed_count=counts.get("failed", 0),
            blocked_count=counts.get("blocked", 0),
            average_duration_ms=avg_duration,
            most_used_workflows=most_used,
            most_common_failed_step=most_common_failed_step,
        )


workflow_history_service = WorkflowHistoryService()
