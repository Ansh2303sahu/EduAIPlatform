"""
Phase 11.3 bounded MCP workflow orchestrator.

This module never calls tool handlers directly. Every workflow step is
delegated to ``executor.execute_tool`` so policy checks, ownership enforcement,
cache scoping, rate limits, timeouts, audit hooks, and per-tool metrics remain
centralised in one place.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from app.mcp.config import mcp_settings
from app.mcp.errors import MCPError, PolicyDeniedError
from app.mcp.executor import execute_tool
from app.mcp.metrics import record_workflow_finished, record_workflow_started
from app.mcp.orchestration_schemas import (
    ProfessorReviewAssistRequest,
    StudentReviewAssistRequest,
    WorkflowExplainabilityMeta,
    WorkflowRequest,
    WorkflowResponse,
    WorkflowSkippedStep,
    WorkflowStepResult,
)
from app.mcp.planner import WorkflowPlan, build_workflow_plan
from app.mcp.schemas import MCPExecuteRequest, MCPFailureEnvelope, MCPSuccessEnvelope
from app.mcp.workflow_history_service import workflow_history_service

logger = logging.getLogger("mcp.orchestrator")


def _success_envelope(envelope: MCPSuccessEnvelope | MCPFailureEnvelope) -> MCPSuccessEnvelope | None:
    if isinstance(envelope, MCPSuccessEnvelope):
        return envelope
    return None


def _effective_meta_for_failure(
    workflow_req: WorkflowRequest,
    *,
    stopped_reason: str,
    step_failures: int = 0,
) -> WorkflowExplainabilityMeta:
    configured_max = max(1, int(mcp_settings.orchestration_max_steps))
    requested_max = workflow_req.max_steps or configured_max
    return WorkflowExplainabilityMeta(
        max_steps_applied=min(configured_max, requested_max),
        continue_on_non_critical_failure=bool(
            workflow_req.continue_on_non_critical_failure
        ),
        partial_completion=False,
        step_failures=step_failures,
        stopped_reason=stopped_reason,
    )


def _build_student_payload(
    workflow_req: StudentReviewAssistRequest,
    tool_name: str,
) -> dict[str, Any]:
    if tool_name == "student.summariser.v1":
        return {
            "text": workflow_req.payload.text,
            "max_sentences": workflow_req.payload.max_sentences,
            "focus_mode": workflow_req.payload.focus_mode,
            "preserve_key_terms": workflow_req.payload.preserve_key_terms,
        }
    if tool_name == "student.structure_improver.v1":
        return {
            "text": workflow_req.payload.text,
            "submission_type": workflow_req.payload.submission_type,
            "expected_sections": workflow_req.payload.expected_sections,
        }
    raise ValueError(f"Unsupported student workflow tool: {tool_name!r}")


def _build_professor_feedback_items(
    workflow_req: ProfessorReviewAssistRequest,
    step_results: list[WorkflowStepResult],
) -> list[str]:
    if workflow_req.payload.feedback_items:
        return workflow_req.payload.feedback_items

    for step_result in reversed(step_results):
        if (
            step_result.tool_name == "professor.rubric_evaluator.v1"
            and isinstance(step_result.envelope, MCPSuccessEnvelope)
        ):
            evaluations = step_result.envelope.result.get("evaluations") or []
            derived = [
                str(item.get("justification") or "").strip()
                for item in evaluations
                if isinstance(item, dict) and str(item.get("justification") or "").strip()
            ]
            if derived:
                return derived[:50]
    return []


def _build_professor_payload(
    workflow_req: ProfessorReviewAssistRequest,
    tool_name: str,
    step_results: list[WorkflowStepResult],
) -> dict[str, Any]:
    if tool_name == "professor.rubric_evaluator.v1":
        return {
            "submission_text": workflow_req.payload.submission_text,
            "rubric_criteria": workflow_req.payload.rubric_criteria,
            "grading_scale": workflow_req.payload.grading_scale,
            "strictness": workflow_req.payload.strictness,
            "max_evidence_quotes": workflow_req.payload.max_evidence_quotes,
        }

    if tool_name == "professor.consistency_checker.v1":
        final_summary = workflow_req.payload.final_summary
        if not final_summary:
            for step_result in reversed(step_results):
                if (
                    step_result.tool_name == "professor.rubric_evaluator.v1"
                    and isinstance(step_result.envelope, MCPSuccessEnvelope)
                ):
                    final_summary = str(
                        step_result.envelope.result.get("overall_assessment") or ""
                    )
                    break

        return {
            "feedback_items": _build_professor_feedback_items(workflow_req, step_results),
            "scores": workflow_req.payload.scores,
            "expected_band_labels": workflow_req.payload.expected_band_labels,
            "final_summary": final_summary,
        }

    raise ValueError(f"Unsupported professor workflow tool: {tool_name!r}")


def _build_step_payload(
    workflow_req: WorkflowRequest,
    tool_name: str,
    step_results: list[WorkflowStepResult],
) -> dict[str, Any]:
    if isinstance(workflow_req, StudentReviewAssistRequest):
        return _build_student_payload(workflow_req, tool_name)
    return _build_professor_payload(workflow_req, tool_name, step_results)


def _mark_remaining_skipped(
    plan: WorkflowPlan,
    *,
    start_index: int,
    reason: str,
) -> list[WorkflowSkippedStep]:
    return [
        WorkflowSkippedStep(
            step_name=step.step_name,
            tool_name=step.tool_name,
            reason=reason,
        )
        for step in plan.steps[start_index:]
    ]


async def orchestrate_workflow(
    workflow_req: WorkflowRequest,
    *,
    user_id: str,
    role: str,
    correlation_id: str,
    file_id: str | None = None,
    submission_id: str | None = None,
) -> WorkflowResponse:
    workflow_run_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)
    t_start = time.perf_counter()
    warnings: list[str] = []
    executed_steps: list[str] = []
    skipped_steps: list[WorkflowSkippedStep] = []
    step_results: list[WorkflowStepResult] = []
    executed_tool_order: list[str] = []
    cache_hit_steps = 0
    llm_used_steps = 0
    fallback_steps = 0
    step_failures = 0
    last_failure: MCPFailureEnvelope | None = None
    final_status = ""
    stopped_reason = ""

    async def _persist_history_best_effort(response: WorkflowResponse) -> None:
        finished_at = datetime.now(timezone.utc)
        duration_ms = (time.perf_counter() - t_start) * 1000
        try:
            await workflow_history_service.record_workflow_run(
                workflow_run_id=workflow_run_id,
                workflow_req=workflow_req,
                workflow_response=response,
                user_id=user_id,
                role=role,
                correlation_id=correlation_id,
                request_id=request_id,
                file_id=file_id,
                submission_id=submission_id,
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=duration_ms,
            )
        except Exception as exc:
            logger.warning(
                "mcp.orchestrator: workflow history persistence failed workflow=%s "
                "run_id=%s user=%s error=%s",
                workflow_req.workflow_name,
                workflow_run_id,
                user_id,
                exc,
            )

    if mcp_settings.metrics_enabled:
        record_workflow_started(workflow_req.workflow_name)

    logger.info(
        "mcp.orchestrator: start workflow=%s role=%s user=%s correlation_id=%s",
        workflow_req.workflow_name,
        role,
        user_id,
        correlation_id,
    )

    try:
        plan = build_workflow_plan(
            workflow_req.workflow_name,
            role=role,
            requested_max_steps=workflow_req.max_steps,
            continue_on_non_critical_failure=workflow_req.continue_on_non_critical_failure,
        )
    except Exception as exc:
        if isinstance(exc, MCPError):
            error_code = exc.error_code
            message = str(exc)
            final_status = "blocked" if isinstance(exc, PolicyDeniedError) else "failed"
        else:
            logger.exception(
                "mcp.orchestrator: planning failure workflow=%s role=%s user=%s",
                workflow_req.workflow_name,
                role,
                user_id,
            )
            error_code = "mcp.internal_error"
            message = "Workflow planning failed. Please try again later."
            final_status = "failed"

        stopped_reason = "planning_failed"
        meta = _effective_meta_for_failure(
            workflow_req,
            stopped_reason=stopped_reason,
        )
        if mcp_settings.metrics_enabled:
            record_workflow_finished(
                workflow_req.workflow_name,
                final_status=final_status,
                executed_tool_order=executed_tool_order,
                step_failures=0,
            )
        response = WorkflowResponse(
            ok=False,
            workflow_name=workflow_req.workflow_name,
            request_id=request_id,
            correlation_id=correlation_id,
            final_status=final_status,
            executed_steps=executed_steps,
            skipped_steps=skipped_steps,
            step_results=step_results,
            warnings=warnings,
            meta=meta,
            error_code=error_code,
            message=message,
        )
        await _persist_history_best_effort(response)
        return response

    try:
        for idx, step in enumerate(plan.steps):
            if len(step_results) >= plan.effective_max_steps:
                skipped_steps.extend(
                    _mark_remaining_skipped(
                        plan,
                        start_index=idx,
                        reason="max_steps_reached",
                    )
                )
                warnings.append(
                    f"Workflow stopped after {plan.effective_max_steps} step(s) due to max-step enforcement."
                )
                stopped_reason = "max_steps_reached"
                break

            step_payload = _build_step_payload(workflow_req, step.tool_name, step_results)
            envelope = await execute_tool(
                MCPExecuteRequest(
                    tool_name=step.tool_name,
                    payload=step_payload,
                    context={
                        "user_id": user_id,
                        "role": role,
                        "correlation_id": correlation_id,
                        "file_id": file_id,
                        "submission_id": submission_id,
                    },
                )
            )
            step_results.append(
                WorkflowStepResult(
                    step_name=step.step_name,
                    tool_name=step.tool_name,
                    index=step.index,
                    critical=step.critical,
                    envelope=envelope,
                )
            )
            executed_steps.append(step.step_name)
            executed_tool_order.append(step.tool_name)

            success = _success_envelope(envelope)
            if success is not None:
                cache_hit_steps += int(success.meta.cache_hit)
                llm_used_steps += int(success.meta.llm_used)
                fallback_steps += int(
                    success.meta.llm_fallback_used or success.meta.deterministic_fallback
                )
                continue

            step_failures += 1
            last_failure = envelope

            if envelope.error_code == "mcp.policy_denied":
                warnings.append(
                    f"Workflow stopped because step {step.step_name!r} was denied by policy."
                )
                stopped_reason = f"policy_denied:{step.step_name}"
                skipped_steps.extend(
                    _mark_remaining_skipped(
                        plan,
                        start_index=idx + 1,
                        reason="stopped_after_policy_denial",
                    )
                )
                final_status = "blocked"
                break

            if step.critical:
                warnings.append(
                    f"Workflow stopped because critical step {step.step_name!r} failed."
                )
                stopped_reason = f"critical_step_failed:{step.step_name}"
                skipped_steps.extend(
                    _mark_remaining_skipped(
                        plan,
                        start_index=idx + 1,
                        reason="stopped_after_critical_failure",
                    )
                )
                final_status = "failed"
                break

            warnings.append(
                f"Non-critical step {step.step_name!r} failed with {envelope.error_code}."
            )
            if not plan.continue_on_non_critical_failure:
                stopped_reason = f"non_critical_step_failed:{step.step_name}"
                skipped_steps.extend(
                    _mark_remaining_skipped(
                        plan,
                        start_index=idx + 1,
                        reason="stopped_after_non_critical_failure",
                    )
                )
                final_status = "partial" if any(sr.envelope.ok for sr in step_results) else "failed"
                break
    except Exception as exc:
        if isinstance(exc, MCPError):
            error_code = exc.error_code
            message = str(exc)
            final_status = "blocked" if isinstance(exc, PolicyDeniedError) else "failed"
        else:
            logger.exception(
                "mcp.orchestrator: unexpected execution failure workflow=%s role=%s user=%s",
                workflow_req.workflow_name,
                role,
                user_id,
            )
            error_code = "mcp.internal_error"
            message = "Workflow execution failed. Please try again later."
            final_status = "failed"

        stopped_reason = "orchestration_exception"
        warnings.append(message)
        skipped_steps.extend(
            _mark_remaining_skipped(
                plan,
                start_index=len(step_results),
                reason="stopped_after_orchestration_error",
            )
        )
        meta = WorkflowExplainabilityMeta(
            continue_on_non_critical_failure=plan.continue_on_non_critical_failure,
            max_steps_applied=plan.effective_max_steps,
            executed_tool_order=executed_tool_order,
            cache_hit_steps=cache_hit_steps,
            llm_used_steps=llm_used_steps,
            fallback_steps=fallback_steps,
            partial_completion=False,
            step_failures=step_failures,
            stopped_reason=stopped_reason,
        )
        if mcp_settings.metrics_enabled:
            record_workflow_finished(
                workflow_req.workflow_name,
                final_status=final_status,
                executed_tool_order=executed_tool_order,
                step_failures=step_failures,
            )
        response = WorkflowResponse(
            ok=False,
            workflow_name=workflow_req.workflow_name,
            request_id=request_id,
            correlation_id=correlation_id,
            final_status=final_status,
            executed_steps=executed_steps,
            skipped_steps=skipped_steps,
            step_results=step_results,
            warnings=warnings,
            meta=meta,
            error_code=error_code,
            message=message,
        )
        await _persist_history_best_effort(response)
        return response

    if final_status not in {"blocked", "failed"}:
        any_failure = any(not step.envelope.ok for step in step_results)
        if any_failure:
            final_status = "partial" if any(step.envelope.ok for step in step_results) else "failed"
        elif skipped_steps:
            final_status = "partial"
        else:
            final_status = "completed"

    if not stopped_reason:
        if final_status == "completed":
            stopped_reason = "completed"
        elif final_status == "partial":
            stopped_reason = "partial_completion"
        else:
            stopped_reason = "workflow_failed"

    meta = WorkflowExplainabilityMeta(
        continue_on_non_critical_failure=plan.continue_on_non_critical_failure,
        max_steps_applied=plan.effective_max_steps,
        executed_tool_order=executed_tool_order,
        cache_hit_steps=cache_hit_steps,
        llm_used_steps=llm_used_steps,
        fallback_steps=fallback_steps,
        partial_completion=final_status == "partial",
        step_failures=step_failures,
        stopped_reason=stopped_reason,
    )

    if mcp_settings.metrics_enabled:
        record_workflow_finished(
            workflow_req.workflow_name,
            final_status=final_status,
            executed_tool_order=executed_tool_order,
            step_failures=step_failures,
        )

    logger.info(
        "mcp.orchestrator: finish workflow=%s status=%s executed=%s skipped=%s order=%s",
        workflow_req.workflow_name,
        final_status,
        len(step_results),
        len(skipped_steps),
        executed_tool_order,
    )

    response = WorkflowResponse(
        ok=final_status in {"completed", "partial"},
        workflow_name=workflow_req.workflow_name,
        request_id=request_id,
        correlation_id=correlation_id,
        final_status=final_status,
        executed_steps=executed_steps,
        skipped_steps=skipped_steps,
        step_results=step_results,
        warnings=warnings,
        meta=meta,
        error_code=last_failure.error_code if last_failure is not None else None,
        message=last_failure.message if last_failure is not None else None,
    )
    await _persist_history_best_effort(response)
    return response
