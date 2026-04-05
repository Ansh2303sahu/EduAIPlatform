"""
Phase 11 MCP API router.

Exposes a single endpoint:

    POST /api/mcp/execute

The endpoint is the *only* HTTP surface for MCP tool execution.  All
auth/policy/rate-limiting/audit logic runs inside ``executor.execute_tool``
— the router's job is purely to:

  1. Verify the caller is authenticated (``get_current_user`` dependency).
  2. Construct the ``MCPExecuteRequest`` from the HTTP body + auth context.
  3. Delegate to the executor.
  4. Return the envelope with HTTP 200 (envelope carries ``ok`` bool).

Security notes
--------------
- Auth failures (missing/invalid JWT) produce HTTP 401 via the dependency.
- Tool-level failures (policy denied, unknown tool, etc.) produce HTTP 200
  with ``ok=False`` — they are not auth failures and must not leak info via
  status codes.
- The ``MCP_ENABLED`` flag is checked at request time so it can be toggled at
  runtime without a server restart.
"""

from __future__ import annotations

import uuid
from typing import Any

from datetime import datetime

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field, ValidationError

from app.core.deps import CurrentUser, get_current_user, require_admin_mfa
from app.core.rate_limit import limiter
from app.mcp.config import mcp_settings
from app.mcp.executor import execute_tool
from app.mcp.orchestration_schemas import (
    WORKFLOW_REQUEST_ADAPTER,
    WorkflowHistoryDetailOut,
    WorkflowHistoryListOut,
    WorkflowHistorySummaryOut,
    WorkflowInfo,
    WorkflowInfoStep,
)
from app.mcp.orchestrator import orchestrate_workflow
from app.mcp.planner import list_visible_workflows
from app.mcp.schemas import MCPExecuteRequest, ToolExecutionContext
from app.mcp.workflow_history_service import workflow_history_service

router = APIRouter(prefix="/mcp", tags=["mcp"])


class MCPExecuteIn(BaseModel):
    """
    Public request body for ``POST /api/mcp/execute``.

    ``correlation_id`` is optional — the API generates one if omitted.
    ``file_id`` and ``submission_id`` are optional ownership context fields
    passed through to the tool handler and audit log.
    """

    model_config = {"extra": "forbid"}

    tool_name: str = Field(..., min_length=1, max_length=128)
    payload: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str | None = Field(default=None, max_length=128)
    file_id: str | None = Field(default=None, max_length=256)
    submission_id: str | None = Field(default=None, max_length=256)


def _require_admin_tool_access(tool_name: str, user: CurrentUser) -> None:
    if not tool_name.startswith("admin."):
        return
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only.")
    if (user.raw_claims or {}).get("aal") != "aal2":
        raise HTTPException(status_code=403, detail="MFA required (aal2)")


@router.post("/execute")
@limiter.limit("30/minute")
async def mcp_execute(
    request: Request,
    body: dict[str, Any] = Body(...),
    user: CurrentUser = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Execute an MCP tool on behalf of the authenticated caller.

    Returns the standard success or failure envelope.  HTTP 200 in both cases
    — callers must inspect ``ok`` to determine the outcome.
    """
    if not mcp_settings.enabled:
        raise HTTPException(status_code=503, detail="MCP tools are currently disabled.")

    try:
        parsed_body = MCPExecuteIn.model_validate(body)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    _require_admin_tool_access(parsed_body.tool_name, user)

    correlation_id = parsed_body.correlation_id or str(uuid.uuid4())

    ctx = ToolExecutionContext(
        user_id=user.id,
        role=user.role,
        correlation_id=correlation_id,
        file_id=parsed_body.file_id,
        submission_id=parsed_body.submission_id,
    )

    req = MCPExecuteRequest(
        tool_name=parsed_body.tool_name,
        payload=parsed_body.payload,
        context=ctx,
    )

    envelope = await execute_tool(req)
    return envelope.model_dump()


@router.post("/orchestrate")
@limiter.limit("20/minute")
async def mcp_orchestrate(
    request: Request,
    body: dict[str, Any] = Body(...),
    user: CurrentUser = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Execute an approved bounded MCP workflow for the authenticated caller.

    Returns a stable workflow response envelope. HTTP 200 is used for both
    completed and blocked workflows; callers must inspect ``ok`` and
    ``final_status``.
    """
    if not mcp_settings.enabled or not mcp_settings.orchestration_enabled:
        raise HTTPException(
            status_code=503,
            detail="MCP orchestration is currently disabled.",
        )

    try:
        parsed_body = WORKFLOW_REQUEST_ADAPTER.validate_python(body)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    correlation_id = parsed_body.correlation_id or str(uuid.uuid4())

    envelope = await orchestrate_workflow(
        parsed_body,
        user_id=user.id,
        role=user.role,
        correlation_id=correlation_id,
        file_id=parsed_body.file_id,
        submission_id=parsed_body.submission_id,
    )
    return envelope.model_dump()


@router.get("/tools")
async def mcp_list_tools(
    user: CurrentUser = Depends(get_current_user),
) -> dict[str, Any]:
    """
    List all enabled tools visible to the caller's role.

    Returns a summary (no handler references) filtered by namespace-role match.
    """
    if not mcp_settings.enabled:
        raise HTTPException(status_code=503, detail="MCP tools are currently disabled.")

    from app.mcp.registry import list_tools

    all_tools = list_tools(include_disabled=False)

    # Filter to tools the caller's role can access (namespace match).
    visible = []
    for defn in all_tools:
        try:
            from app.mcp.enums import ToolRole
            role_enum = ToolRole(user.role)
        except ValueError:
            break
        if role_enum in defn.allowed_roles:
            visible.append(
                {
                    "tool_name": defn.tool_name,
                    "namespace": defn.namespace.value,
                    "version": defn.version,
                    "description": defn.description,
                    "risk_level": defn.risk_level.value,
                    "supports_idempotency": defn.supports_idempotency,
                    "safe_for_multi_step": defn.safe_for_multi_step,
                    "timeout_seconds": defn.timeout_seconds,
                }
            )

    return {"ok": True, "tools": visible, "count": len(visible)}


@router.get("/workflows")
async def mcp_list_workflows(
    user: CurrentUser = Depends(get_current_user),
) -> dict[str, Any]:
    """
    List approved bounded MCP workflows visible to the caller's role.
    """
    if not mcp_settings.enabled or not mcp_settings.orchestration_enabled:
        raise HTTPException(
            status_code=503,
            detail="MCP orchestration is currently disabled.",
        )

    workflows = [
        WorkflowInfo(
            workflow_name=rule.workflow_name,
            description=rule.description,
            allowed_roles=sorted(role.value for role in rule.allowed_roles),
            max_steps=min(
                mcp_settings.orchestration_max_steps,
                len(rule.steps),
            ),
            continue_on_non_critical_failure=rule.continue_on_non_critical_failure,
            steps=[
                WorkflowInfoStep(
                    step_name=step.step_name,
                    tool_name=step.tool_name,
                    critical=step.critical,
                )
                for step in rule.steps
            ],
        )
        for rule in list_visible_workflows(user.role)
    ]

    return {
        "ok": True,
        "workflows": [workflow.model_dump() for workflow in workflows],
        "count": len(workflows),
    }


@router.get("/admin/workflows")
async def mcp_admin_workflows(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    workflow_name: str | None = Query(default=None, max_length=128),
    role: str | None = Query(default=None, max_length=64),
    final_status: str | None = Query(default=None, max_length=32),
    user_id: str | None = Query(default=None, max_length=256),
    partial_only: bool = Query(default=False),
    failed_only: bool = Query(default=False),
    date_from: datetime | None = Query(default=None),
    date_to: datetime | None = Query(default=None),
    admin: CurrentUser = Depends(require_admin_mfa()),
) -> dict[str, Any]:
    """
    Admin-only paginated workflow history listing.
    """
    if not mcp_settings.enabled or not mcp_settings.orchestration_enabled:
        raise HTTPException(
            status_code=503,
            detail="MCP orchestration is currently disabled.",
        )

    result: WorkflowHistoryListOut = await workflow_history_service.list_workflow_runs(
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
    return {"ok": True, **result.model_dump()}


@router.get("/admin/workflows/summary")
async def mcp_admin_workflows_summary(
    admin: CurrentUser = Depends(require_admin_mfa()),
) -> dict[str, Any]:
    """
    Admin-only workflow history summary.
    """
    if not mcp_settings.enabled or not mcp_settings.orchestration_enabled:
        raise HTTPException(
            status_code=503,
            detail="MCP orchestration is currently disabled.",
        )

    result: WorkflowHistorySummaryOut = await workflow_history_service.get_workflow_summary()
    return {"ok": True, **result.model_dump()}


@router.get("/admin/workflows/{workflow_run_id}")
async def mcp_admin_workflow_detail(
    workflow_run_id: str,
    admin: CurrentUser = Depends(require_admin_mfa()),
) -> dict[str, Any]:
    """
    Admin-only workflow history detail.
    """
    if not mcp_settings.enabled or not mcp_settings.orchestration_enabled:
        raise HTTPException(
            status_code=503,
            detail="MCP orchestration is currently disabled.",
        )

    result: WorkflowHistoryDetailOut | None = await workflow_history_service.get_workflow_run_detail(
        workflow_run_id
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Workflow history run not found.")
    return {"ok": True, **result.model_dump()}


@router.get("/metrics")
async def mcp_metrics(
    user: CurrentUser = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Return in-process MCP tool execution metrics.

    Restricted to admin callers — students and professors receive 403.
    Returns a JSON-serialisable snapshot of success/failure/latency counters.
    """
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only.")

    from app.mcp.metrics import snapshot
    return {"ok": True, "metrics": snapshot()}


@router.delete("/cache/{tool_name}")
async def mcp_invalidate_cache(
    tool_name: str,
    user: CurrentUser = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Invalidate all cached results for ``tool_name``.

    Restricted to admin callers.  Returns the number of entries evicted.
    """
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only.")

    from app.mcp.cache import invalidate
    evicted = invalidate(tool_name)
    return {"ok": True, "tool_name": tool_name, "evicted": evicted}
