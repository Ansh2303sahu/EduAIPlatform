"""Adapter for the existing Phase 11 MCP executor and orchestrator.

Phase 12 must never call MCP handlers directly. This wrapper ensures any future
graph tool use goes through the same policy, ownership, audit, cache, and
history protections already implemented in Phase 11.
"""

from __future__ import annotations

from typing import Any

from app.mcp.executor import execute_tool
from app.mcp.orchestrator import orchestrate_workflow
from app.mcp.orchestration_schemas import WorkflowRequest
from app.mcp.schemas import MCPExecuteRequest, ToolExecutionContext

from ..state import Phase12GraphState


def build_tool_context(
    state: Phase12GraphState,
    *,
    file_id: str | None = None,
    submission_id: str | None = None,
) -> ToolExecutionContext:
    """Build the standard MCP tool execution context from graph state."""

    return ToolExecutionContext(
        user_id=state.user_id,
        role=state.role,
        file_id=file_id or state.pipeline_context.file_id,
        submission_id=submission_id or state.pipeline_context.submission_id or None,
        correlation_id=state.correlation_id,
        request_id=state.pipeline_context.request_id,
    )


async def execute_tool_for_state(
    state: Phase12GraphState,
    *,
    tool_name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Execute a single MCP tool through the existing executor."""

    context = build_tool_context(state)
    result = await execute_tool(
        MCPExecuteRequest(
            tool_name=tool_name,
            payload=payload,
            context=context,
        )
    )
    return result.model_dump(mode="json")


async def orchestrate_for_state(
    state: Phase12GraphState,
    workflow_request: WorkflowRequest,
) -> dict[str, Any]:
    """Execute an approved MCP workflow through the existing orchestrator."""

    result = await orchestrate_workflow(
        workflow_request,
        user_id=state.user_id,
        role=state.role,
        correlation_id=state.correlation_id,
        file_id=state.pipeline_context.file_id,
        submission_id=state.pipeline_context.submission_id or None,
    )
    return result.model_dump(mode="json")
