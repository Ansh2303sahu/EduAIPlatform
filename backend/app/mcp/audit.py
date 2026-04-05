"""
Phase 11 MCP audit adapter.

Wraps ``app.services.audit.audit_log`` with MCP-specific event names and
payload shapes.  All calls are best-effort — failures are logged but never
propagate to the caller, matching the existing audit service contract.

Events emitted
--------------
``mcp.tool.started``   — before the handler runs.
``mcp.tool.succeeded`` — after successful handler + output validation.
``mcp.tool.failed``    — on any error (MCPError or unexpected exception).
"""

from __future__ import annotations

import logging

from app.mcp.schemas import ToolExecutionContext
from app.services.audit import audit_log

logger = logging.getLogger("mcp.audit")


async def emit_tool_started(
    *,
    tool_name: str,
    tool_version: str,
    ctx: ToolExecutionContext,
    request_id: str,
) -> None:
    """Emit an audit event immediately before a tool handler is invoked."""
    await audit_log(
        actor_user_id=ctx.user_id,
        action="mcp.tool.started",
        entity_type="mcp_tool",
        entity_id=tool_name,
        metadata={
            "tool_name": tool_name,
            "tool_version": tool_version,
            "request_id": request_id,
            "correlation_id": ctx.correlation_id,
            "role": ctx.role,
            "file_id": ctx.file_id,
            "submission_id": ctx.submission_id,
        },
    )


async def emit_tool_succeeded(
    *,
    tool_name: str,
    tool_version: str,
    ctx: ToolExecutionContext,
    request_id: str,
    execution_ms: float,
) -> None:
    """Emit an audit event after successful tool execution and output validation."""
    await audit_log(
        actor_user_id=ctx.user_id,
        action="mcp.tool.succeeded",
        entity_type="mcp_tool",
        entity_id=tool_name,
        metadata={
            "tool_name": tool_name,
            "tool_version": tool_version,
            "request_id": request_id,
            "correlation_id": ctx.correlation_id,
            "role": ctx.role,
            "execution_ms": round(execution_ms, 2),
        },
    )


async def emit_tool_failed(
    *,
    tool_name: str,
    tool_version: str | None,
    ctx: ToolExecutionContext,
    request_id: str,
    error_code: str,
    execution_ms: float,
) -> None:
    """
    Emit an audit event when a tool call fails for any reason.

    ``tool_version`` is ``None`` when the tool could not be resolved (i.e.
    ``UnknownToolError`` was raised before the ``ToolDefinition`` was fetched).
    """
    await audit_log(
        actor_user_id=ctx.user_id,
        action="mcp.tool.failed",
        entity_type="mcp_tool",
        entity_id=tool_name,
        metadata={
            "tool_name": tool_name,
            "tool_version": tool_version,
            "request_id": request_id,
            "correlation_id": ctx.correlation_id,
            "role": ctx.role,
            "error_code": error_code,
            "execution_ms": round(execution_ms, 2),
        },
    )
