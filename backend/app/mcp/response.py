"""
Phase 11 / 11.2 MCP response envelope builders.

``build_success`` now accepts an optional ``ExplainabilityMeta`` so handlers
can surface LLM-vs-fallback context without coupling to the envelope shape.
"""

from __future__ import annotations

from app.mcp.errors import MCPError
from app.mcp.models import ToolDefinition
from app.mcp.schemas import ExplainabilityMeta, MCPFailureEnvelope, MCPSuccessEnvelope


def build_success(
    *,
    defn: ToolDefinition,
    result: object,
    request_id: str,
    correlation_id: str,
    execution_ms: float,
    warnings: list[str] | None = None,
    meta: ExplainabilityMeta | None = None,
) -> MCPSuccessEnvelope:
    """Build a success envelope from a completed tool execution."""
    return MCPSuccessEnvelope(
        tool_name=defn.tool_name,
        tool_version=defn.version,
        request_id=request_id,
        correlation_id=correlation_id,
        result=result,
        warnings=warnings or [],
        execution_ms=round(execution_ms, 2),
        meta=meta or ExplainabilityMeta(),
    )


def build_failure(
    *,
    tool_name: str,
    tool_version: str | None,
    request_id: str,
    correlation_id: str,
    error: MCPError | Exception,
    execution_ms: float,
) -> MCPFailureEnvelope:
    """
    Build a failure envelope from any exception.

    ``MCPError`` subclasses carry typed ``error_code`` and ``retryable`` fields.
    Unexpected exceptions are mapped to ``mcp.internal_error`` with a safe,
    non-leaking message — the actual exception is logged by the executor.
    """
    if isinstance(error, MCPError):
        error_code = error.error_code
        retryable = error.retryable
        message = str(error)
    else:
        error_code = "mcp.internal_error"
        retryable = False
        message = "An internal error occurred. Please try again later."

    return MCPFailureEnvelope(
        tool_name=tool_name,
        tool_version=tool_version,
        request_id=request_id,
        correlation_id=correlation_id,
        error_code=error_code,
        message=message,
        retryable=retryable,
        execution_ms=round(execution_ms, 2),
    )
