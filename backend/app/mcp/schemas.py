"""
Phase 11 / 11.2 MCP Pydantic schemas for request and response objects.

Design principles
-----------------
- ``extra = "forbid"`` on all request schemas prevents prompt-injected fields
  from silently passing through to handlers.
- Field lengths are bounded to prevent oversized payloads.
- Success and failure envelopes share a stable top-level shape so callers can
  always read ``ok``, ``tool_name``, ``request_id``, and ``execution_ms``
  regardless of outcome.
- Phase 11.2 adds ``ExplainabilityMeta`` to success envelopes.  It is always
  present (never None) with safe defaults so callers do not need null guards.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolExecutionContext(BaseModel):
    """
    Caller-supplied execution context.

    This is populated by the API layer from the authenticated user — callers
    cannot forge it.  It is threaded through policy checks, the handler, and
    audit hooks so every layer has a consistent view of the caller.
    """

    model_config = {"extra": "forbid"}

    user_id: str = Field(..., min_length=1, max_length=256)
    role: str = Field(..., min_length=1, max_length=64)
    correlation_id: str = Field(..., min_length=1, max_length=128)
    # Optional ownership context — enforced by mcp.ownership module.
    file_id: str | None = Field(default=None, max_length=256)
    submission_id: str | None = Field(default=None, max_length=256)


class MCPExecuteRequest(BaseModel):
    """
    Internal request object passed to ``executor.execute_tool``.

    The API layer constructs this from the HTTP request + authenticated user.
    It is never parsed directly from untrusted input.
    """

    model_config = {"extra": "forbid"}

    tool_name: str = Field(..., min_length=1, max_length=128)
    payload: dict[str, Any] = Field(default_factory=dict)
    context: ToolExecutionContext


class ExplainabilityMeta(BaseModel):
    """
    Safe metadata about how the tool produced its result.

    Never exposes internal prompts, secrets, or raw LLM output.
    Included in every MCPSuccessEnvelope so callers can surface it to users.
    """

    model_config = {"extra": "forbid"}

    # Whether the result came from the in-process cache (no LLM call).
    cache_hit: bool = False
    # Whether the LLM was invoked during this call.
    llm_used: bool = False
    # Whether the LLM fallback model was used instead of the primary.
    llm_fallback_used: bool = False
    # Which model was used (empty string when LLM was not used).
    model_used: str = ""
    # Whether the deterministic fallback path was taken (LLM unavailable/failed).
    deterministic_fallback: bool = False
    # Free-text note about confidence / limitations safe for end-user display.
    confidence_note: str = ""


class MCPSuccessEnvelope(BaseModel):
    """Standard success response envelope returned for all completed tool calls."""

    ok: Literal[True] = True
    tool_name: str
    tool_version: str
    request_id: str
    correlation_id: str
    result: Any
    warnings: list[str] = Field(default_factory=list)
    execution_ms: float
    # Phase 11.2: always present explainability metadata.
    meta: ExplainabilityMeta = Field(default_factory=ExplainabilityMeta)


class MCPFailureEnvelope(BaseModel):
    """Standard failure response envelope returned for all rejected or failed tool calls."""

    ok: Literal[False] = False
    tool_name: str
    # tool_version is None when the tool could not be resolved at all.
    tool_version: str | None
    request_id: str
    correlation_id: str
    error_code: str
    message: str
    retryable: bool
    execution_ms: float
