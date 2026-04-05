"""
Phase 11 MCP domain models.

``ToolDefinition`` is a frozen dataclass (not a Pydantic model) because it
carries a live callable (``handler``) that cannot be validated by Pydantic.
Using ``frozen=True`` prevents accidental mutation after registration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Type

from pydantic import BaseModel

from app.mcp.enums import RiskLevel, ToolNamespace, ToolRole


@dataclass(frozen=True)
class ToolDefinition:
    """
    Registry entry for a single versioned MCP tool.

    Fields
    ------
    tool_name
        Canonical dot-separated name, e.g. ``"student.summariser.v1"``.
        Must be unique across the registry.
    namespace
        Audience namespace — ``STUDENT`` or ``PROFESSOR``.
    version
        Semantic version string, e.g. ``"v1"``.
    description
        Human-readable description shown in tool listings.
    allowed_roles
        Frozen set of roles permitted to call this tool.
        The policy engine checks caller role against this set.
    risk_level
        Inherent risk classification used for audit and future routing.
    enabled
        If ``False``, all calls are rejected with ``DisabledToolError``
        before any policy or handler logic runs.
    timeout_seconds
        Maximum wall-clock seconds the handler may run.
        Enforced by ``timeouts.run_with_timeout``.
    supports_idempotency
        Advisory flag: if ``True``, safe callers may retry on transient errors.
    safe_for_multi_step
        Advisory flag: if ``True``, this tool may participate in
        bounded multi-step orchestration sequences.
    input_model
        Pydantic ``BaseModel`` subclass.  The executor validates the raw
        payload against this before calling the handler.
    output_model
        Pydantic ``BaseModel`` subclass.  The executor validates the handler
        return value against this before building the success envelope.
    handler
        Async callable ``(validated_input, ctx) -> output_model_instance``.
        The LLM never calls this directly — only the executor may invoke it.
    """

    tool_name: str
    namespace: ToolNamespace
    version: str
    description: str
    allowed_roles: frozenset[ToolRole]
    risk_level: RiskLevel
    enabled: bool
    timeout_seconds: float
    supports_idempotency: bool
    safe_for_multi_step: bool
    input_model: Type[BaseModel]
    output_model: Type[BaseModel]
    # Callable[InputModel, ToolExecutionContext] -> Awaitable[OutputModel]
    handler: Callable[..., Awaitable[Any]]
