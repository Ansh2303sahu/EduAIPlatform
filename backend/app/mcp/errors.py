"""
Phase 11 MCP exception hierarchy.

All MCP failures are represented as typed exceptions so the executor can
distinguish them cleanly without string-matching.  Each subclass carries an
``error_code`` that maps to ``MCPErrorCode`` and a ``retryable`` flag used
to populate the failure envelope.
"""

from __future__ import annotations


class MCPError(Exception):
    """Base class for all MCP execution failures."""

    error_code: str = "mcp.error"
    retryable: bool = False

    def __init__(self, message: str, *, retryable: bool | None = None) -> None:
        super().__init__(message)
        # Allow per-instance retryable override without breaking frozen class defaults.
        if retryable is not None:
            # Store as instance attribute to shadow the class default.
            object.__setattr__(self, "retryable", retryable)


class UnknownToolError(MCPError):
    """Raised when the requested tool name is not registered."""

    error_code = "mcp.unknown_tool"
    retryable = False


class DisabledToolError(MCPError):
    """Raised when the resolved tool is registered but has ``enabled=False``."""

    error_code = "mcp.disabled_tool"
    retryable = False


class PolicyDeniedError(MCPError):
    """Raised when the policy engine rejects access to a tool."""

    error_code = "mcp.policy_denied"
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        reason: str = "",
        retryable: bool = False,
    ) -> None:
        super().__init__(message, retryable=retryable)
        self.reason = reason


class ToolTimeoutError(MCPError):
    """Raised when a tool handler exceeds its configured timeout."""

    error_code = "mcp.timeout"
    retryable = True


class RateLimitError(MCPError):
    """Raised when a caller exceeds the per-tool rate limit."""

    error_code = "mcp.rate_limit"
    retryable = True


class InputValidationError(MCPError):
    """Raised when the caller-supplied payload fails input model validation."""

    error_code = "mcp.input_validation"
    retryable = False


class OutputValidationError(MCPError):
    """Raised when the tool handler returns output that fails output model validation."""

    error_code = "mcp.output_validation"
    retryable = False


class ExternalServiceError(MCPError):
    """Raised when a controlled upstream proxy call fails."""

    error_code = "mcp.external_service"
    retryable = True
