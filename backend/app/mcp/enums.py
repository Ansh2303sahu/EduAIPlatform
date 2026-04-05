"""
Phase 11 MCP enumerations.

All enums are ``str`` subclasses so they serialise transparently to/from JSON
and compare directly against plain strings — matching the pattern established
in ``app.langchain.enums``.
"""

from __future__ import annotations

from enum import Enum


class ToolNamespace(str, Enum):
    """Which audience a tool belongs to."""

    STUDENT = "student"
    PROFESSOR = "professor"
    ADMIN = "admin"


class RiskLevel(str, Enum):
    """
    Inherent risk level of a tool.

    LOW    — read-only or purely transformative; no persistent side-effects.
    MEDIUM — may read sensitive data or produce output stored downstream.
    HIGH   — writes to persistent stores, external services, or is destructive.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ToolRole(str, Enum):
    """Roles that may be granted access to a tool."""

    STUDENT = "student"
    PROFESSOR = "professor"
    ADMIN = "admin"


class MCPErrorCode(str, Enum):
    """Canonical error codes returned in failure envelopes."""

    UNKNOWN_TOOL = "mcp.unknown_tool"
    DISABLED_TOOL = "mcp.disabled_tool"
    POLICY_DENIED = "mcp.policy_denied"
    TIMEOUT = "mcp.timeout"
    RATE_LIMIT = "mcp.rate_limit"
    INPUT_VALIDATION = "mcp.input_validation"
    OUTPUT_VALIDATION = "mcp.output_validation"
    EXTERNAL_SERVICE = "mcp.external_service"
    INTERNAL_ERROR = "mcp.internal_error"
