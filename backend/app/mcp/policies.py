"""
Phase 11 / 11.2 MCP policy engine.

All access-control decisions live here.  The executor calls ``enforce_policy``
before touching the handler.  Synchronous policy checks remain pure functions.
The async ownership check is called separately by the executor after sync checks
pass (it requires a network call, so it must be awaited).

Checks performed (in order)
----------------------------
1. Role allowed  — caller role must be in ``defn.allowed_roles``.
2. Namespace-role match — student namespace tools reject professor callers and
   vice versa.  Admin callers bypass namespace checks.
3. Ownership (async, Phase 11.2) — if file_id or submission_id is provided,
   verify the caller owns the resource.  Called via
   ``await enforce_ownership_policy(ctx)`` from the executor.

Design notes
------------
- All denial reasons are explicit strings so audit logs are actionable.
- ``check_policy`` returns a ``PolicyResult`` (never raises) for testability.
- ``enforce_policy`` raises ``PolicyDeniedError`` so the executor can treat it
  uniformly alongside other ``MCPError`` subclasses.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.mcp.enums import ToolNamespace, ToolRole
from app.mcp.errors import PolicyDeniedError
from app.mcp.models import ToolDefinition
from app.mcp.ownership import check_resource_ownership as _check_resource_ownership
from app.mcp.schemas import ToolExecutionContext


@dataclass(frozen=True)
class PolicyResult:
    """Structured outcome of a policy check."""

    allowed: bool
    denial_reason: str = ""


def _check_role_allowed(defn: ToolDefinition, ctx: ToolExecutionContext) -> PolicyResult:
    """Check 1: caller role must appear in the tool's allowed_roles set."""
    try:
        role_enum = ToolRole(ctx.role)
    except ValueError:
        return PolicyResult(
            allowed=False,
            denial_reason=(
                f"Unrecognised caller role {ctx.role!r}. "
                f"Allowed roles: {[r.value for r in defn.allowed_roles]}"
            ),
        )

    if role_enum not in defn.allowed_roles:
        return PolicyResult(
            allowed=False,
            denial_reason=(
                f"Role {ctx.role!r} is not permitted for tool {defn.tool_name!r}. "
                f"Allowed roles: {[r.value for r in defn.allowed_roles]}"
            ),
        )
    return PolicyResult(allowed=True)


def _check_namespace_role_match(
    defn: ToolDefinition, ctx: ToolExecutionContext
) -> PolicyResult:
    """
    Check 2: namespace–role consistency.

    - student-namespace tools may only be called by student or admin callers.
    - professor-namespace tools may only be called by professor or admin callers.
    Admin role bypasses namespace restrictions.
    """
    if ctx.role == ToolRole.ADMIN.value:
        return PolicyResult(allowed=True)

    if defn.namespace == ToolNamespace.STUDENT and ctx.role not in (
        ToolRole.STUDENT.value,
        ToolRole.ADMIN.value,
    ):
        return PolicyResult(
            allowed=False,
            denial_reason=(
                f"Namespace mismatch: tool {defn.tool_name!r} belongs to the "
                f"'student' namespace but caller role is {ctx.role!r}."
            ),
        )

    if defn.namespace == ToolNamespace.PROFESSOR and ctx.role not in (
        ToolRole.PROFESSOR.value,
        ToolRole.ADMIN.value,
    ):
        return PolicyResult(
            allowed=False,
            denial_reason=(
                f"Namespace mismatch: tool {defn.tool_name!r} belongs to the "
                f"'professor' namespace but caller role is {ctx.role!r}."
            ),
        )

    return PolicyResult(allowed=True)


def check_policy(defn: ToolDefinition, ctx: ToolExecutionContext) -> PolicyResult:
    """
    Run synchronous policy checks in order and return the first denial found.

    Returns ``PolicyResult(allowed=True)`` only if all checks pass.
    Ownership check is async and handled separately via ``enforce_ownership_policy``.
    """
    for check_fn in (
        _check_role_allowed,
        _check_namespace_role_match,
    ):
        result = check_fn(defn, ctx)
        if not result.allowed:
            return result
    return PolicyResult(allowed=True)


def enforce_policy(defn: ToolDefinition, ctx: ToolExecutionContext) -> None:
    """
    Run synchronous policy checks and raise ``PolicyDeniedError`` on failure.

    Used by the executor before the async ownership check.
    """
    result = check_policy(defn, ctx)
    if not result.allowed:
        raise PolicyDeniedError(
            f"Access denied to {defn.tool_name!r}: {result.denial_reason}",
            reason=result.denial_reason,
        )


async def check_resource_ownership(
    *,
    user_id: str,
    role: str,
    file_id: str | None,
    submission_id: str | None,
):
    """
    Thin wrapper around ``mcp.ownership.check_resource_ownership``.

    Keeping this wrapper in ``policies`` makes the ownership path easy to patch
    in focused policy/executor tests without changing runtime behavior.
    """
    return await _check_resource_ownership(
        user_id=user_id,
        role=role,
        file_id=file_id,
        submission_id=submission_id,
    )


async def enforce_ownership_policy(ctx: ToolExecutionContext) -> None:
    """
    Async ownership check — Phase 11.2 addition.

    Calls ``mcp.ownership.check_resource_ownership`` when file_id or
    submission_id is provided in the context.  Raises ``PolicyDeniedError``
    if the caller does not own the resource.

    Skipped entirely when ``mcp_settings.ownership_check_enabled`` is False
    (useful in tests / local dev without Supabase).
    """
    from app.mcp.config import mcp_settings

    if not mcp_settings.ownership_check_enabled:
        return

    # Fast path: skip DB round-trip when no resource context is supplied.
    if not ctx.file_id and not ctx.submission_id:
        return

    # Professors and admins are allowed to access resource-scoped tools for
    # review/moderation workflows without ownership checks.
    if ctx.role in {ToolRole.PROFESSOR.value, ToolRole.ADMIN.value}:
        return

    result = await check_resource_ownership(
        user_id=ctx.user_id,
        role=ctx.role,
        file_id=ctx.file_id,
        submission_id=ctx.submission_id,
    )
    if not result.allowed:
        raise PolicyDeniedError(
            f"Ownership check failed: {result.denial_reason}",
            reason=result.denial_reason,
        )
