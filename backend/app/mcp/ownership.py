"""
Phase 11.2 MCP ownership enforcement.

``check_resource_ownership`` verifies that the caller may access the resource
referenced by ``file_id`` or ``submission_id`` in the execution context.  It
uses the existing Supabase service-role pattern (same headers as files_repo and
submissions_repo) and is intentionally async so it can be awaited in the policy
or executor path without blocking.

Ownership rules
---------------
- If neither ``file_id`` nor ``submission_id`` is provided the check trivially
  passes — it is not an ownership violation to call a tool without resource
  context.
- Students may only access resources they own (``user_id`` matches record).
- Professors may access any resource for grading/review purposes.
  This reflects the existing project rule: professors see student submissions.
- Admins bypass all ownership checks.
- If the resource is not found the check fails (safe default: deny unknown).
- If Supabase is unreachable the check fails-safe with an explicit denial reason
  rather than silently passing, preventing a DB-down bypass.

Returns ``OwnershipResult`` (never raises) so the executor can integrate the
result cleanly without try/except noise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import settings

logger = logging.getLogger("mcp.ownership")


@dataclass(frozen=True)
class OwnershipResult:
    allowed: bool
    denial_reason: str = ""


def _service_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.supabase_service_role_key}",
        "apikey": settings.supabase_service_role_key,
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def _supabase_configured() -> bool:
    return bool(settings.supabase_url and settings.supabase_service_role_key)


async def _fetch_row(table: str, id_value: str) -> dict[str, Any] | None:
    """Fetch a single row by ``id`` from ``table`` using service-role headers."""
    url = f"{settings.supabase_url.rstrip('/')}/rest/v1/{table}"
    params = {"id": f"eq.{id_value}", "select": "id,user_id"}
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(url, headers=_service_headers(), params=params)
        if resp.status_code >= 300:
            logger.warning(
                "mcp.ownership: Supabase %s lookup failed status=%s id=%s",
                table,
                resp.status_code,
                id_value,
            )
            return None
        rows = resp.json() or []
        return rows[0] if rows else None
    except Exception as exc:
        logger.warning(
            "mcp.ownership: Supabase %s lookup error id=%s error=%s",
            table,
            id_value,
            exc,
        )
        return None


async def check_resource_ownership(
    *,
    user_id: str,
    role: str,
    file_id: str | None,
    submission_id: str | None,
) -> OwnershipResult:
    """
    Verify that ``user_id`` with ``role`` may access the referenced resource.

    Called from the executor after policy role/namespace checks pass.
    """
    # No resource context → no ownership restriction.
    if not file_id and not submission_id:
        return OwnershipResult(allowed=True)

    # Admins bypass resource ownership checks.
    if role == "admin":
        return OwnershipResult(allowed=True)

    # Professors may read any resource for assessment — no ownership restriction.
    if role == "professor":
        return OwnershipResult(allowed=True)

    # Student: must own the resource.
    if not _supabase_configured():
        # Safe default: if we cannot check, deny with explicit reason.
        return OwnershipResult(
            allowed=False,
            denial_reason=(
                "Ownership check skipped — Supabase is not configured. "
                "Denying resource access as a safe default."
            ),
        )

    if file_id:
        row = await _fetch_row("files", file_id)
        if row is None:
            return OwnershipResult(
                allowed=False,
                denial_reason=f"Resource not found: file_id={file_id!r}",
            )
        if row.get("user_id") != user_id:
            return OwnershipResult(
                allowed=False,
                denial_reason=(
                    f"Student {user_id!r} does not own file_id={file_id!r}"
                ),
            )

    if submission_id:
        row = await _fetch_row("submissions", submission_id)
        if row is None:
            return OwnershipResult(
                allowed=False,
                denial_reason=f"Resource not found: submission_id={submission_id!r}",
            )
        if row.get("user_id") != user_id:
            return OwnershipResult(
                allowed=False,
                denial_reason=(
                    f"Student {user_id!r} does not own submission_id={submission_id!r}"
                ),
            )

    return OwnershipResult(allowed=True)
