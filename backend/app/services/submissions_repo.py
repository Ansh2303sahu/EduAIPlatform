from __future__ import annotations

from typing import Any

import httpx

from app.core.config import settings


def _require_supabase_config() -> None:
    if not settings.supabase_url:
        raise RuntimeError("SUPABASE_URL is not configured")
    if not settings.supabase_service_role_key:
        raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY is not configured")


def _headers(*, prefer_return: bool = False) -> dict[str, str]:
    """
    Service-role headers for Supabase PostgREST.
    prefer_return=False => minimal response body (best for writes)
    """
    _require_supabase_config()

    headers = {
        "Authorization": f"Bearer {settings.supabase_service_role_key}",
        "apikey": settings.supabase_service_role_key,
        "Content-Type": "application/json",
    }
    headers["Prefer"] = "return=minimal" if not prefer_return else "return=representation"
    return headers


async def submissions_insert(*, row: dict[str, Any]) -> None:
    """
    Insert a row into public.submissions using service role.
    """
    _require_supabase_config()
    url = f"{settings.supabase_url.rstrip('/')}/rest/v1/submissions"

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers=_headers(prefer_return=False), json=row)

    if r.status_code >= 300:
        raise RuntimeError(f"submissions insert failed: {r.status_code} {r.text}")
