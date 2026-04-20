from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from app.core.config import settings
from app.services.uuid_normalization import normalize_uuid_insert_payload


def _require_supabase() -> None:
    if not settings.supabase_url:
        raise RuntimeError("SUPABASE_URL is not configured")
    if not settings.supabase_service_role_key:
        raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY is not configured")


def service_headers(*, prefer_return: bool) -> Dict[str, str]:
    _require_supabase()
    return {
        "apikey": settings.supabase_service_role_key,
        "Authorization": f"Bearer {settings.supabase_service_role_key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation" if prefer_return else "return=minimal",
    }


async def insert_row(*, table: str, row: Dict[str, Any]) -> Dict[str, Any]:
    _require_supabase()
    safe_row = normalize_uuid_insert_payload(row)
    url = f"{settings.supabase_url.rstrip('/')}/rest/v1/{table}"
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(url, headers=service_headers(prefer_return=True), json=safe_row)
    if resp.status_code >= 300:
        raise RuntimeError(f"insert {table} failed: {resp.status_code} {resp.text}")
    data = resp.json() or []
    if not data:
        # Prefer:return=representation should return row, but fail safe.
        return {}
    return data[0]


async def insert_rows(*, table: str, rows: list[Dict[str, Any]]) -> None:
    if not rows:
        return
    _require_supabase()
    safe_rows = [normalize_uuid_insert_payload(row) for row in rows]
    url = f"{settings.supabase_url.rstrip('/')}/rest/v1/{table}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, headers=service_headers(prefer_return=False), json=safe_rows)
    if resp.status_code >= 300:
        raise RuntimeError(f"bulk insert {table} failed: {resp.status_code} {resp.text}")
