from __future__ import annotations

import os
from typing import Any, Optional

import httpx


def _supabase_url() -> Optional[str]:
    return os.getenv("SUPABASE_URL", "").strip() or None


def _supabase_key() -> Optional[str]:
    return os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip() or None


def _headers() -> dict[str, str]:
    key = _supabase_key() or ""
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }


async def audit_log(
    *,
    actor_user_id: Optional[str],
    action: str,
    metadata: Optional[dict[str, Any]] = None,
    ip: Optional[str] = None,
    user_agent: Optional[str] = None,
    entity_type: Optional[str] = None,
    entity_id: Optional[str] = None,
) -> None:
    base = _supabase_url()
    key = _supabase_key()
    if not base or not key:
        return

    payload: dict[str, Any] = {
        "actor_user_id": actor_user_id,
        "action": action,
        "metadata": metadata or {},
        "entity_type": entity_type,
        "entity_id": entity_id,
    }
    if ip:
        payload["ip"] = ip
    if user_agent:
        payload["user_agent"] = user_agent

    url = f"{base.rstrip('/')}/rest/v1/audit_logs"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=10.0)) as client:
            await client.post(url, headers=_headers(), json=payload)
    except Exception:
        return