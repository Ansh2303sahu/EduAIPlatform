from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from app.core.config import settings


def _require_supabase_config() -> None:
    if not settings.supabase_url:
        raise RuntimeError("SUPABASE_URL is not configured")
    if not settings.supabase_service_role_key:
        raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY is not configured")


def _headers() -> dict[str, str]:
    _require_supabase_config()
    return {
        "Authorization": f"Bearer {settings.supabase_service_role_key}",
        "apikey": settings.supabase_service_role_key,
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }


async def audit_log(
    *,
    actor_user_id: str,
    action: str,
    entity_type: Optional[str] = None,
    entity_id: Optional[str] = None,
    ip: Optional[str] = None,
    user_agent: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """
    Best-effort audit logging.
    If it fails, it should NOT break the main request.
    """
    try:
        payload = {
            "actor_user_id": actor_user_id,
            "action": action,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "ip": ip,
            "user_agent": user_agent,
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        url = f"{settings.supabase_url.rstrip('/')}/rest/v1/audit_logs"

        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(url, headers=_headers(), json=payload)

        # swallow errors (best-effort)
        if r.status_code >= 300:
            return
    except Exception:
        return
