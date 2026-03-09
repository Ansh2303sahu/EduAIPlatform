from __future__ import annotations

import httpx
from typing import Any, Dict, Optional

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


class FilesRepo:
    """
    Service-role repository for interacting with public.files.
    Used by worker and internal processes only.
    """

    def __init__(self) -> None:
        _require_supabase_config()
        self.base_url = settings.supabase_url.rstrip("/")

    # ---------------------------------------------------
    # INSERT
    # ---------------------------------------------------
    async def insert(self, *, row: dict[str, Any]) -> None:
        url = f"{self.base_url}/rest/v1/files"

        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(url, headers=_headers(prefer_return=False), json=row)

        if r.status_code >= 300:
            raise RuntimeError(f"files insert failed: {r.status_code} {r.text}")

    # ---------------------------------------------------
    # UPDATE (defense-in-depth includes user_id filter)
    # ---------------------------------------------------
    async def update(self, *, file_id: str, user_id: str, patch: dict[str, Any]) -> None:
        url = (
            f"{self.base_url}/rest/v1/files"
            f"?id=eq.{file_id}&user_id=eq.{user_id}"
        )

        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.patch(url, headers=_headers(prefer_return=False), json=patch)

        if r.status_code >= 300:
            raise RuntimeError(f"files update failed: {r.status_code} {r.text}")

    # ---------------------------------------------------
    # READ (service-role, worker usage)
    # ---------------------------------------------------
    async def get_file_record_service(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Service-role read from PostgREST for worker/internal tasks.
        Returns storage fields + security fields for gating.
        """

        url = f"{self.base_url}/rest/v1/files"

        params = {
            "id": f"eq.{file_id}",
            "select": ",".join(
    [
        "id",
        "user_id",
        "submission_id",
        "mime_type",
        "size_bytes",
        "sha256",
        "scan_result",
        "quarantined_until",
        "original_name",
        "bucket",
        "object_path",
    ]
),

        }

        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url, headers=_headers(prefer_return=True), params=params)

        if resp.status_code >= 300:
            raise RuntimeError(f"files read failed: {resp.status_code} {resp.text}")

        rows = resp.json()
        if not rows:
            return None

        return rows[0]
