from __future__ import annotations

from typing import Any, Dict, Optional

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


class ExtractedTablesRepo:
    def __init__(self) -> None:
        _require_supabase_config()
        self.base_url = settings.supabase_url.rstrip("/")

    async def insert_table(
        self,
        *,
        user_id: str,
        submission_id: Optional[str],
        file_id: str,
        job_id: str,
        source_sha256: str,
        table_index: int,
        sheet_name: Optional[str],
        columns: list[Any],
        rows: list[Any],
    ) -> None:
        url = f"{self.base_url}/rest/v1/extracted_tables"
        row = {
            "user_id": user_id,
            "submission_id": submission_id,
            "file_id": file_id,
            "job_id": job_id,
            "source_sha256": source_sha256,
            "table_index": table_index,
            "sheet_name": sheet_name,
            "columns": columns,
            "rows": rows,
        }

        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(url, headers=_headers(), json=row)

        if r.status_code >= 300:
            raise RuntimeError(f"extracted_tables insert failed: {r.status_code} {r.text}")
