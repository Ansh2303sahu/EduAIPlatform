"""
Phase 11.4 MCP workflow history persistence repository.

This repository stores and retrieves redacted workflow metadata through the
same Supabase PostgREST service-role pattern used elsewhere in the backend.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from app.core.config import settings


def _require_supabase_config() -> None:
    if not settings.supabase_url:
        raise RuntimeError("SUPABASE_URL is not configured")
    if not settings.supabase_service_role_key:
        raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY is not configured")


def _headers(*, prefer_return: bool = False, prefer_count: bool = False) -> dict[str, str]:
    _require_supabase_config()
    headers = {
        "Authorization": f"Bearer {settings.supabase_service_role_key}",
        "apikey": settings.supabase_service_role_key,
        "Content-Type": "application/json",
    }
    if prefer_count:
        headers["Prefer"] = "count=exact"
    else:
        headers["Prefer"] = "return=representation" if prefer_return else "return=minimal"
    return headers


def _base_url() -> str:
    _require_supabase_config()
    return settings.supabase_url.rstrip("/")


def _count_from_content_range(resp: httpx.Response) -> int:
    content_range = (resp.headers.get("content-range") or "").strip()
    if "/" not in content_range:
        return 0
    try:
        return int(content_range.split("/")[-1])
    except Exception:
        return 0


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat()


class WorkflowHistoryRepo:
    runs_table = "mcp_workflow_runs"
    steps_table = "mcp_workflow_steps"

    async def insert_run(self, row: dict[str, Any]) -> dict[str, Any]:
        url = f"{_base_url()}/rest/v1/{self.runs_table}"
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(url, headers=_headers(prefer_return=True), json=row)
        if resp.status_code >= 300:
            raise RuntimeError(
                f"{self.runs_table} insert failed: {resp.status_code} {resp.text}"
            )
        data = resp.json() or []
        return data[0] if data else {}

    async def insert_steps(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        url = f"{_base_url()}/rest/v1/{self.steps_table}"
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(url, headers=_headers(prefer_return=False), json=rows)
        if resp.status_code >= 300:
            raise RuntimeError(
                f"{self.steps_table} bulk insert failed: {resp.status_code} {resp.text}"
            )

    async def list_runs(
        self,
        *,
        limit: int,
        offset: int,
        workflow_name: str | None = None,
        role: str | None = None,
        final_status: str | None = None,
        user_id: str | None = None,
        partial_only: bool = False,
        failed_only: bool = False,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        params: dict[str, str] = {
            "select": "*",
            "order": "started_at.desc",
            "limit": str(limit),
            "offset": str(offset),
        }
        if workflow_name:
            params["workflow_name"] = f"eq.{workflow_name}"
        if role:
            params["role"] = f"eq.{role}"
        if user_id:
            params["user_id"] = f"eq.{user_id}"
        if final_status:
            params["final_status"] = f"eq.{final_status}"
        elif partial_only:
            params["final_status"] = "eq.partial"
        elif failed_only:
            params["final_status"] = "in.(failed,blocked)"
        if date_from:
            params["started_at"] = f"gte.{_iso(date_from)}"
        if date_to:
            params["finished_at"] = f"lte.{_iso(date_to)}"

        url = f"{_base_url()}/rest/v1/{self.runs_table}"
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url, headers=_headers(prefer_count=True), params=params)
        if resp.status_code >= 300:
            raise RuntimeError(
                f"{self.runs_table} list failed: {resp.status_code} {resp.text}"
            )
        rows = resp.json() or []
        return rows, _count_from_content_range(resp)

    async def get_run(self, workflow_run_id: str) -> dict[str, Any] | None:
        url = f"{_base_url()}/rest/v1/{self.runs_table}"
        params = {
            "workflow_run_id": f"eq.{workflow_run_id}",
            "select": "*",
            "limit": "1",
        }
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url, headers=_headers(prefer_return=True), params=params)
        if resp.status_code >= 300:
            raise RuntimeError(
                f"{self.runs_table} get failed: {resp.status_code} {resp.text}"
            )
        rows = resp.json() or []
        return rows[0] if rows else None

    async def list_steps(self, workflow_run_id: str) -> list[dict[str, Any]]:
        url = f"{_base_url()}/rest/v1/{self.steps_table}"
        params = {
            "workflow_run_id": f"eq.{workflow_run_id}",
            "select": "*",
            "order": "step_index.asc",
        }
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url, headers=_headers(prefer_return=True), params=params)
        if resp.status_code >= 300:
            raise RuntimeError(
                f"{self.steps_table} list failed: {resp.status_code} {resp.text}"
            )
        return resp.json() or []

    async def list_failed_steps(self, *, limit: int = 500) -> list[dict[str, Any]]:
        url = f"{_base_url()}/rest/v1/{self.steps_table}"
        params = {
            "select": "tool_name,step_name,error_code",
            "step_status": "eq.failed",
            "order": "created_at.desc",
            "limit": str(limit),
        }
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url, headers=_headers(prefer_return=True), params=params)
        if resp.status_code >= 300:
            raise RuntimeError(
                f"{self.steps_table} failed-step list failed: {resp.status_code} {resp.text}"
            )
        return resp.json() or []
