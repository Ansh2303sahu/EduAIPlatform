from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import httpx

from app.core.config import settings


class ProfIngestionRepo:
    """
    Professor ingestion pipeline (separate consumer).

    - Worker calls: service-role key (SUPABASE_SERVICE_ROLE_KEY)
    - Professor UI calls (create job): anon key + user access token
      so auth.uid() works inside create_prof_ingestion_job().
    """

    def __init__(self, *, service_role_key: str) -> None:
        if not settings.supabase_url:
            raise RuntimeError("SUPABASE_URL is not configured")
        if not service_role_key:
            raise RuntimeError("service_role_key is required")

        self.base = settings.supabase_url.rstrip("/")
        self.service_role_key = service_role_key

    def _service_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.service_role_key}",
            "apikey": self.service_role_key,
            "Content-Type": "application/json",
        }

    def _user_headers(self, *, user_access_token: str) -> Dict[str, str]:
        if not settings.supabase_anon_key:
            raise RuntimeError("SUPABASE_ANON_KEY is not configured")
        if not user_access_token:
            raise RuntimeError("user_access_token is required")

        return {
            "Authorization": f"Bearer {user_access_token}",
            "apikey": settings.supabase_anon_key,
            "Content-Type": "application/json",
        }

    async def _post_rpc(
        self,
        *,
        rpc_name: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        timeout: float = 20.0,
        retries: int = 5,
    ) -> httpx.Response:
        url = f"{self.base}/rest/v1/rpc/{rpc_name}"

        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(url, headers=headers, json=payload)
                return resp
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
                last_exc = e
                await asyncio.sleep(0.5 * (2**attempt))

        raise RuntimeError(f"RPC call failed after retries: {rpc_name}: {last_exc}")

    # -----------------------------
    # USER RPC: create professor job
    # -----------------------------
    async def create_job_rpc(
        self,
        *,
        file_id: str,
        job_type: str,
        user_access_token: str,
    ) -> str:
        payload = {"p_file_id": file_id, "p_job_type": job_type}

        resp = await self._post_rpc(
            rpc_name="create_prof_ingestion_job",
            payload=payload,
            headers=self._user_headers(user_access_token=user_access_token),
            timeout=30.0,
            retries=5,
        )

        if resp.status_code >= 300:
            raise ValueError(f"create_prof_ingestion_job failed: {resp.status_code} {resp.text}")

        data = resp.json()
        if not data:
            raise ValueError("create_prof_ingestion_job returned empty response")

        return str(data)

    # -----------------------------
    # SERVICE: read professor job
    # -----------------------------
    async def get_job(self, *, job_id: str) -> Optional[Dict[str, Any]]:
        url = f"{self.base}/rest/v1/prof_ingestion_jobs"
        params = {"id": f"eq.{job_id}", "select": "*"}
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url, headers=self._service_headers(), params=params)

        if resp.status_code >= 300:
            raise RuntimeError(f"get prof job failed: {resp.status_code} {resp.text}")

        rows = resp.json() or []
        return rows[0] if rows else None

    # -----------------------------
    # WORKER RPC: claim / done / failed
    # -----------------------------
    async def claim_next_job(
        self,
        *,
        worker_id: str,
        max_attempts: int = 3,
        lock_timeout_seconds: int = 600,
    ) -> Optional[Dict[str, Any]]:
        payload = {
            "p_worker_id": worker_id,
            "p_max_attempts": int(max_attempts),
            "p_lock_timeout_seconds": int(lock_timeout_seconds),
        }

        resp = await self._post_rpc(
            rpc_name="claim_next_prof_job",
            payload=payload,
            headers=self._service_headers(),
            timeout=20.0,
            retries=5,
        )

        if resp.status_code >= 300:
            raise RuntimeError(f"claim_next_prof_job failed: {resp.status_code} {resp.text}")

        rows = resp.json() or []
        return rows[0] if rows else None

    async def mark_done(self, *, job_id: str, worker_id: str, details: dict | None = None) -> None:
        payload = {"p_job_id": job_id, "p_worker_id": worker_id, "p_details": details or {}}

        resp = await self._post_rpc(
            rpc_name="mark_prof_job_done",
            payload=payload,
            headers=self._service_headers(),
            timeout=20.0,
            retries=5,
        )

        if resp.status_code >= 300:
            raise RuntimeError(f"mark_prof_job_done failed: {resp.status_code} {resp.text}")

    async def mark_failed(
        self,
        *,
        job_id: str,
        worker_id: str,
        error_code: str,
        error_message: str,
    ) -> None:
        payload = {
            "p_job_id": job_id,
            "p_worker_id": worker_id,
            "p_error_code": error_code,
            "p_error_message": error_message,
        }

        resp = await self._post_rpc(
            rpc_name="mark_prof_job_failed",
            payload=payload,
            headers=self._service_headers(),
            timeout=20.0,
            retries=5,
        )

        if resp.status_code >= 300:
            raise RuntimeError(f"mark_prof_job_failed failed: {resp.status_code} {resp.text}")

    # ----------------------------------------------------------
    # SERVICE RPC: insert extracted text safely into prof_insights
    # ----------------------------------------------------------
    async def insert_prof_insight(
        self,
        *,
        file_id: str,
        job_id: str,
        user_id: str,
        source_sha256: str,
        redacted_text: str,
        redaction_summary: dict | None,
    ) -> None:
        payload = {
            "p_file_id": file_id,
            "p_job_id": job_id,
            "p_user_id": user_id,
            "p_source_sha256": source_sha256,
            "p_redacted_text": redacted_text,
            "p_redaction_summary": redaction_summary or {},
        }

        resp = await self._post_rpc(
            rpc_name="insert_prof_insight",
            payload=payload,
            headers=self._service_headers(),
            timeout=30.0,
            retries=5,
        )

        if resp.status_code >= 300:
            raise RuntimeError(f"insert_prof_insight failed: {resp.status_code} {resp.text}")

    # ----------------------------------------------------------
    # SERVICE: fetch latest insight row for a file (results page)
    # ----------------------------------------------------------
    async def get_latest_insight_by_file(self, *, file_id: str) -> Optional[Dict[str, Any]]:
        url = f"{self.base}/rest/v1/prof_insights"
        params = {
            "file_id": f"eq.{file_id}",
            "select": "redacted_text,redaction_summary,created_at,job_id",
            "order": "created_at.desc",
            "limit": "1",
        }
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url, headers=self._service_headers(), params=params)

        if resp.status_code >= 300:
            raise RuntimeError(f"get_latest_insight_by_file failed: {resp.status_code} {resp.text}")

        rows = resp.json() or []
        return rows[0] if rows else None
