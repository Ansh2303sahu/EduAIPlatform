from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, Optional

import httpx

from app.core.config import settings


TRANSIENT_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class IngestionRepo:
    """
    - Worker calls: service-role key (SUPABASE_SERVICE_ROLE_KEY)
    - User calls (create job): anon key + user access token (so auth.uid() works in SQL)

    This repo talks DIRECTLY to Supabase PostgREST/RPC.
    """

    def __init__(self, *, service_role_key: str, client: httpx.AsyncClient | None = None) -> None:
        base = self._resolve_base_url()
        if not base:
            raise RuntimeError("SUPABASE_URL is not configured")

        if not service_role_key:
            raise RuntimeError("service_role_key is required")

        self.base = base
        self.service_role_key = service_role_key
        self._client = client

    # -----------------------------
    # Base URL helpers
    # -----------------------------
    @staticmethod
    def _clean_url(value: str | None) -> str:
        """
        Normalizes URL from env/settings:
        - strips spaces
        - strips wrapping quotes
        - removes trailing slash
        - removes trailing /rest/v1 if present
        """
        raw = (value or "").strip()

        if not raw:
            return ""

        if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
            raw = raw[1:-1].strip()

        raw = raw.rstrip("/")

        # If user mistakenly stored the REST path already, normalize it.
        if raw.endswith("/rest/v1"):
            raw = raw[:-8].rstrip("/")

        return raw

    def _resolve_base_url(self) -> str:
        """
        Prefer explicit env if present, otherwise fall back to settings.supabase_url.
        Still resolves to Supabase base URL, not backend URL.
        """
        candidates = [
            os.getenv("SUPABASE_URL"),
            getattr(settings, "supabase_url", None),
        ]

        for candidate in candidates:
            cleaned = self._clean_url(candidate)
            if cleaned:
                return cleaned

        return ""

    # -----------------------------
    # Headers helpers
    # -----------------------------
    def _service_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.service_role_key}",
            "apikey": self.service_role_key,
            "Content-Type": "application/json",
        }

    def _user_headers(self, *, user_access_token: str) -> Dict[str, str]:
        if not settings.supabase_anon_key:
            raise RuntimeError("SUPABASE_ANON_KEY is not configured (required for user RPC calls)")
        if not user_access_token:
            raise RuntimeError("user_access_token is required")

        return {
            "Authorization": f"Bearer {user_access_token}",
            "apikey": settings.supabase_anon_key,
            "Content-Type": "application/json",
        }

    # -----------------------------
    # URL builders
    # -----------------------------
    def _rpc_url(self, rpc_name: str) -> str:
        return f"{self.base}/rest/v1/rpc/{rpc_name}"

    def _table_url(self, table_name: str) -> str:
        return f"{self.base}/rest/v1/{table_name}"

    # -----------------------------
    # HTTP helper with retry
    # -----------------------------
    def _timeout(self, timeout: float) -> httpx.Timeout:
        return httpx.Timeout(
            connect=_env_float("INGESTION_HTTP_CONNECT_TIMEOUT_SECONDS", 30.0),
            read=_env_float("INGESTION_HTTP_READ_TIMEOUT_SECONDS", timeout),
            write=_env_float("INGESTION_HTTP_WRITE_TIMEOUT_SECONDS", timeout),
            pool=_env_float("INGESTION_HTTP_POOL_TIMEOUT_SECONDS", 30.0),
        )

    async def _post_json(
        self,
        *,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        timeout: float,
    ) -> httpx.Response:
        request_timeout = self._timeout(timeout)
        if self._client is not None:
            return await self._client.post(url, headers=headers, json=payload, timeout=request_timeout)

        async with httpx.AsyncClient(timeout=request_timeout) as client:
            return await client.post(url, headers=headers, json=payload)

    async def _get_json(
        self,
        *,
        url: str,
        headers: Dict[str, str],
        params: Dict[str, Any] | None,
        timeout: float,
    ) -> httpx.Response:
        request_timeout = self._timeout(timeout)
        if self._client is not None:
            return await self._client.get(url, headers=headers, params=params, timeout=request_timeout)

        async with httpx.AsyncClient(timeout=request_timeout) as client:
            return await client.get(url, headers=headers, params=params)

    async def _sleep_before_retry(
        self,
        *,
        attempt: int,
        response: httpx.Response | None = None,
    ) -> None:
        retry_after = response.headers.get("retry-after") if response is not None else None
        if retry_after:
            try:
                await asyncio.sleep(float(retry_after))
                return
            except ValueError:
                pass

        base_sleep = _env_float("INGESTION_HTTP_RETRY_BASE_SLEEP", 1.5)
        await asyncio.sleep(base_sleep * (2 ** (attempt - 1)))

    async def _post_rpc(
        self,
        *,
        rpc_name: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        timeout: float = 20.0,
        retries: int = 5,
    ) -> httpx.Response:
        url = self._rpc_url(rpc_name)

        last_exc: Exception | None = None

        for attempt in range(1, retries + 1):
            try:
                resp = await self._post_json(url=url, headers=headers, payload=payload, timeout=timeout)
                if resp.status_code in TRANSIENT_HTTP_STATUS_CODES and attempt < retries:
                    print(
                        f"[IngestionRepo] RPC attempt {attempt}/{retries} got "
                        f"{resp.status_code} for {rpc_name}; retrying"
                    )
                    await self._sleep_before_retry(attempt=attempt, response=resp)
                    continue
                return resp

            except (
                httpx.ConnectTimeout,
                httpx.ConnectError,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
                httpx.NetworkError,
            ) as e:
                last_exc = e
                print(f"[IngestionRepo] RPC attempt {attempt}/{retries} failed for {rpc_name}: {type(e).__name__}: {e}")
                if attempt < retries:
                    await self._sleep_before_retry(attempt=attempt)

        raise RuntimeError(
            f"RPC call failed after retries: {rpc_name}: url={url} error={type(last_exc).__name__ if last_exc else 'Unknown'}: {last_exc}"
        )

    async def _get(
        self,
        *,
        url: str,
        headers: Dict[str, str],
        params: Dict[str, Any] | None = None,
        timeout: float = 20.0,
        retries: int = 5,
    ) -> httpx.Response:
        last_exc: Exception | None = None

        for attempt in range(1, retries + 1):
            try:
                resp = await self._get_json(url=url, headers=headers, params=params, timeout=timeout)
                if resp.status_code in TRANSIENT_HTTP_STATUS_CODES and attempt < retries:
                    print(f"[IngestionRepo] GET attempt {attempt}/{retries} got {resp.status_code} for {url}; retrying")
                    await self._sleep_before_retry(attempt=attempt, response=resp)
                    continue
                return resp
            except (
                httpx.ConnectTimeout,
                httpx.ConnectError,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
                httpx.NetworkError,
            ) as e:
                last_exc = e
                print(f"[IngestionRepo] GET attempt {attempt}/{retries} failed for {url}: {type(e).__name__}: {e}")
                if attempt < retries:
                    await self._sleep_before_retry(attempt=attempt)

        raise RuntimeError(
            f"GET request failed after retries: url={url} error={type(last_exc).__name__ if last_exc else 'Unknown'}: {last_exc}"
        ) from last_exc

    # -----------------------------
    # USER RPC: create job
    # -----------------------------
    async def create_job_rpc(
        self,
        *,
        file_id: str,
        job_type: str,
        user_access_token: str,
    ) -> str:
        """
        Calls SQL RPC: create_ingestion_job(p_file_id, p_job_type)
        IMPORTANT: must be called with user JWT so auth.uid() is set.
        """
        payload = {"p_file_id": file_id, "p_job_type": job_type}

        resp = await self._post_rpc(
            rpc_name="create_ingestion_job",
            payload=payload,
            headers=self._user_headers(user_access_token=user_access_token),
            timeout=30.0,
            retries=5,
        )

        if resp.status_code >= 300:
            raise ValueError(
                f"create_ingestion_job failed: status={resp.status_code} url={self._rpc_url('create_ingestion_job')} body={resp.text}"
            )

        data = resp.json()
        if data is None or data == "":
            raise ValueError("create_ingestion_job returned empty response")

        if isinstance(data, dict) and "job_id" in data:
            return str(data["job_id"])

        return str(data)

    # -----------------------------
    # SERVICE: read job
    # -----------------------------
    async def get_job(self, *, job_id: str) -> Optional[Dict[str, Any]]:
        url = self._table_url("ingestion_jobs")
        params = {"id": f"eq.{job_id}", "select": "*"}

        resp = await self._get(
            url=url,
            headers=self._service_headers(),
            params=params,
            timeout=20.0,
        )

        if resp.status_code >= 300:
            raise RuntimeError(f"get_job failed: status={resp.status_code} url={url} body={resp.text}")

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
        """
        DB has overloaded claim_next_ingestion_job; ALWAYS pass p_worker_mode to disambiguate.
        worker_mode is read from env WORKER_MODE (all/light/heavy).
        """
        worker_mode = os.getenv("WORKER_MODE", "all").lower().strip()

        payload = {
            "p_worker_id": worker_id,
            "p_worker_mode": worker_mode,
            "p_max_attempts": int(max_attempts),
            "p_lock_timeout_seconds": int(lock_timeout_seconds),
        }

        resp = await self._post_rpc(
            rpc_name="claim_next_ingestion_job",
            payload=payload,
            headers=self._service_headers(),
            timeout=20.0,
            retries=5,
        )

        if resp.status_code >= 300:
            raise RuntimeError(
                f"claim_next_ingestion_job failed: status={resp.status_code} url={self._rpc_url('claim_next_ingestion_job')} body={resp.text}"
            )

        rows = resp.json() or []
        return rows[0] if rows else None

    async def mark_done(self, *, job_id: str, worker_id: str, details: dict | None = None) -> None:
        payload = {
            "p_job_id": job_id,
            "p_worker_id": worker_id,
            "p_details": details or {},
        }

        resp = await self._post_rpc(
            rpc_name="mark_ingestion_job_done",
            payload=payload,
            headers=self._service_headers(),
            timeout=20.0,
            retries=5,
        )

        if resp.status_code >= 300:
            raise RuntimeError(
                f"mark_ingestion_job_done failed: status={resp.status_code} url={self._rpc_url('mark_ingestion_job_done')} body={resp.text}"
            )

    async def mark_failed(
        self,
        *,
        job_id: str,
        worker_id: str,
        error_code: str,
        error_message: str,
        details: dict | None = None,
    ) -> None:
        payload = {
            "p_job_id": job_id,
            "p_worker_id": worker_id,
            "p_error_code": error_code,
            "p_error_message": error_message,
            "p_details": details or {},
        }

        resp = await self._post_rpc(
            rpc_name="mark_ingestion_job_failed",
            payload=payload,
            headers=self._service_headers(),
            timeout=20.0,
            retries=5,
        )

        if resp.status_code >= 300:
            raise RuntimeError(
                f"mark_ingestion_job_failed failed: status={resp.status_code} url={self._rpc_url('mark_ingestion_job_failed')} body={resp.text}"
            )
