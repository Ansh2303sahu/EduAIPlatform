from __future__ import annotations

from typing import Any, Optional

from app.core.config import settings
from app.services.ingestion_repo import IngestionRepo


class IngestionService:
    """
    Service layer.

    - Create job RPC must use USER JWT (so auth.uid() works)
    - Worker/job status reads can use SERVICE ROLE
    """

    def _extract_access_token(self, user: Any) -> str:
        """
        Supports:
        - dict-like: {"access_token": "..."}
        - object-like: user.access_token
        """
        if isinstance(user, dict):
            tok = user.get("access_token")
            if tok:
                return str(tok)

        tok = getattr(user, "access_token", None)
        if tok:
            return str(tok)

        raise ValueError("Missing access_token in user context")

    def _extract_user_id(self, user: Any) -> Optional[str]:
        if isinstance(user, dict):
            uid = user.get("id") or user.get("user_id")
            return str(uid) if uid else None
        uid = getattr(user, "id", None) or getattr(user, "user_id", None)
        return str(uid) if uid else None

    def _extract_role(self, user: Any) -> Optional[str]:
        if isinstance(user, dict):
            r = user.get("role")
            return str(r) if r else None
        r = getattr(user, "role", None)
        return str(r) if r else None

    def _repo(self) -> IngestionRepo:
        # Repo initialized with service role (for worker-safe ops)
        return IngestionRepo(service_role_key=settings.supabase_service_role_key)

    async def create_job_for_user(
        self,
        *,
        file_id: str,
        job_type: str,
        user: Any,
    ) -> str:
        user_token = self._extract_access_token(user)
        repo = self._repo()

        # IMPORTANT: create_ingestion_job uses auth.uid() so pass user token
        return await repo.create_job_rpc(
            file_id=file_id,
            job_type=job_type,
            user_access_token=user_token,
        )

    async def get_job_for_user(
        self,
        *,
        job_id: str,
        user: Any,
    ) -> Optional[dict]:
        """
        Service-role read + enforce ownership in backend before returning.
        """
        repo = self._repo()
        job = await repo.get_job(job_id=job_id)
        if not job:
            return None

        user_id = self._extract_user_id(user)
        role = self._extract_role(user)

        if role != "admin" and user_id and str(job.get("user_id")) != str(user_id):
            return None

        return job

    async def create_job_for_access_token(
        self,
        *,
        file_id: str,
        job_type: str,
        access_token: str,
    ) -> str:
        """
        Convenience wrapper when the route has a raw Bearer token.
        """
        repo = self._repo()
        return await repo.create_job_rpc(
            file_id=file_id,
            job_type=job_type,
            user_access_token=access_token,
        )