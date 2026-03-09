from __future__ import annotations

from typing import Any, Dict

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.deps import CurrentUser, get_current_user

router = APIRouter(prefix="/prof/ingestion", tags=["prof-ingestion"])


class EnqueueIn(BaseModel):
    file_id: str = Field(..., min_length=10)
    job_type: str = Field(default="full", pattern="^(full|extract_text)$")


class EnqueueOut(BaseModel):
    job_id: str


def _require_supabase_rpc_config() -> None:
    if not settings.supabase_url:
        raise HTTPException(status_code=500, detail="SUPABASE_URL not configured")
    if not settings.supabase_anon_key:
        raise HTTPException(status_code=500, detail="SUPABASE_ANON_KEY not configured")


def _user_headers(user: CurrentUser) -> Dict[str, str]:
    _require_supabase_rpc_config()
    if not user.access_token:
        raise HTTPException(status_code=401, detail="Missing user access token")
    return {
        "apikey": settings.supabase_anon_key,
        "Authorization": f"Bearer {user.access_token}",
        "Content-Type": "application/json",
    }


def _require_professor(user: CurrentUser) -> None:
    # your project already has roles in profile/me
    if user.role not in ("professor", "admin"):
        raise HTTPException(status_code=403, detail="Professor access required")


@router.post("/enqueue", response_model=EnqueueOut)
async def enqueue_prof_job(body: EnqueueIn, user: CurrentUser = Depends(get_current_user)):
    """
    Calls RPC create_prof_ingestion_job as the logged-in user
    so auth.uid() works in SQL.
    """
    _require_professor(user)

    url = f"{settings.supabase_url.rstrip('/')}/rest/v1/rpc/create_prof_ingestion_job"
    payload: Dict[str, Any] = {"p_file_id": body.file_id, "p_job_type": body.job_type}

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers=_user_headers(user), json=payload)

    if r.status_code >= 300:
        raise HTTPException(
            status_code=400,
            detail=f"enqueue failed: {r.status_code} {r.text}",
        )

    job_id = r.json()
    if isinstance(job_id, dict) and "job_id" in job_id:
        job_id = job_id["job_id"]

    return EnqueueOut(job_id=str(job_id))
