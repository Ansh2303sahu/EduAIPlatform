# backend/app/api/ingestion.py
from __future__ import annotations

from typing import Optional, Dict, Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.deps import CurrentUser, get_current_user

router = APIRouter(prefix="/ingestion", tags=["ingestion"])


class EnqueueIn(BaseModel):
    file_id: str = Field(..., min_length=1)
    job_type: Optional[str] = Field(default="full")  # ✅ default FULL


class EnqueueOut(BaseModel):
    job_id: str
    job_type: str


def _require_supabase_config() -> None:
    if not settings.supabase_url:
        raise HTTPException(status_code=500, detail="SUPABASE_URL not configured")
    if not settings.supabase_anon_key:
        raise HTTPException(status_code=500, detail="SUPABASE_ANON_KEY not configured")


def _user_headers(user: CurrentUser) -> Dict[str, str]:
    _require_supabase_config()
    token = getattr(user, "access_token", None)
    if not token:
        raise HTTPException(status_code=401, detail="Missing user access token")
    return {
        "apikey": settings.supabase_anon_key,
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


@router.post("/enqueue", response_model=EnqueueOut)
async def enqueue(body: EnqueueIn, user: CurrentUser = Depends(get_current_user)):
    """
    Enqueue ingestion job for this file_id.
    Default is FULL so images get OCR, tables get parsed, media gets transcript.
    """
    job_type = (body.job_type or "full").strip().lower()
    if job_type not in {"full", "extract_text", "parse_tables", "extract_images", "transcribe_audio"}:
        raise HTTPException(status_code=400, detail=f"Invalid job_type: {job_type}")

    url = f"{settings.supabase_url.rstrip('/')}/rest/v1/rpc/create_ingestion_job"
    payload = {"p_file_id": body.file_id, "p_job_type": job_type}

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers=_user_headers(user), json=payload)

    if r.status_code >= 300:
        raise HTTPException(status_code=400, detail=f"enqueue failed: {r.status_code} {r.text}")

    job_id = r.json()
    if isinstance(job_id, dict) and "job_id" in job_id:
        job_id = job_id["job_id"]

    return EnqueueOut(job_id=str(job_id), job_type=job_type)
