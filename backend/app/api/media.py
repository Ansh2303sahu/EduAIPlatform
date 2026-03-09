from __future__ import annotations

from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.core.deps import CurrentUser, get_current_user
from app.services.storage import create_signed_download_url

router = APIRouter(prefix="/media", tags=["media"])


class SignedUrlOut(BaseModel):
    url: str


def _require_supabase() -> None:
    if not settings.supabase_url:
        raise HTTPException(status_code=500, detail="SUPABASE_URL not configured")
    if not settings.supabase_service_role_key:
        raise HTTPException(status_code=500, detail="SUPABASE_SERVICE_ROLE_KEY not configured")
    if not settings.signed_url_expires_seconds:
        raise HTTPException(status_code=500, detail="SIGNED_URL_EXPIRES_SECONDS not configured")


def _service_headers() -> Dict[str, str]:
    _require_supabase()
    return {
        "apikey": settings.supabase_service_role_key,
        "Authorization": f"Bearer {settings.supabase_service_role_key}",
        "Content-Type": "application/json",
    }


async def _get_rows(path: str) -> list:
    url = f"{settings.supabase_url.rstrip('/')}/rest/v1/{path}"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, headers=_service_headers())
    if r.status_code >= 400:
        raise HTTPException(status_code=400, detail=f"Supabase fetch failed: {r.status_code} {r.text}")
    return r.json() or []


@router.get("/signed-url", response_model=SignedUrlOut)
async def get_media_signed_url(
    file_id: str,
    derived_path: str,
    derived_bucket: Optional[str] = None,
    user: CurrentUser = Depends(get_current_user),
):
    """
    Returns a signed URL for derived media (images extracted by worker).
    Ownership enforced via service-role read: file.user_id must match current user (unless admin).
    """
    _require_supabase()

    # 1) ownership check: file must belong to user (unless admin)
    if user.role == "admin":
        files = await _get_rows(f"files?id=eq.{file_id}&select=id,user_id")
    else:
        files = await _get_rows(f"files?id=eq.{file_id}&user_id=eq.{user.id}&select=id,user_id")

    if not files:
        raise HTTPException(status_code=404, detail="Not found")

    # 2) issue signed URL
    bucket = derived_bucket or settings.uploads_bucket
    if not bucket:
        raise HTTPException(status_code=500, detail="UPLOADS_BUCKET not configured")

    url = await create_signed_download_url(
        bucket=bucket,
        path=derived_path,
        expires_in=settings.signed_url_expires_seconds,
    )
    return SignedUrlOut(url=url)
