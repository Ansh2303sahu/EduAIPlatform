from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException

from app.core.config import settings
from app.core.deps import CurrentUser, get_current_user


router = APIRouter(prefix="/prof", tags=["prof-results"])


def _require_professor(user: CurrentUser) -> None:
    if user.role not in ("professor", "admin"):
        raise HTTPException(status_code=403, detail="Professor access required")


def _service_headers() -> Dict[str, str]:
    if not settings.supabase_url:
        raise HTTPException(status_code=500, detail="SUPABASE_URL not configured")
    if not settings.supabase_service_role_key:
        raise HTTPException(status_code=500, detail="SUPABASE_SERVICE_ROLE_KEY not configured")

    key = settings.supabase_service_role_key
    return {
        "Authorization": f"Bearer {key}",
        "apikey": key,
        "Content-Type": "application/json",
    }


async def _get_json(url: str, *, params: Dict[str, str]) -> list:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, headers=_service_headers(), params=params)
    if r.status_code >= 300:
        raise HTTPException(status_code=400, detail=f"Supabase fetch failed: {r.status_code} {r.text}")
    return r.json() or []


@router.get("/results/{file_id}")
async def get_prof_results(file_id: str, user: CurrentUser = Depends(get_current_user)):
    """
    Returns exactly what the frontend expects:
    {
      file: {id,status,mime_type,created_at},
      jobs: [...],
      text: {redacted_text, redaction_summary, created_at} | null,
      events: [...]
    }
    """
    _require_professor(user)

    base = settings.supabase_url.rstrip("/")
    files_url = f"{base}/rest/v1/files"
    jobs_url = f"{base}/rest/v1/prof_ingestion_jobs"
    insights_url = f"{base}/rest/v1/prof_insights"

    # 1) FILE (and ownership gate)
    file_rows = await _get_json(
        files_url,
        params={
            "id": f"eq.{file_id}",
            "select": "id,status,mime_type,created_at,user_id",
            "limit": "1",
        },
    )
    if not file_rows:
        raise HTTPException(status_code=404, detail="File not found")

    file_row = file_rows[0]

    # If you want strict ownership (recommended):
    # allow admin to view all, professor can view only own files
    if user.role != "admin":
        # CurrentUser in your project already has user_id/id (commonly "id")
        # Adjust if your CurrentUser uses "user_id" instead.
        req_user_id = getattr(user, "id", None) or getattr(user, "user_id", None)
        if req_user_id and str(file_row.get("user_id")) != str(req_user_id):
            raise HTTPException(status_code=403, detail="Not allowed to view this file")

    file_out = {
        "id": str(file_row.get("id")),
        "status": str(file_row.get("status") or ""),
        "mime_type": file_row.get("mime_type"),
        "created_at": file_row.get("created_at"),
    }

    # 2) JOBS
    jobs = await _get_json(
        jobs_url,
        params={
            "file_id": f"eq.{file_id}",
            "select": "id,status,job_type,created_at,error_code,error_message",
            "order": "created_at.desc",
            "limit": "50",
        },
    )
    jobs_out = [
        {
            "id": str(j.get("id")),
            "status": j.get("status"),
            "job_type": j.get("job_type"),
            "created_at": j.get("created_at"),
            "error_code": j.get("error_code"),
            "error_message": j.get("error_message"),
        }
        for j in jobs
    ]

    # 3) TEXT (latest from prof_insights)
    insight_rows = await _get_json(
        insights_url,
        params={
            "file_id": f"eq.{file_id}",
            "select": "redacted_text,redaction_summary,created_at,job_id",
            "order": "created_at.desc",
            "limit": "1",
        },
    )
    if insight_rows:
        i = insight_rows[0]
        text_out = {
            "redacted_text": i.get("redacted_text"),
            "redaction_summary": i.get("redaction_summary"),
            "created_at": i.get("created_at"),
        }
    else:
        text_out = None

    # 4) EVENTS (your project sometimes has prof_events OR prof_processing_events)
    events_out: List[Dict[str, Any]] = []
    for table_name in ("prof_events", "prof_processing_events"):
        try:
            events_url = f"{base}/rest/v1/{table_name}"
            evs = await _get_json(
                events_url,
                params={
                    "file_id": f"eq.{file_id}",
                    "select": "event_type,created_at,details",
                    "order": "created_at.asc",
                    "limit": "200",
                },
            )
            events_out = [
                {
                    "event_type": e.get("event_type"),
                    "created_at": e.get("created_at"),
                    "details": e.get("details"),
                }
                for e in evs
            ]
            break
        except HTTPException:
            continue

    return {
        "file": file_out,
        "jobs": jobs_out,
        "text": text_out,
        "events": events_out,
    }
