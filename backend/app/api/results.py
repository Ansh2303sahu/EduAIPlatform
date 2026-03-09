from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.deps import CurrentUser, get_current_user

router = APIRouter(prefix="/results", tags=["results"])


# -------------------------
# Models
# -------------------------

class ResultFile(BaseModel):
    id: str
    status: str
    mime_type: Optional[str] = None
    submission_id: Optional[str] = None
    created_at: Optional[str] = None


class ResultText(BaseModel):
    redacted_text: str = ""
    redaction_summary: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None


class ResultTable(BaseModel):
    table_index: int
    sheet_name: Optional[str] = None
    columns: List[str] = Field(default_factory=list)
    rows: List[List[Any]] = Field(default_factory=list)
    created_at: Optional[str] = None


class ResultMedia(BaseModel):
    media_index: int
    media_type: str
    width: Optional[int] = None
    height: Optional[int] = None
    caption: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None


class ResultTranscript(BaseModel):
    redacted_transcript: str = ""
    redaction_summary: Dict[str, Any] = Field(default_factory=dict)
    asr_model_name: Optional[str] = None  # ✅ renamed to avoid Pydantic model_ warning
    confidence: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None


class ProcessingEvent(BaseModel):
    event_type: str
    details: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None


class IngestionJob(BaseModel):
    id: str
    status: str
    job_type: str
    created_at: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class ResultsOut(BaseModel):
    file: ResultFile
    jobs: List[IngestionJob] = Field(default_factory=list)
    text: Optional[ResultText] = None
    tables: List[ResultTable] = Field(default_factory=list)
    media: List[ResultMedia] = Field(default_factory=list)
    transcript: Optional[ResultTranscript] = None
    events: List[ProcessingEvent] = Field(default_factory=list)


class TranscriptOut(BaseModel):
    ready: bool
    items: List[Dict[str, Any]] = Field(default_factory=list)


# -------------------------
# Helpers
# -------------------------

def _require_supabase_config() -> None:
    if not settings.supabase_url:
        raise HTTPException(status_code=500, detail="SUPABASE_URL not configured")
    if not settings.supabase_service_role_key:
        raise HTTPException(status_code=500, detail="SUPABASE_SERVICE_ROLE_KEY not configured")


def _service_headers() -> Dict[str, str]:
    _require_supabase_config()
    return {
        "apikey": settings.supabase_service_role_key,
        "Authorization": f"Bearer {settings.supabase_service_role_key}",
        "Content-Type": "application/json",
    }


async def _get_rows(path: str, headers: Dict[str, str]) -> list:
    url = f"{settings.supabase_url.rstrip('/')}/rest/v1/{path}"
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, headers=headers)
    if r.status_code >= 400:
        raise HTTPException(status_code=400, detail=f"Supabase fetch failed: {r.status_code} {r.text}")
    return r.json() or []


async def _get_rows_adaptive(
    *,
    path_with_user_filter: str,
    path_without_user_filter: str,
    headers: Dict[str, str],
) -> list:
    """
    Some tables might not have user_id.
    Try user-filtered first (safer). If schema error, retry without.
    """
    try:
        return await _get_rows(path_with_user_filter, headers)
    except HTTPException as e:
        msg = str(e.detail).lower()
        if "could not find" in msg or "does not exist" in msg or "column" in msg:
            return await _get_rows(path_without_user_filter, headers)
        raise


def _map_transcript_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    DB uses 'model_name'. API returns 'asr_model_name'.
    """
    out = dict(row or {})
    out["asr_model_name"] = out.pop("model_name", None)
    return out


# -------------------------
# Routes
# -------------------------

@router.get("/transcript/{file_id}", response_model=TranscriptOut)
async def get_transcript(file_id: str, user: CurrentUser = Depends(get_current_user)):
    """
    Server-side read (service role) with ownership enforcement.
    Never 404 when not ready; returns {ready:false, items: []}.
    """
    headers = _service_headers()

    # Verify file exists and belongs to user (unless admin)
    file_rows = await _get_rows(f"files?id=eq.{file_id}&select=id,user_id", headers)
    if not file_rows:
        return TranscriptOut(ready=False, items=[])

    owner_id = str(file_rows[0].get("user_id") or "")
    if user.role != "admin" and owner_id != str(user.id):
        return TranscriptOut(ready=False, items=[])

    rows = await _get_rows_adaptive(
        headers=headers,
        path_with_user_filter=(
            f"transcripts?file_id=eq.{file_id}"
            f"&user_id=eq.{user.id}"
            f"&select=*"
            f"&order=created_at.desc"
            f"&limit=1"
        ),
        path_without_user_filter=(
            f"transcripts?file_id=eq.{file_id}"
            f"&select=*"
            f"&order=created_at.desc"
            f"&limit=1"
        ),
    )

    # map model_name -> asr_model_name if present
    mapped = [_map_transcript_row(r) for r in rows]
    return TranscriptOut(ready=bool(mapped), items=mapped)


@router.get("/{file_id}", response_model=ResultsOut)
async def get_results(file_id: str, user: CurrentUser = Depends(get_current_user)):
    """
    Returns ALL processing outputs for a given file_id.
    Uses service-role reads + strict ownership checks (avoid RLS empty results).
    """
    headers = _service_headers()

    # File (enforce ownership)
    if user.role == "admin":
        files = await _get_rows(
            f"files?id=eq.{file_id}&select=id,status,mime_type,submission_id,created_at,user_id",
            headers,
        )
    else:
        files = await _get_rows(
            f"files?id=eq.{file_id}&user_id=eq.{user.id}&select=id,status,mime_type,submission_id,created_at,user_id",
            headers,
        )

    if not files:
        raise HTTPException(status_code=404, detail="File not found")

    file_row = files[0]

    # Jobs
    jobs = await _get_rows_adaptive(
        headers=headers,
        path_with_user_filter=(
            f"ingestion_jobs?file_id=eq.{file_id}"
            f"&user_id=eq.{user.id}"
            f"&select=id,status,job_type,created_at,error_code,error_message"
            f"&order=created_at.desc&limit=20"
        ),
        path_without_user_filter=(
            f"ingestion_jobs?file_id=eq.{file_id}"
            f"&select=id,status,job_type,created_at,error_code,error_message"
            f"&order=created_at.desc&limit=20"
        ),
    )

    # Text
    text_rows = await _get_rows_adaptive(
        headers=headers,
        path_with_user_filter=(
            f"extracted_text?file_id=eq.{file_id}"
            f"&user_id=eq.{user.id}"
            f"&select=redacted_text,redaction_summary,created_at"
            f"&order=created_at.desc&limit=1"
        ),
        path_without_user_filter=(
            f"extracted_text?file_id=eq.{file_id}"
            f"&select=redacted_text,redaction_summary,created_at"
            f"&order=created_at.desc&limit=1"
        ),
    )

    # Tables
    table_rows = await _get_rows_adaptive(
        headers=headers,
        path_with_user_filter=(
            f"extracted_tables?file_id=eq.{file_id}"
            f"&user_id=eq.{user.id}"
            f"&select=table_index,sheet_name,columns,rows,created_at"
            f"&order=created_at.desc&limit=50"
        ),
        path_without_user_filter=(
            f"extracted_tables?file_id=eq.{file_id}"
            f"&select=table_index,sheet_name,columns,rows,created_at"
            f"&order=created_at.desc&limit=50"
        ),
    )

    # Media
    media_rows = await _get_rows_adaptive(
        headers=headers,
        path_with_user_filter=(
            f"extracted_media?file_id=eq.{file_id}"
            f"&user_id=eq.{user.id}"
            f"&select=media_index,media_type,width,height,caption,metadata,created_at"
            f"&order=created_at.desc&limit=50"
        ),
        path_without_user_filter=(
            f"extracted_media?file_id=eq.{file_id}"
            f"&select=media_index,media_type,width,height,caption,metadata,created_at"
            f"&order=created_at.desc&limit=50"
        ),
    )

    # Transcript
    transcript_rows = await _get_rows_adaptive(
        headers=headers,
        path_with_user_filter=(
            f"transcripts?file_id=eq.{file_id}"
            f"&user_id=eq.{user.id}"
            f"&select=redacted_transcript,redaction_summary,model_name,confidence,created_at"
            f"&order=created_at.desc&limit=1"
        ),
        path_without_user_filter=(
            f"transcripts?file_id=eq.{file_id}"
            f"&select=redacted_transcript,redaction_summary,model_name,confidence,created_at"
            f"&order=created_at.desc&limit=1"
        ),
    )
    transcript_rows = [_map_transcript_row(r) for r in transcript_rows]

    # Events
    event_rows = await _get_rows_adaptive(
        headers=headers,
        path_with_user_filter=(
            f"processing_events?file_id=eq.{file_id}"
            f"&user_id=eq.{user.id}"
            f"&select=event_type,details,created_at"
            f"&order=created_at.asc&limit=200"
        ),
        path_without_user_filter=(
            f"processing_events?file_id=eq.{file_id}"
            f"&select=event_type,details,created_at"
            f"&order=created_at.asc&limit=200"
        ),
    )

    return ResultsOut(
        file=ResultFile(
            id=file_row["id"],
            status=file_row["status"],
            mime_type=file_row.get("mime_type"),
            submission_id=file_row.get("submission_id"),
            created_at=file_row.get("created_at"),
        ),
        jobs=[IngestionJob(**j) for j in jobs],
        text=ResultText(**text_rows[0]) if text_rows else None,
        tables=[ResultTable(**t) for t in table_rows],
        media=[ResultMedia(**m) for m in media_rows],
        transcript=ResultTranscript(**transcript_rows[0]) if transcript_rows else None,
        events=[ProcessingEvent(**e) for e in event_rows],
    )
