# backend/app/api/files.py
import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

import httpx
import magic
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.deps import CurrentUser, get_current_user
from app.core.rate_limit import limiter
from app.services.clamav import clamav_scan_bytes

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/files", tags=["files"])

READ_CHUNK_BYTES = 1024 * 1024  # 1 MB

MIME_PDF = "application/pdf"
MIME_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
MIME_TEXT = "text/plain"

MIME_CSV = "text/csv"
MIME_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

MIME_IMAGES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"}
MIME_AUDIO = {
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/mp4",
    "audio/x-m4a",
    "audio/aac",
    "audio/flac",
    "audio/ogg",
    "audio/webm",
}
MIME_VIDEO = {"video/mp4", "video/webm", "video/quicktime", "video/x-matroska"}

MIME_OCTET = "application/octet-stream"

ALLOWED_MIME = {
    MIME_PDF,
    MIME_DOCX,
    MIME_TEXT,
    MIME_CSV,
    MIME_XLSX,
    *MIME_IMAGES,
    *MIME_AUDIO,
    *MIME_VIDEO,
    MIME_OCTET,
}

# ✅ Canonical job types (match SQL + worker code)
JOB_EXTRACT_TEXT = "extract_text"
JOB_EXTRACT_TABLES = "extract_tables"
JOB_EXTRACT_IMAGES = "extract_images"
JOB_TRANSCRIBE = "transcribe_audio"


class UploadOut(BaseModel):
    file_id: str
    status: str
    scan_result: str
    mime_type: str
    file_kind: str
    submission_id: Optional[str] = None
    enqueued_jobs: List[str] = Field(default_factory=list)


class FileStatusOut(BaseModel):
    id: str
    status: str
    scan_engine: Optional[str] = None
    scan_result: Optional[str] = None
    scanned_at: Optional[str] = None
    processed_at: Optional[str] = None
    submission_id: Optional[str] = None


class ExtractedOut(BaseModel):
    exText: List[dict] = Field(default_factory=list)
    exTables: List[dict] = Field(default_factory=list)
    exTranscript: List[dict] = Field(default_factory=list)


def sanitize_text(s: Optional[str]) -> str:
    return (s or "").replace("\x00", "")


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_filename(name: str) -> str:
    return (name or "upload.bin").replace("/", "_").replace("\\", "_")


def _require_supabase_config() -> None:
    if not settings.supabase_url:
        raise HTTPException(status_code=500, detail="SUPABASE_URL is not configured")
    if not settings.supabase_service_role_key:
        raise HTTPException(status_code=500, detail="SUPABASE_SERVICE_ROLE_KEY is not configured")
    if not settings.supabase_anon_key:
        raise HTTPException(status_code=500, detail="SUPABASE_ANON_KEY is not configured")
    if not settings.uploads_bucket:
        raise HTTPException(status_code=500, detail="UPLOADS_BUCKET is not configured")


def _service_headers(prefer_return: bool = True) -> Dict[str, str]:
    _require_supabase_config()
    return {
        "apikey": settings.supabase_service_role_key,
        "Authorization": f"Bearer {settings.supabase_service_role_key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation" if prefer_return else "return=minimal",
    }


def _user_rpc_headers(user: CurrentUser) -> Dict[str, str]:
    _require_supabase_config()
    token = getattr(user, "access_token", None)
    if not token:
        raise HTTPException(status_code=401, detail="Missing user access token")
    return {
        "apikey": settings.supabase_anon_key,
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def _storage_upload_headers(content_type: str) -> Dict[str, str]:
    _require_supabase_config()
    return {
        "apikey": settings.supabase_service_role_key,
        "Authorization": f"Bearer {settings.supabase_service_role_key}",
        "Content-Type": content_type or MIME_OCTET,
        "x-upsert": "true",
    }


def _guess_kind(sniffed_mime: str, filename: str) -> str:
    m = (sniffed_mime or "").lower()
    f = (filename or "").lower()

    if m in {MIME_PDF, MIME_DOCX, MIME_TEXT}:
        return "text"
    if m in {MIME_CSV, MIME_XLSX} or f.endswith(".csv") or f.endswith(".xlsx"):
        return "table"
    if m in MIME_IMAGES or f.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
        return "image"
    if m in MIME_AUDIO or f.endswith((".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".webm")):
        return "audio"
    if m in MIME_VIDEO or f.endswith((".mp4", ".mov", ".mkv", ".webm")):
        return "video"
    return "binary"


def _is_allowed(sniffed_mime: str, filename: str) -> bool:
    m = (sniffed_mime or "").lower()
    if m in ALLOWED_MIME and m != MIME_OCTET:
        return True

    if m == MIME_OCTET:
        f = (filename or "").lower()
        return any(
            f.endswith(ext)
            for ext in (
                ".pdf",
                ".docx",
                ".txt",
                ".csv",
                ".xlsx",
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
                ".gif",
                ".mp3",
                ".wav",
                ".m4a",
                ".aac",
                ".flac",
                ".ogg",
                ".webm",
                ".mp4",
                ".mov",
                ".mkv",
            )
        )
    return False


def _sniff_mime(buf: bytes, filename: str) -> str:
    sniffed = (magic.from_buffer(buf[:4096], mime=True) or "").lower().strip()
    f = (filename or "").lower()

    if f.endswith(".csv") and sniffed in {"text/plain", ""}:
        return MIME_CSV
    if f.endswith(".xlsx") and sniffed in {"application/zip", ""}:
        return MIME_XLSX
    if f.endswith(".docx") and sniffed in {"application/zip", ""}:
        return MIME_DOCX

    return sniffed or MIME_OCTET


def _validate_kind_size(file_kind: str, total_bytes: int) -> None:
    if file_kind == "audio":
        if total_bytes > int(getattr(settings, "max_audio_bytes", 30_000_000)):
            raise HTTPException(status_code=413, detail="Audio file too large")
    if file_kind == "video":
        if total_bytes > int(getattr(settings, "max_video_bytes", 50_000_000)):
            raise HTTPException(status_code=413, detail="Video file too large")


async def _postgrest_insert(table: str, row: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{settings.supabase_url.rstrip('/')}/rest/v1/{table}"
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers=_service_headers(prefer_return=True), json=row)
    if r.status_code >= 400:
        raise HTTPException(status_code=400, detail=f"Failed to insert into {table}: {r.text}")
    rows = r.json() or []
    if not rows:
        raise HTTPException(status_code=500, detail=f"Supabase did not return inserted row for {table}")
    return rows[0]


async def _postgrest_patch(table: str, where: str, patch: Dict[str, Any]) -> None:
    url = f"{settings.supabase_url.rstrip('/')}/rest/v1/{table}?{where}"
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.patch(url, headers=_service_headers(prefer_return=False), json=patch)
    if r.status_code >= 400:
        raise HTTPException(status_code=400, detail=f"Failed to update {table}: {r.text}")


async def _storage_upload_bytes(*, bucket: str, object_path: str, data: bytes, content_type: str) -> None:
    url = f"{settings.supabase_url.rstrip('/')}/storage/v1/object/{bucket}/{object_path}"
    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(url, headers=_storage_upload_headers(content_type), content=data)
    if r.status_code >= 300:
        raise HTTPException(status_code=500, detail=f"Storage upload failed: {r.status_code} {r.text}")


async def _enqueue_ingestion_job(*, file_id: str, job_type: str, user: CurrentUser) -> str:
    """
    Calls SQL RPC: create_ingestion_job(p_file_id, p_job_type)
    Must be called with USER token so auth.uid() works.
    """
    url = f"{settings.supabase_url.rstrip('/')}/rest/v1/rpc/create_ingestion_job"
    payload: Dict[str, Any] = {"p_file_id": file_id, "p_job_type": job_type}

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers=_user_rpc_headers(user), json=payload)

    if r.status_code >= 300:
        raise HTTPException(status_code=400, detail=f"enqueue failed: {r.status_code} {r.text}")

    job_id = r.json()
    if isinstance(job_id, dict) and "job_id" in job_id:
        job_id = job_id["job_id"]

    jid = str(job_id)
    logger.info("Enqueued job: file_id=%s job_type=%s job_id=%s", file_id, job_type, jid)
    return jid


def _jobs_for_kind(file_kind: str) -> List[str]:
    if file_kind == "text":
        return [JOB_EXTRACT_TEXT]
    if file_kind == "table":
        return [JOB_EXTRACT_TABLES]
    if file_kind == "image":
        return [JOB_EXTRACT_TEXT, JOB_EXTRACT_IMAGES]
    if file_kind in ("audio", "video"):
        return [JOB_TRANSCRIBE]
    return []


@router.post("/upload", response_model=UploadOut)
@limiter.limit("10/minute")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    user: CurrentUser = Depends(get_current_user),
):
    _require_supabase_config()

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    max_bytes = int(settings.max_upload_bytes)

    h = hashlib.sha256()
    total = 0
    buf = bytearray()

    while True:
        chunk = await file.read(READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail="File too large")
        h.update(chunk)
        buf.extend(chunk)

    if total == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    sha256 = h.hexdigest()
    safe_name = _safe_filename(file.filename)

    sniffed = _sniff_mime(bytes(buf), safe_name)
    if not _is_allowed(sniffed, safe_name):
        raise HTTPException(status_code=400, detail=f"File type not allowed: {sniffed}")

    file_kind = _guess_kind(sniffed, safe_name)

    # ✅ enforce per-kind size limits too
    _validate_kind_size(file_kind, total)

    file_id = str(uuid.uuid4())
    bucket = settings.uploads_bucket
    object_path = f"submissions/{user.id}/{file_id}/{safe_name}"

    await _postgrest_insert(
        "files",
        {
            "id": file_id,
            "user_id": user.id,
            "bucket": bucket,
            "object_path": object_path,
            "mime_type": sniffed,
            "size_bytes": total,
            "sha256": sha256,
            "created_at": _utc_iso(),
            "status": "uploaded",
        },
    )

    await _storage_upload_bytes(bucket=bucket, object_path=object_path, data=bytes(buf), content_type=sniffed)

    # ---- AV scan ----
    await _postgrest_patch("files", f"id=eq.{file_id}&user_id=eq.{user.id}", {"status": "scanning"})
    is_clean, scan_raw = await clamav_scan_bytes(bytes(buf))
    scan_raw = sanitize_text(scan_raw)

    scan_result = "clean" if is_clean else "infected"
    await _postgrest_patch(
        "files",
        f"id=eq.{file_id}&user_id=eq.{user.id}",
        {
            "scan_engine": "clamav",
            "scan_result": scan_result,
            "scanned_at": _utc_iso(),
            "status": "clean" if is_clean else "quarantined",
        },
    )

    if not is_clean:
        return UploadOut(
            file_id=file_id,
            status="quarantined",
            scan_result=scan_result,
            mime_type=sniffed,
            file_kind=file_kind,
            submission_id=None,
            enqueued_jobs=[],
        )

    await _postgrest_patch("files", f"id=eq.{file_id}&user_id=eq.{user.id}", {"status": "clean"})

    # ---- enqueue jobs ----
    jobs = _jobs_for_kind(file_kind)
    enqueued: List[str] = []
    for jt in jobs:
        await _enqueue_ingestion_job(file_id=file_id, job_type=jt, user=user)
        enqueued.append(jt)

    return UploadOut(
        file_id=file_id,
        status="clean",
        scan_result="clean",
        mime_type=sniffed,
        file_kind=file_kind,
        submission_id=None,
        enqueued_jobs=enqueued,
    )


@router.get("/{file_id}/status", response_model=FileStatusOut)
async def get_file_status(file_id: str, user: CurrentUser = Depends(get_current_user)):
    _require_supabase_config()

    if user.role == "admin":
        where = f"id=eq.{file_id}&select=id,status,scan_engine,scan_result,scanned_at,processed_at,submission_id"
    else:
        where = f"id=eq.{file_id}&user_id=eq.{user.id}&select=id,status,scan_engine,scan_result,scanned_at,processed_at,submission_id"

    url = f"{settings.supabase_url.rstrip('/')}/rest/v1/files?{where}"

    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, headers=_service_headers(prefer_return=False))

    if r.status_code >= 400:
        raise HTTPException(status_code=400, detail=f"Failed to fetch file: {r.text}")

    rows = r.json() or []
    if not rows:
        raise HTTPException(status_code=404, detail="Not found")

    return rows[0]


@router.get("/{file_id}/extracted", response_model=ExtractedOut)
async def get_extracted(file_id: str, user: CurrentUser = Depends(get_current_user)):
    _require_supabase_config()

    # ownership gate (or admin)
    if user.role == "admin":
        where_file = f"id=eq.{file_id}&select=id"
    else:
        where_file = f"id=eq.{file_id}&user_id=eq.{user.id}&select=id"

    base = f"{settings.supabase_url.rstrip('/')}/rest/v1"
    headers = _service_headers(prefer_return=False)

    async with httpx.AsyncClient(timeout=10) as client:
        fr = await client.get(f"{base}/files?{where_file}", headers=headers)
        if fr.status_code >= 400:
            raise HTTPException(status_code=400, detail=f"Failed to fetch file: {fr.text}")
        if not (fr.json() or []):
            raise HTTPException(status_code=404, detail="Not found")

        # return the latest extracted payloads (real content)
        t = await client.get(
            f"{base}/extracted_text?file_id=eq.{file_id}&select=*&order=created_at.desc&limit=1",
            headers=headers,
        )
        tb = await client.get(
            f"{base}/extracted_tables?file_id=eq.{file_id}&select=*&order=created_at.desc&limit=1",
            headers=headers,
        )
        tr = await client.get(
            f"{base}/transcripts?file_id=eq.{file_id}&select=*&order=created_at.desc&limit=1",
            headers=headers,
        )

    return ExtractedOut(
        exText=t.json() if t.status_code == 200 else [],
        exTables=tb.json() if tb.status_code == 200 else [],
        exTranscript=tr.json() if tr.status_code == 200 else [],
    )