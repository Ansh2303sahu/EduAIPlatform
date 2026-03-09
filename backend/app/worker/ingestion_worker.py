# backend/app/worker/ingestion_worker.py
from __future__ import annotations

import asyncio
import base64
import os
from datetime import datetime, timezone
from typing import Any, Dict

import httpx

from app.core.config import settings
from app.services.ingestion_repo import IngestionRepo
from app.services.files_repo import FilesRepo
from app.services.storage import create_signed_download_url, upload_bytes_to_storage
from app.services.processing_events_repo import ProcessingEventsRepo
from app.services.extracted_repo import ExtractedRepo
from app.services.extracted_tables_repo import ExtractedTablesRepo
from app.services.extracted_media_repo import ExtractedMediaRepo
from app.services.transcripts_repo import TranscriptsRepo
from app.services.redaction import redact_pii
from app.services.normalize import clamp_text, sanitize_pg_text, normalize_table

WORKER_ID = os.getenv("WORKER_ID", "ingestion-worker-1")

POLL_SECONDS = int(os.getenv("INGESTION_POLL_SECONDS", "2"))
MAX_ATTEMPTS = int(os.getenv("INGESTION_MAX_ATTEMPTS", "3"))
LOCK_TIMEOUT_SECONDS = int(os.getenv("INGESTION_LOCK_TIMEOUT_SECONDS", "600"))

MAX_TEXT_CHARS = int(os.getenv("WORKER_MAX_TEXT_CHARS", "200000"))
MAX_TABLES = int(os.getenv("WORKER_MAX_TABLES", "25"))

MAX_ROWS_PER_TABLE = int(os.getenv("WORKER_MAX_ROWS_PER_TABLE", "5000"))
MAX_COLS_PER_TABLE = int(os.getenv("WORKER_MAX_COLS_PER_TABLE", "50"))
MAX_CELL_CHARS = int(os.getenv("WORKER_MAX_CELL_CHARS", "2000"))

MAX_IMAGES = int(os.getenv("WORKER_MAX_IMAGES", "50"))

MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", "30000000"))  # 30MB
MAX_VIDEO_BYTES = int(os.getenv("MAX_VIDEO_BYTES", "50000000"))  # 50MB

WORKER_MODE = os.getenv("WORKER_MODE", "all").lower().strip()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _is_quarantined(quarantined_until: Any) -> bool:
    if quarantined_until is None:
        return False
    if isinstance(quarantined_until, str):
        dt = datetime.fromisoformat(quarantined_until.replace("Z", "+00:00"))
    elif isinstance(quarantined_until, datetime):
        dt = quarantined_until
    else:
        return True
    return dt.astimezone(timezone.utc) > _utc_now()


def _pick_storage_location(file_row: Dict[str, Any]) -> tuple[str, str]:
    bucket = file_row.get("storage_bucket") or file_row.get("bucket")
    path = file_row.get("storage_path") or file_row.get("object_path")
    if not bucket or not path:
        raise RuntimeError(f"Invalid storage fields: bucket={bucket}, path={path}")
    return str(bucket), str(path)


def _pick_filename(file_row: Dict[str, Any]) -> str:
    return (
        file_row.get("original_name")
        or file_row.get("filename")
        or file_row.get("name")
        or (file_row.get("object_path", "") or "").split("/")[-1]
        or "upload.bin"
    )


def _parser_base() -> str:
    base = (settings.parser_url or "").strip().rstrip("/")
    if not base:
        raise RuntimeError("PARSER_URL is not configured")
    if base.endswith("/parse"):
        return base
    return f"{base}/parse"


def _guess_image_content_type(filename: str | None) -> str:
    f = (filename or "").lower()
    if f.endswith(".png"):
        return "image/png"
    if f.endswith(".jpg") or f.endswith(".jpeg"):
        return "image/jpeg"
    if f.endswith(".webp"):
        return "image/webp"
    if f.endswith(".gif"):
        return "image/gif"
    return "application/octet-stream"


def _is_table_mime(mime: str | None, filename: str | None) -> bool:
    m = (mime or "").lower()
    f = (filename or "").lower()
    return (
        "text/csv" in m
        or "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" in m
        or f.endswith(".csv")
        or f.endswith(".xlsx")
    )


def _is_image_mime(mime: str | None, filename: str | None) -> bool:
    m = (mime or "").lower()
    f = (filename or "").lower()
    return (
        m.startswith("image/")
        or f.endswith(".png")
        or f.endswith(".jpg")
        or f.endswith(".jpeg")
        or f.endswith(".webp")
        or f.endswith(".gif")
    )


def _is_audio_or_video(mime: str | None, filename: str | None) -> str | None:
    m = (mime or "").lower()
    f = (filename or "").lower()

    if m.startswith("audio/") or f.endswith((".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".webm")):
        return "audio"
    if m.startswith("video/") or f.endswith((".mp4", ".mov", ".mkv", ".webm")):
        return "video"
    return None


def _mode_allows(kind: str) -> bool:
    if WORKER_MODE == "all":
        return True
    if WORKER_MODE == "light":
        return kind not in ("audio", "video")
    if WORKER_MODE == "heavy":
        return kind in ("audio", "video")
    return True


def _require_supabase_config() -> None:
    if not settings.supabase_url:
        raise RuntimeError("SUPABASE_URL is not configured")
    if not settings.supabase_service_role_key:
        raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY is not configured")


def _service_headers(prefer_return: bool = False) -> dict[str, str]:
    _require_supabase_config()
    key = settings.supabase_service_role_key
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation" if prefer_return else "return=minimal",
    }


async def _mark_file_processed(
    client: httpx.AsyncClient,
    *,
    file_id: str,
    processed_at: datetime,
) -> None:
    """
    ✅ Critical: unblocks frontend Extract step (processed_at != null).
    Also sets status='processed' (if your schema has status).
    """
    url = f"{settings.supabase_url.rstrip('/')}/rest/v1/files?id=eq.{file_id}"
    payload = {
        "processed_at": processed_at.isoformat(),
        "status": "processed",
    }
    r = await client.patch(url, headers=_service_headers(prefer_return=False), json=payload)
    if r.status_code >= 400:
        raise RuntimeError(f"Failed to update files.processed_at: {r.status_code} {r.text}")


async def run_loop() -> None:
    if not settings.supabase_service_role_key:
        raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY is not configured")
    if not settings.parser_secret:
        raise RuntimeError("PARSER_SECRET is not configured")
    if not settings.signed_url_expires_seconds:
        raise RuntimeError("SIGNED_URL_EXPIRES_SECONDS is not configured")

    repo = IngestionRepo(service_role_key=settings.supabase_service_role_key)
    files_repo = FilesRepo()

    events_repo = ProcessingEventsRepo()
    extracted_text_repo = ExtractedRepo()
    extracted_tables_repo = ExtractedTablesRepo()
    extracted_media_repo = ExtractedMediaRepo()
    transcripts_repo = TranscriptsRepo()

    parser_base = _parser_base()
    parser_secret = settings.parser_secret

    async with httpx.AsyncClient(timeout=120) as client:
        while True:
            job = await repo.claim_next_job(
                worker_id=WORKER_ID,
                max_attempts=MAX_ATTEMPTS,
                lock_timeout_seconds=LOCK_TIMEOUT_SECONDS,
            )

            if not job:
                await asyncio.sleep(POLL_SECONDS)
                continue

            job_id = str(job.get("job_id"))
            file_id = str(job.get("file_id"))
            job_type = (job.get("job_type") or "full").lower()

            try:
                file_row = await files_repo.get_file_record_service(file_id)
                if not file_row:
                    await repo.mark_failed(job_id=job_id, worker_id=WORKER_ID, error_code="FILE_NOT_FOUND", error_message="File not found")
                    continue

                user_id = str(file_row["user_id"])
                submission_id = str(file_row.get("submission_id")) if file_row.get("submission_id") else None
                mime_type = file_row.get("mime_type")
                filename = _pick_filename(file_row)
                source_sha256 = str(file_row.get("sha256") or "")

                if file_row.get("scan_result") != "clean":
                    await repo.mark_failed(job_id=job_id, worker_id=WORKER_ID, error_code="SCAN_NOT_CLEAN", error_message=f"scan_result={file_row.get('scan_result')}")
                    continue

                if _is_quarantined(file_row.get("quarantined_until")):
                    await repo.mark_failed(job_id=job_id, worker_id=WORKER_ID, error_code="STILL_QUARANTINED", error_message="File quarantine not expired")
                    continue

                bucket, path = _pick_storage_location(file_row)
                signed_url = await create_signed_download_url(bucket=bucket, path=path, expires_in=settings.signed_url_expires_seconds)

                await events_repo.log(user_id=user_id, job_id=job_id, file_id=file_id, event_type="FILE_DOWNLOAD_START", details={"bucket": bucket, "path": path})

                resp = await client.get(signed_url)
                resp.raise_for_status()
                file_bytes = resp.content

                await events_repo.log(user_id=user_id, job_id=job_id, file_id=file_id, event_type="FILE_DOWNLOADED", details={"bytes": len(file_bytes), "mime_type": mime_type, "filename": filename})

                media_kind = _is_audio_or_video(mime_type, filename)
                if media_kind in ("audio", "video") and not _mode_allows(media_kind):
                    await repo.mark_failed(job_id=job_id, worker_id=WORKER_ID, error_code="HEAVY_WORKER_REQUIRED", error_message=f"Worker mode '{WORKER_MODE}' does not process {media_kind}.")
                    continue

                # ---------------- TEXT / OCR ----------------
                if job_type in ("full", "extract_text"):
                    if _is_image_mime(mime_type, filename):
                        oresp = await client.post(
                            f"{parser_base}/ocr",
                            headers={"X-Parser-Secret": parser_secret},
                            files={"file": (filename, file_bytes)},
                        )
                        oresp.raise_for_status()
                        ojson = oresp.json() if oresp.content else {}

                        raw_text = sanitize_pg_text((ojson.get("text") or "") if isinstance(ojson, dict) else "").strip()
                        if not raw_text:
                            raw_text = f"[Image uploaded: {filename}] (OCR found no readable text)"
                        raw_text, truncated = clamp_text(raw_text, max_chars=MAX_TEXT_CHARS)

                        await events_repo.log(
                            user_id=user_id,
                            job_id=job_id,
                            file_id=file_id,
                            event_type="OCR_EXTRACTED",
                            details={"chars": len(raw_text), "truncated": truncated, "parser_meta": ojson.get("meta", {}) if isinstance(ojson, dict) else {}},
                        )

                        red = redact_pii(raw_text)
                        await extracted_text_repo.insert_extracted_text(
                            user_id=user_id,
                            submission_id=submission_id,
                            file_id=file_id,
                            job_id=job_id,
                            source_sha256=source_sha256,
                            redacted_text=red.redacted_text,
                            redaction_summary=red.summary,
                        )

                        derived_bucket = getattr(settings, "uploads_bucket", None) or bucket
                        derived_path = f"derived/{file_id}/0_{filename}"
                        await upload_bytes_to_storage(bucket=derived_bucket, path=derived_path, data=file_bytes, content_type=_guess_image_content_type(filename))

                        await extracted_media_repo.insert_media_metadata(
                            user_id=user_id,
                            submission_id=submission_id,
                            file_id=file_id,
                            job_id=job_id,
                            source_sha256=source_sha256,
                            media_index=0,
                            media_type="image",
                            width=None,
                            height=None,
                            perceptual_hash=None,
                            caption=None,
                            metadata={"derived_bucket": derived_bucket, "derived_path": derived_path, "bytes": len(file_bytes), "note": "original image stored"},
                        )
                    else:
                        presp = await client.post(
                            f"{parser_base}/text",
                            headers={"X-Parser-Secret": parser_secret},
                            files={"file": (filename, file_bytes)},
                        )
                        presp.raise_for_status()
                        parsed = presp.json() if presp.content else {}

                        raw_text = sanitize_pg_text((parsed.get("text") or "") if isinstance(parsed, dict) else "")
                        raw_text, truncated = clamp_text(raw_text, max_chars=MAX_TEXT_CHARS)

                        await events_repo.log(
                            user_id=user_id,
                            job_id=job_id,
                            file_id=file_id,
                            event_type="TEXT_EXTRACTED",
                            details={"chars": len(raw_text), "truncated": truncated, "parser_meta": parsed.get("meta", {}) if isinstance(parsed, dict) else {}},
                        )

                        red = redact_pii(raw_text)
                        await extracted_text_repo.insert_extracted_text(
                            user_id=user_id,
                            submission_id=submission_id,
                            file_id=file_id,
                            job_id=job_id,
                            source_sha256=source_sha256,
                            redacted_text=red.redacted_text,
                            redaction_summary=red.summary,
                        )

                # ---------------- TABLES ----------------
                if job_type in ("full", "parse_tables") and _is_table_mime(mime_type, filename):
                    t_resp = await client.post(
                        f"{parser_base}/tables",
                        headers={"X-Parser-Secret": parser_secret},
                        files={"file": (filename, file_bytes)},
                    )
                    t_resp.raise_for_status()
                    tdata = t_resp.json() if t_resp.content else {}
                    tables = (tdata.get("tables", []) if isinstance(tdata, dict) else [])[:MAX_TABLES]

                    stored_tables = 0
                    for t in tables:
                        cols, rows = normalize_table(
                            columns=t.get("columns", []),
                            rows=t.get("rows", []),
                            max_cols=MAX_COLS_PER_TABLE,
                            max_rows=MAX_ROWS_PER_TABLE,
                            max_cell_chars=MAX_CELL_CHARS,
                        )
                        await extracted_tables_repo.insert_table(
                            user_id=user_id,
                            submission_id=submission_id,
                            file_id=file_id,
                            job_id=job_id,
                            source_sha256=source_sha256,
                            table_index=int(t.get("table_index", stored_tables)),
                            sheet_name=t.get("sheet_name"),
                            columns=cols,
                            rows=rows,
                        )
                        stored_tables += 1

                # ---------------- IMAGES (embedded) ----------------
                if job_type in ("full", "extract_images"):
                    img_resp = await client.post(
                        f"{parser_base}/images",
                        headers={"X-Parser-Secret": parser_secret},
                        files={"file": (filename, file_bytes)},
                    )
                    img_resp.raise_for_status()
                    idata = img_resp.json() if img_resp.content else {}
                    images = (idata.get("images", []) if isinstance(idata, dict) else [])[:MAX_IMAGES]

                    derived_bucket = getattr(settings, "uploads_bucket", None) or bucket
                    stored = 0

                    for im in images:
                        try:
                            b64 = im.get("b64")
                            if not b64:
                                continue

                            img_bytes = base64.b64decode(b64)
                            img_name = im.get("name") or f"image_{stored}.bin"
                            derived_path = f"derived/{file_id}/{stored}_{img_name}"

                            await upload_bytes_to_storage(bucket=derived_bucket, path=derived_path, data=img_bytes, content_type=_guess_image_content_type(img_name))

                            meta = im.get("meta", {}) if isinstance(im.get("meta"), dict) else {}
                            await extracted_media_repo.insert_media_metadata(
                                user_id=user_id,
                                submission_id=submission_id,
                                file_id=file_id,
                                job_id=job_id,
                                source_sha256=source_sha256,
                                media_index=int(im.get("index", stored)),
                                media_type="image",
                                width=meta.get("width"),
                                height=meta.get("height"),
                                perceptual_hash=None,
                                caption=None,
                                metadata={"derived_bucket": derived_bucket, "derived_path": derived_path},
                            )
                            stored += 1
                        except Exception:
                            continue

                # ---------------- TRANSCRIBE ----------------
                if job_type in ("full", "transcribe_audio") and media_kind in ("audio", "video"):
                    if media_kind == "audio" and len(file_bytes) > MAX_AUDIO_BYTES:
                        raise RuntimeError("Audio too large for transcription")
                    if media_kind == "video" and len(file_bytes) > MAX_VIDEO_BYTES:
                        raise RuntimeError("Video too large for transcription")

                    await events_repo.log(user_id=user_id, job_id=job_id, file_id=file_id, event_type="TRANSCRIBE_START", details={"kind": media_kind, "filename": filename})

                    tresp = await client.post(
                        f"{parser_base}/transcribe",
                        headers={"X-Parser-Secret": parser_secret},
                        files={"file": (filename, file_bytes)},
                    )
                    tresp.raise_for_status()
                    tparsed = tresp.json() if tresp.content else {}

                    raw_transcript = sanitize_pg_text((tparsed.get("text") or "") if isinstance(tparsed, dict) else "").strip()
                    if not raw_transcript:
                        raw_transcript = f"[{media_kind.upper()} uploaded: {filename}] (transcription produced no text)"
                    raw_transcript, truncated = clamp_text(raw_transcript, max_chars=MAX_TEXT_CHARS)

                    red = redact_pii(raw_transcript)
                    meta = (tparsed.get("meta") or {}) if isinstance(tparsed, dict) else {}

                    await transcripts_repo.insert_transcript(
                        user_id=user_id,
                        submission_id=submission_id,
                        file_id=file_id,
                        job_id=job_id,
                        source_sha256=source_sha256,
                        redacted_transcript=red.redacted_text,
                        redaction_summary=red.summary,
                        model_name=str(meta.get("model") or "whisper"),
                        model_version=None,
                        confidence={
                            "language": meta.get("language"),
                            "language_probability": meta.get("language_probability"),
                            "segments_count": meta.get("segments_count"),
                            "truncated": truncated,
                            "engine": meta.get("engine"),
                            "device": meta.get("device"),
                        },
                    )

                # ✅ CRITICAL: unblock frontend "Extract"
                done_at = _utc_now()
                await _mark_file_processed(client, file_id=file_id, processed_at=done_at)

                await events_repo.log(
                    user_id=user_id,
                    job_id=job_id,
                    file_id=file_id,
                    event_type="FILE_PROCESSING_DONE",
                    details={"processed_at": done_at.isoformat(), "job_type": job_type, "mime_type": mime_type},
                )

                await repo.mark_done(job_id=job_id, worker_id=WORKER_ID, details={"job_type": job_type, "mime_type": mime_type})

            except httpx.HTTPError as e:
                await repo.mark_failed(job_id=job_id, worker_id=WORKER_ID, error_code="HTTP_ERROR", error_message=str(e))
            except Exception as e:
                await repo.mark_failed(job_id=job_id, worker_id=WORKER_ID, error_code="WORKER_ERROR", error_message=str(e))
            finally:
                await asyncio.sleep(POLL_SECONDS)