# app/worker/prof_ingestion_worker.py
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import httpx

from app.core.config import settings
from app.services.files_repo import FilesRepo
from app.services.storage import create_signed_download_url
from app.services.redaction import redact_pii
from app.services.normalize import clamp_text, sanitize_pg_text

from app.services.media_utils import (
    MediaTooLong,
    ffprobe_duration_seconds,
    enforce_media_duration,
    extract_audio_from_video,
    segment_audio_to_wavs,
)
from app.services.whisper_local import transcribe_with_faster_whisper

log = logging.getLogger("prof-worker")

WORKER_ID = os.getenv("PROF_WORKER_ID", "prof-worker-1")
POLL_SECONDS = int(os.getenv("PROF_INGESTION_POLL_SECONDS", "2"))
MAX_ATTEMPTS = int(os.getenv("PROF_INGESTION_MAX_ATTEMPTS", "3"))
LOCK_TIMEOUT_SECONDS = int(os.getenv("PROF_INGESTION_LOCK_TIMEOUT_SECONDS", "600"))
MAX_TEXT_CHARS = int(os.getenv("PROF_WORKER_MAX_TEXT_CHARS", "200000"))

MAX_AUDIO_BYTES = int(os.getenv("PROF_MAX_AUDIO_BYTES", "30000000"))
MAX_VIDEO_BYTES = int(os.getenv("PROF_MAX_VIDEO_BYTES", "50000000"))

# Phase 5: hard cap guardrails
MAX_AUDIO_SECONDS = int(os.getenv("PROF_MAX_AUDIO_SECONDS", "600"))  # 10 min
MAX_VIDEO_SECONDS = int(os.getenv("PROF_MAX_VIDEO_SECONDS", "600"))  # 10 min

# Phase 5: segmentation threshold (chunking)
SEGMENT_SECONDS = int(os.getenv("PROF_AUDIO_SEGMENT_SECONDS", "300"))  # 5 min

WORKER_MODE = os.getenv("PROF_WORKER_MODE", "all").lower().strip()

# Increased connect timeout + sane pool timeout to avoid transient ConnectTimeout crashes
HTTP_TIMEOUT = httpx.Timeout(connect=30.0, read=120.0, write=60.0, pool=30.0)

# Retry policy for transient network hiccups
MAX_HTTP_RETRIES = int(os.getenv("PROF_HTTP_RETRIES", "5"))
RETRY_BASE_SLEEP = float(os.getenv("PROF_HTTP_RETRY_BASE_SLEEP", "1.5"))

# --- file type groups ---
TEXT_EXT = {".pdf", ".docx", ".txt", ".md"}
TABLE_EXT = {".csv", ".xlsx"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
VIDEO_EXT = {".webm", ".mp4", ".mov", ".mkv"}


def _parser_base() -> str:
    base = (settings.parser_url or "").strip().rstrip("/")
    if not base:
        raise RuntimeError("PARSER_URL is not configured")
    return base


def _parser_text_url() -> str:
    return f"{_parser_base()}/parse/text"


def _parser_tables_url() -> str:
    return f"{_parser_base()}/parse/tables"


def _parser_ocr_url() -> str:
    return f"{_parser_base()}/parse/ocr"


def _pick_storage_location(file_row: Dict[str, Any]) -> Tuple[str, str]:
    bucket = file_row.get("bucket")
    path = file_row.get("object_path")
    if not bucket or not path:
        raise RuntimeError("Missing bucket/object_path on files row")
    return str(bucket), str(path)


def _pick_filename(file_row: Dict[str, Any]) -> str:
    return (
        file_row.get("original_name")
        or (file_row.get("object_path", "") or "").split("/")[-1]
        or "upload.bin"
    )


def _ext_of(name: str) -> str:
    name = (name or "").lower().strip()
    if "." not in name:
        return ""
    return "." + name.rsplit(".", 1)[-1]


def _classify_file(filename: str, mime_type: Optional[str]) -> str:
    ext = _ext_of(filename)
    if ext in TEXT_EXT:
        return "text"
    if ext in TABLE_EXT:
        return "table"
    if ext in IMAGE_EXT:
        return "image"
    if ext in AUDIO_EXT:
        return "audio"
    if ext in VIDEO_EXT:
        return "video"

    mt = (mime_type or "").lower()
    if mt.startswith("image/"):
        return "image"
    if mt.startswith("audio/"):
        return "audio"
    if mt.startswith("video/"):
        return "video"
    return "other"


def _mode_allows(kind: str) -> bool:
    if WORKER_MODE == "all":
        return True
    if WORKER_MODE == "light":
        return kind not in ("audio", "video")
    if WORKER_MODE == "heavy":
        return kind in ("audio", "video")
    return True


def _tables_to_text(tables_payload: dict) -> str:
    tables = (tables_payload or {}).get("tables") or []
    out_lines: list[str] = []
    for t in tables[:25]:
        sheet = t.get("sheet_name") or "sheet"
        cols = t.get("columns") or []
        rows = t.get("rows") or []
        out_lines.append(f"=== TABLE: {sheet} ===")
        if cols:
            out_lines.append(" | ".join(str(c) for c in cols))
        for r in rows[:200]:
            if isinstance(r, list):
                out_lines.append(" | ".join(str(x) for x in r))
            else:
                out_lines.append(str(r))
        out_lines.append("")
    return "\n".join(out_lines).strip()


async def _post_json_with_retry(
    client: httpx.AsyncClient,
    *,
    url: str,
    headers: Dict[str, str],
    json_payload: Dict[str, Any],
) -> httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(MAX_HTTP_RETRIES):
        try:
            r = await client.post(url, headers=headers, json=json_payload)
            # Retry on transient server errors
            if r.status_code in (502, 503, 504):
                wait = RETRY_BASE_SLEEP * (attempt + 1)
                log.warning("POST %s -> %s (retry in %.1fs)", url, r.status_code, wait)
                await asyncio.sleep(wait)
                continue
            return r
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError) as e:
            last_exc = e
            wait = RETRY_BASE_SLEEP * (attempt + 1)
            log.warning("POST %s timeout/network (%s). retry in %.1fs", url, repr(e), wait)
            await asyncio.sleep(wait)
    raise RuntimeError(f"POST failed after retries: {url}") from last_exc


async def _get_with_retry(client: httpx.AsyncClient, *, url: str) -> httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(MAX_HTTP_RETRIES):
        try:
            r = await client.get(url)
            if r.status_code in (502, 503, 504):
                wait = RETRY_BASE_SLEEP * (attempt + 1)
                log.warning("GET %s -> %s (retry in %.1fs)", url, r.status_code, wait)
                await asyncio.sleep(wait)
                continue
            return r
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError) as e:
            last_exc = e
            wait = RETRY_BASE_SLEEP * (attempt + 1)
            log.warning("GET %s timeout/network (%s). retry in %.1fs", url, repr(e), wait)
            await asyncio.sleep(wait)
    raise RuntimeError(f"GET failed after retries: {url}") from last_exc


class ProfIngestionRepo:
    def __init__(self, client: httpx.AsyncClient) -> None:
        if not settings.supabase_url:
            raise RuntimeError("SUPABASE_URL is not configured")
        if not settings.supabase_service_role_key:
            raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY is not configured")
        self.base = settings.supabase_url.rstrip("/")
        self.key = settings.supabase_service_role_key
        self.client = client

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.key}",
            "apikey": self.key,
            "Content-Type": "application/json",
        }

    async def claim_next_job(self) -> Optional[Dict[str, Any]]:
        url = f"{self.base}/rest/v1/rpc/claim_next_prof_job"
        payload = {
            "p_worker_id": WORKER_ID,
            "p_max_attempts": MAX_ATTEMPTS,
            "p_lock_timeout_seconds": LOCK_TIMEOUT_SECONDS,
        }
        r = await _post_json_with_retry(self.client, url=url, headers=self._headers(), json_payload=payload)
        if r.status_code >= 300:
            raise RuntimeError(f"claim_next_prof_job failed: {r.status_code} {r.text}")
        rows = r.json() or []
        return rows[0] if rows else None

    async def mark_done(self, job_id: str, details: dict | None = None) -> None:
        url = f"{self.base}/rest/v1/rpc/mark_prof_job_done"
        payload = {"p_job_id": job_id, "p_worker_id": WORKER_ID, "p_details": details or {}}
        r = await _post_json_with_retry(self.client, url=url, headers=self._headers(), json_payload=payload)
        if r.status_code >= 300:
            raise RuntimeError(f"mark_prof_job_done failed: {r.status_code} {r.text}")

    async def mark_failed(self, job_id: str, code: str, msg: str, details: dict | None = None) -> None:
        url = f"{self.base}/rest/v1/rpc/mark_prof_job_failed"
        payload = {
            "p_job_id": job_id,
            "p_worker_id": WORKER_ID,
            "p_error_code": code,
            "p_error_message": msg,
            "p_details": details or {},
        }
        r = await _post_json_with_retry(self.client, url=url, headers=self._headers(), json_payload=payload)
        if r.status_code >= 300:
            raise RuntimeError(f"mark_prof_job_failed failed: {r.status_code} {r.text}")

    async def log_event(self, user_id: str, job_id: str, file_id: str, event_type: str, details: dict) -> None:
        url = f"{self.base}/rest/v1/prof_events"
        row = {
            "user_id": user_id,
            "job_id": job_id,
            "file_id": file_id,
            "event_type": event_type,
            "details": details,
        }
        headers = {**self._headers(), "Prefer": "return=minimal"}
        r = await _post_json_with_retry(self.client, url=url, headers=headers, json_payload=row)
        if r.status_code >= 300:
            log.warning("event insert failed: %s %s", r.status_code, r.text)

    async def insert_prof_insight_adaptive(
        self,
        *,
        user_id: str,
        job_id: str,
        file_id: str,
        source_sha256: str,
        redacted_text: str,
        redaction_summary: dict,
        extra_meta: dict | None = None,
    ) -> None:
        url = f"{self.base}/rest/v1/prof_insights"

        text_cols = ["redacted_text", "text", "content"]
        summary_cols = ["redaction_summary", "summary", "meta"]

        base_row_common = {
            "user_id": user_id,
            "job_id": job_id,
            "file_id": file_id,
            "source_sha256": source_sha256,
        }

        meta_payload = {"redaction": redaction_summary}
        if extra_meta:
            meta_payload.update(extra_meta)

        last_err: tuple[int, str] | None = None
        for tcol in text_cols:
            for scol in summary_cols:
                row = dict(base_row_common)
                row[tcol] = redacted_text
                row[scol] = redaction_summary if scol != "meta" else meta_payload

                headers = {**self._headers(), "Prefer": "return=minimal"}
                r = await _post_json_with_retry(self.client, url=url, headers=headers, json_payload=row)

                if r.status_code < 300:
                    return

                txt = (r.text or "").lower()
                if "does not exist" in txt or "could not find" in txt:
                    last_err = (r.status_code, r.text)
                    continue

                raise RuntimeError(f"prof_insights insert failed: {r.status_code} {r.text}")

        raise RuntimeError(f"prof_insights insert failed (schema mismatch): {last_err}")


async def _transcribe_prof(
    repo: ProfIngestionRepo,
    *,
    user_id: str,
    job_id: str,
    file_id: str,
    filename: str,
    file_bytes: bytes,
    kind: str,
) -> dict:
    if kind == "audio" and len(file_bytes) > MAX_AUDIO_BYTES:
        raise RuntimeError("Audio too large")
    if kind == "video" and len(file_bytes) > MAX_VIDEO_BYTES:
        raise RuntimeError("Video too large")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        src = td_path / filename
        src.write_bytes(file_bytes)

        # Duration check + guardrail
        duration = ffprobe_duration_seconds(src)
        max_sec = MAX_AUDIO_SECONDS if kind == "audio" else MAX_VIDEO_SECONDS
        await repo.log_event(
            user_id,
            job_id,
            file_id,
            "FFPROBE_DURATION",
            {"seconds": duration, "max_seconds": max_sec, "kind": kind},
        )

        # Hard cap reject (protect compute)
        enforce_media_duration(src, kind=kind, max_seconds=max_sec)

        # Convert/extract to wav
        wav = td_path / "audio.wav"
        if kind == "video":
            await repo.log_event(user_id, job_id, file_id, "AUDIO_EXTRACT_START", {"from": "video"})
            extract_audio_from_video(src, wav)
            await repo.log_event(user_id, job_id, file_id, "AUDIO_EXTRACT_DONE", {"wav": str(wav.name)})
        else:
            # convert audio->wav using ffmpeg
            from subprocess import run

            cmd = ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", str(wav)]
            p = run(cmd, capture_output=True, text=True)
            if p.returncode != 0:
                raise RuntimeError(f"ffmpeg audio convert failed: {p.stderr.strip()}")

        model = os.getenv("PROF_WHISPER_MODEL", "base")

        # Segment if long (but still <= max)
        if duration > float(SEGMENT_SECONDS):
            await repo.log_event(user_id, job_id, file_id, "WHISPER_SEGMENT_START", {"segment_seconds": SEGMENT_SECONDS})
            seg_dir = td_path / "segs"
            chunks = segment_audio_to_wavs(wav, seg_dir, SEGMENT_SECONDS)
            await repo.log_event(user_id, job_id, file_id, "WHISPER_SEGMENT_DONE", {"chunks": len(chunks)})

            texts: list[str] = []
            meta_first: dict = {}
            for i, cpath in enumerate(chunks):
                await repo.log_event(user_id, job_id, file_id, "WHISPER_CHUNK_START", {"chunk_index": i, "file": cpath.name})
                res = transcribe_with_faster_whisper(str(cpath), model_name=model)
                await repo.log_event(
                    user_id,
                    job_id,
                    file_id,
                    "WHISPER_CHUNK_DONE",
                    {"chunk_index": i, "chars": len(res.get("text", "") or ""), "meta": res.get("meta", {})},
                )
                texts.append((res.get("text") or "").strip())
                if not meta_first:
                    meta_first = res.get("meta", {}) or {}

            merged = "\n".join([t for t in texts if t]).strip()
            meta_first.update({"chunked": True, "chunks": len(chunks), "segment_seconds": SEGMENT_SECONDS})
            await repo.log_event(user_id, job_id, file_id, "WHISPER_MERGED", {"chars": len(merged)})

            return {"text": merged, "meta": meta_first}

        # Single-pass whisper
        await repo.log_event(user_id, job_id, file_id, "WHISPER_START", {"model": model})
        res = transcribe_with_faster_whisper(str(wav), model_name=model)
        await repo.log_event(
            user_id,
            job_id,
            file_id,
            "WHISPER_DONE",
            {"chars": len(res.get("text", "") or ""), "meta": res.get("meta", {})},
        )
        return {"text": res.get("text", "") or "", "meta": res.get("meta", {}) or {}}


async def run_loop() -> None:
    if not settings.signed_url_expires_seconds:
        raise RuntimeError("SIGNED_URL_EXPIRES_SECONDS is not configured")

    parser_text_url = _parser_text_url()
    parser_tables_url = _parser_tables_url()
    parser_ocr_url = _parser_ocr_url()

    files_repo = FilesRepo()

    log.info("prof-worker started WORKER_ID=%s mode=%s", WORKER_ID, WORKER_MODE)

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        repo = ProfIngestionRepo(client)

        while True:
            try:
                job = await repo.claim_next_job()
            except Exception as e:
                log.exception("claim_next_job failed: %s", e)
                await asyncio.sleep(POLL_SECONDS)
                continue

            if not job:
                await asyncio.sleep(POLL_SECONDS)
                continue

            job_id = str(job["job_id"])
            file_id = str(job["file_id"])
            job_type = str(job.get("job_type") or "extract_text").lower()
            user_id = str(job.get("user_id") or "")

            try:
                file_row = await files_repo.get_file_record_service(file_id)
                if not file_row:
                    await repo.mark_failed(job_id, "FILE_NOT_FOUND", "File not found")
                    await asyncio.sleep(POLL_SECONDS)
                    continue

                if file_row.get("scan_result") != "clean":
                    await repo.mark_failed(job_id, "SCAN_NOT_CLEAN", f"scan_result={file_row.get('scan_result')}")
                    await asyncio.sleep(POLL_SECONDS)
                    continue

                bucket, path = _pick_storage_location(file_row)
                filename = _pick_filename(file_row)
                sha256 = str(file_row.get("sha256") or "")
                mime_type = file_row.get("mime_type")

                kind = _classify_file(filename, mime_type)

                # Heavy isolation gate
                if kind in ("audio", "video") and not _mode_allows(kind):
                    await repo.mark_failed(
                        job_id,
                        "HEAVY_WORKER_REQUIRED",
                        f"Worker mode '{WORKER_MODE}' does not process {kind}.",
                    )
                    await asyncio.sleep(POLL_SECONDS)
                    continue

                signed = await create_signed_download_url(
                    bucket=bucket,
                    path=path,
                    expires_in=settings.signed_url_expires_seconds,
                )

                await repo.log_event(user_id, job_id, file_id, "FILE_DOWNLOAD_START", {"bucket": bucket, "path": path})
                resp = await _get_with_retry(client, url=signed)
                resp.raise_for_status()
                file_bytes = resp.content
                await repo.log_event(
                    user_id,
                    job_id,
                    file_id,
                    "FILE_DOWNLOADED",
                    {"bytes": len(file_bytes), "filename": filename, "kind": kind},
                )

                extracted_text = ""
                extra_meta: dict = {"kind": kind, "filename": filename, "mime_type": mime_type}

                if job_type in ("full", "extract_text"):
                    if kind == "text":
                        presp = await client.post(
                            parser_text_url,
                            headers={"X-Parser-Secret": settings.parser_secret},
                            files={"file": (filename, file_bytes)},
                        )
                        presp.raise_for_status()
                        parsed = presp.json() if presp.content else {}
                        extracted_text = sanitize_pg_text((parsed.get("text") or "") if isinstance(parsed, dict) else "")
                        extracted_text, truncated = clamp_text(extracted_text, max_chars=MAX_TEXT_CHARS)
                        extra_meta["truncated"] = truncated

                    elif kind == "table":
                        tresp = await client.post(
                            parser_tables_url,
                            headers={"X-Parser-Secret": settings.parser_secret},
                            files={"file": (filename, file_bytes)},
                        )
                        tresp.raise_for_status()
                        tables_payload = tresp.json() if tresp.content else {}
                        extracted_text = sanitize_pg_text(_tables_to_text(tables_payload))
                        extracted_text, truncated = clamp_text(extracted_text, max_chars=MAX_TEXT_CHARS)
                        extra_meta["truncated"] = truncated

                    elif kind == "image":
                        oresp = await client.post(
                            parser_ocr_url,
                            headers={"X-Parser-Secret": settings.parser_secret},
                            files={"file": (filename, file_bytes)},
                        )
                        if oresp.status_code == 200:
                            ojson = oresp.json() if oresp.content else {}
                            extracted_text = sanitize_pg_text((ojson.get("text") or "") if isinstance(ojson, dict) else "")
                        extracted_text = extracted_text.strip() or f"[Image file: {filename}] (OCR produced no text)"
                        extracted_text, truncated = clamp_text(extracted_text, max_chars=MAX_TEXT_CHARS)
                        extra_meta["truncated"] = truncated

                    elif kind in ("audio", "video"):
                        t = await _transcribe_prof(
                            repo,
                            user_id=user_id,
                            job_id=job_id,
                            file_id=file_id,
                            filename=filename,
                            file_bytes=file_bytes,
                            kind=kind,
                        )
                        extracted_text = (t.get("text") or "").strip() or f"[{kind.upper()} file: {filename}] (no transcript)"
                        extracted_text, truncated = clamp_text(extracted_text, max_chars=MAX_TEXT_CHARS)
                        extra_meta["truncated"] = truncated
                        extra_meta["transcribe_meta"] = t.get("meta", {})

                    else:
                        extracted_text = f"[Unsupported file type: {filename}]"

                    red = redact_pii(extracted_text or "")
                    await repo.insert_prof_insight_adaptive(
                        user_id=user_id,
                        job_id=job_id,
                        file_id=file_id,
                        source_sha256=sha256,
                        redacted_text=red.redacted_text,
                        redaction_summary=red.summary,
                        extra_meta=extra_meta,
                    )
                    await repo.log_event(user_id, job_id, file_id, "INSIGHT_WRITTEN", {"chars": len(red.redacted_text)})

                await repo.mark_done(job_id, {"job_type": job_type, "kind": kind})
            except MediaTooLong as e:
                await repo.log_event(user_id, job_id, file_id, "MEDIA_REJECTED_TOO_LONG", {"error": str(e)})
                await repo.mark_failed(job_id, "MEDIA_TOO_LONG", str(e))
            except Exception as e:
                log.exception("job failed job_id=%s: %s", job_id, e)
                try:
                    await repo.mark_failed(job_id, "PROF_WORKER_ERROR", str(e))
                except Exception:
                    log.exception("mark_failed also failed job_id=%s", job_id)

            await asyncio.sleep(POLL_SECONDS)