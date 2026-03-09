from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.core.deps import CurrentUser, get_current_user

router = APIRouter(prefix="/progress", tags=["progress"])


def _require_supabase_config() -> None:
    if not settings.supabase_url:
        raise HTTPException(status_code=500, detail="SUPABASE_URL not configured")
    if not settings.supabase_service_role_key:
        raise HTTPException(status_code=500, detail="SUPABASE_SERVICE_ROLE_KEY not configured")


def _service_headers() -> Dict[str, str]:
    _require_supabase_config()
    key = settings.supabase_service_role_key
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


async def _get_rows(path: str) -> List[dict]:
    url = f"{settings.supabase_url.rstrip('/')}/rest/v1/{path}"
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url, headers=_service_headers())
    if r.status_code >= 400:
        raise HTTPException(status_code=400, detail=f"Supabase fetch failed: {r.status_code} {r.text}")
    return r.json() or []


async def _load_file_owner(file_id: str) -> Optional[str]:
    rows = await _get_rows(f"files?id=eq.{file_id}&select=id,user_id&limit=1")
    if not rows:
        return None
    return str(rows[0].get("user_id") or "")


def _step_from_event(event_type: str) -> str:
    e = (event_type or "").lower()
    if "scan" in e:
        return "scan"
    if "extract" in e or "ocr" in e or "table" in e or "transcrib" in e:
        return "extract"
    if "infer" in e or "ml" in e:
        return "ml"
    if "llm" in e or "report" in e:
        return "llm"
    if "done" in e or "complete" in e or "processed" in e:
        return "report"
    return "extract"


def _normalize_events(rows: List[dict]) -> Dict[str, Any]:
    # expected DB: processing_events(file_id,event_type,details,created_at)
    events = []
    for r in rows:
        event_type = str(r.get("event_type") or "")
        events.append(
            {
                "event_type": event_type,
                "step": _step_from_event(event_type),
                "details": r.get("details") or {},
                "created_at": r.get("created_at"),
            }
        )
    # compute current step (last event)
    current = events[-1]["step"] if events else "scan"
    return {"current_step": current, "events": events}


@router.get("/latest/{file_id}")
async def latest(file_id: str, user: CurrentUser = Depends(get_current_user)):
    owner = await _load_file_owner(file_id)
    if not owner:
        raise HTTPException(status_code=404, detail="File not found")
    if user.role != "admin" and owner != str(user.id):
        raise HTTPException(status_code=404, detail="File not found")

    rows = await _get_rows(
        f"processing_events?file_id=eq.{file_id}&select=event_type,details,created_at&order=created_at.asc&limit=200"
    )
    return _normalize_events(rows)


@router.get("/stream/{file_id}")
async def stream(file_id: str, request: Request, user: CurrentUser = Depends(get_current_user)):
    """
    SSE stream of processing_events.
    Frontend will auto-reconnect; backend sends heartbeat.
    """
    owner = await _load_file_owner(file_id)
    if not owner:
        raise HTTPException(status_code=404, detail="File not found")
    if user.role != "admin" and owner != str(user.id):
        raise HTTPException(status_code=404, detail="File not found")

    async def event_gen():
        last_len = 0
        last_send = 0.0

        # initial snapshot
        try:
            rows = await _get_rows(
                f"processing_events?file_id=eq.{file_id}&select=event_type,details,created_at&order=created_at.asc&limit=200"
            )
            payload = _normalize_events(rows)
            last_len = len(payload["events"])
            yield f"event: snapshot\ndata: {json.dumps(payload)}\n\n"
        except Exception:
            yield "event: snapshot\ndata: {}\n\n"

        while True:
            # stop if client disconnects
            if await request.is_disconnected():
                break

            now = time.time()
            # heartbeat every 10s
            if now - last_send > 10:
                yield "event: ping\ndata: {}\n\n"
                last_send = now

            # poll for new events every 2s (SSE transport, polling source)
            await asyncio.sleep(2)

            try:
                rows = await _get_rows(
                    f"processing_events?file_id=eq.{file_id}&select=event_type,details,created_at&order=created_at.asc&limit=200"
                )
                payload = _normalize_events(rows)
                if len(payload["events"]) != last_len:
                    last_len = len(payload["events"])
                    yield f"event: update\ndata: {json.dumps(payload)}\n\n"
            except Exception:
                # keep stream alive
                continue

    return StreamingResponse(event_gen(), media_type="text/event-stream")
