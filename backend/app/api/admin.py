from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.config import settings
from app.core.deps import CurrentUser, require_admin_mfa

router = APIRouter(prefix="/admin", tags=["admin"])


# -------------------------
# Supabase helpers (service role)
# -------------------------
def _require_supabase_config() -> None:
    if not settings.supabase_url:
        raise HTTPException(status_code=500, detail="SUPABASE_URL not configured")
    if not settings.supabase_service_role_key:
        raise HTTPException(status_code=500, detail="SUPABASE_SERVICE_ROLE_KEY not configured")


def _service_headers(*, prefer_count: bool = False) -> Dict[str, str]:
    _require_supabase_config()
    key = settings.supabase_service_role_key
    h = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if prefer_count:
        h["Prefer"] = "count=exact"
    return h


def _rest_base() -> str:
    return f"{settings.supabase_url.rstrip('/')}/rest/v1"


async def _get(path: str, *, prefer_count: bool = False) -> httpx.Response:
    url = f"{_rest_base()}/{path.lstrip('/')}"
    async with httpx.AsyncClient(timeout=20) as client:
        return await client.get(url, headers=_service_headers(prefer_count=prefer_count))


def _count_from_content_range(r: httpx.Response) -> int:
    cr = (r.headers.get("content-range") or "").strip()  # e.g. "0-0/123"
    if "/" in cr:
        try:
            return int(cr.split("/")[-1])
        except Exception:
            return 0
    return 0


def _utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


async def _count(table: str, where: str = "") -> int:
    # Count via Prefer: count=exact, limit=1 for speed
    r = await _get(f"{table}?select=id{where}&limit=1", prefer_count=True)
    if r.status_code >= 400:
        return 0
    return _count_from_content_range(r)


# -------------------------
# Admin: Metrics (matches your frontend keys)
# -------------------------
@router.get("/metrics")
async def metrics(admin: CurrentUser = Depends(require_admin_mfa())):
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=24)
    since_iso = _utc_iso(since)

    total_users = await _count("profiles")
    total_files = await _count("files")
    total_reports = await _count("ai_reports")  # Phase 7

    needs_review = await _count("ai_reports", "&needs_review=eq.true")
    audit_events_total = await _count("audit_logs")

    failed_jobs = await _count("ingestion_jobs", f"&status=eq.failed&created_at=gte.{since_iso}")
    quarantined_files = await _count("files", f"&status=eq.quarantined&created_at=gte.{since_iso}")
    failed_files = await _count("files", f"&status=eq.failed&created_at=gte.{since_iso}")

    return {
        "total_users": total_users,
        "total_files": total_files,
        "total_reports": total_reports,
        "needs_review": needs_review,
        "audit_events_total": audit_events_total,
        "failures_24h": {
            "failed_jobs": failed_jobs,
            "quarantined_files": quarantined_files,
            "failed_files": failed_files,
        },
        "window": {"since": since_iso, "now": _utc_iso(now)},
    }


# -------------------------
# Admin: Audit logs
# -------------------------
@router.get("/audit")
async def audit(
    limit: int = Query(100, ge=1, le=500),
    action_prefix: Optional[str] = Query(default=None),
    admin: CurrentUser = Depends(require_admin_mfa()),
):
    q = f"audit_logs?select=*&order=created_at.desc&limit={limit}"
    if action_prefix:
        q += f"&action=ilike.{action_prefix}*"

    r = await _get(q)
    if r.status_code >= 400:
        raise HTTPException(status_code=400, detail=f"Failed to fetch audit logs: {r.text}")
    return {"items": r.json() or []}


# -------------------------
# Admin: Models (your real tables)
# -------------------------
@router.get("/models")
async def models(
    limit: int = Query(100, ge=1, le=200),
    admin: CurrentUser = Depends(require_admin_mfa()),
):
    r1 = await _get(f"model_registry?select=*&order=created_at.desc&limit={limit}")
    if r1.status_code >= 400:
        raise HTTPException(status_code=400, detail=f"Failed model_registry: {r1.text}")
    registry = r1.json() or []

    r2 = await _get("model_deployments?select=*&order=created_at.desc&limit=50")
    deployments = r2.json() if r2.status_code < 400 else []

    r3 = await _get("dataset_versions?select=*&order=created_at.desc&limit=50")
    datasets = r3.json() if r3.status_code < 400 else []

    return {
        "model_registry": registry,
        "model_deployments": deployments,
        "dataset_versions": datasets,
    }


# -------------------------
# Admin: Workers / Jobs
# -------------------------
@router.get("/workers")
async def workers(admin: CurrentUser = Depends(require_admin_mfa())):
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=24)
    since_iso = _utc_iso(since)

    r1 = await _get(
        "ingestion_jobs?"
        "select=id,status,job_type,created_at,error_code,error_message,file_id&"
        f"created_at=gte.{since_iso}&order=created_at.desc&limit=200"
    )
    jobs = r1.json() if r1.status_code < 400 else []

    r2 = await _get(
        "prof_ingestion_jobs?"
        "select=id,status,job_type,created_at,error_code,error_message,file_id&"
        f"created_at=gte.{since_iso}&order=created_at.desc&limit=200"
    )
    prof_jobs = r2.json() if r2.status_code < 400 else []

    def summarize(items):
        counts: Dict[str, int] = {}
        for j in items:
            s = (j.get("status") or "unknown").lower()
            counts[s] = counts.get(s, 0) + 1
        return counts

    return {
        "window": {"since": since_iso, "now": _utc_iso(now)},
        "student": {"counts": summarize(jobs), "recent": jobs[:50]},
        "professor": {"counts": summarize(prof_jobs), "recent": prof_jobs[:50]},
    }


# -------------------------
# Admin: Security Alerts
# -------------------------
@router.get("/security-alerts")
async def security_alerts(
    limit: int = Query(50, ge=1, le=200),
    admin: CurrentUser = Depends(require_admin_mfa()),
):
    rf = await _get(
        "files?"
        "select=id,user_id,status,scan_engine,scan_result,scanned_at,created_at&"
        f"status=eq.quarantined&order=scanned_at.desc&limit={limit}"
    )
    quarantined = rf.json() if rf.status_code < 400 else []

    ri = await _get(f"inference_events?select=*&order=created_at.desc&limit={limit}")
    inference = ri.json() if ri.status_code < 400 else []

    rpi = await _get(f"prof_events?select=*&order=created_at.desc&limit={limit}")
    prof_inference = rpi.json() if rpi.status_code < 400 else []

    return {
        "quarantined_files": quarantined,
        "inference_events": inference,
        "prof_events": prof_inference,
    }


@router.get("/analytics")
async def analytics(admin: CurrentUser = Depends(require_admin_mfa())):
    r = await _get("ai_reports?select=needs_review,role,model_versions,created_at&order=created_at.desc&limit=200")
    rows = r.json() if r.status_code < 400 else []

    buckets = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "unknown": 0}
    needs_review = 0
    by_role = {"student": 0, "professor": 0}

    for x in rows:
        if x.get("needs_review"):
            needs_review += 1
        role = str(x.get("role") or "unknown")
        if role in by_role:
            by_role[role] += 1
        mv = x.get("model_versions") or {}
        ag = mv.get("agreement") or {}
        b = ag.get("ml_bucket_0_to_4", None)
        if b is None:
            buckets["unknown"] += 1
        else:
            buckets[str(int(b))] = buckets.get(str(int(b)), 0) + 1

    total = len(rows) or 1
    return {
        "window": {"items": len(rows)},
        "confidence_buckets": buckets,
        "needs_review_rate": needs_review / total,
        "reports_by_role": by_role,
    }