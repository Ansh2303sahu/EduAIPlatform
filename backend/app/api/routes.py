from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.deps import CurrentUser, get_current_user, require_admin_mfa

from app.api.files import router as files_router
from app.api.ingestion import router as ingestion_router
from app.api.results import router as results_router
from app.api.prof_ingestion import router as prof_ingestion_router
from app.api.prof_results import router as prof_results_router
from app.api.media import router as media_router
from app.api.phase7 import router as phase7_router
from app.api.admin import router as admin_router

from app.api.progress import router as progress_router
# main.py includes this with prefix="/api"
api_router = APIRouter()
router = APIRouter(prefix="/prof", tags=["prof-results"])

# -------------------------
# Helpers
# -------------------------

def _require_supabase_config() -> None:
    if not settings.supabase_url:
        raise HTTPException(status_code=500, detail="SUPABASE_URL is not configured")
    if not settings.supabase_service_role_key:
        raise HTTPException(status_code=500, detail="SUPABASE_SERVICE_ROLE_KEY is not configured")


def _service_headers(*, prefer_return: bool) -> dict[str, str]:
    _require_supabase_config()
    key = settings.supabase_service_role_key
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation" if prefer_return else "return=minimal",
    }


async def _audit_log(
    action: str,
    actor_user_id: str,
    entity_type: str | None = None,
    entity_id: str | None = None,
    metadata: dict | None = None,
) -> None:
    try:
        url = f"{settings.supabase_url.rstrip('/')}/rest/v1/audit_logs"
        payload = {
            "actor_user_id": actor_user_id,
            "action": action,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "metadata": metadata or {},
        }
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(url, headers=_service_headers(prefer_return=False), json=payload)
    except Exception:
        return


# -------------------------
# Auth-ish
# -------------------------

@api_router.get("/me", tags=["auth"])
async def me(user: CurrentUser = Depends(get_current_user)):
    return {"id": user.id, "email": user.email, "role": user.role}


# -------------------------
# Admin
# -------------------------

class RoleUpdateIn(BaseModel):
    role: str = Field(..., pattern="^(student|professor|admin)$")


@api_router.post("/admin/users/{user_id}/role", tags=["admin"])
async def set_user_role(
    user_id: str,
    body: RoleUpdateIn,
    admin: CurrentUser = Depends(require_admin_mfa()),
):
    _require_supabase_config()

    url = f"{settings.supabase_url.rstrip('/')}/rest/v1/profiles?id=eq.{user_id}"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.patch(url, headers=_service_headers(prefer_return=True), json={"role": body.role})

    if r.status_code >= 400:
        raise HTTPException(status_code=400, detail=f"Failed to update role: {r.text}")

    return {"updated": r.json()}


# -------------------------
# Submissions
# -------------------------

class SubmissionIn(BaseModel):
    content: str = Field(..., min_length=1)


class SubmissionOut(BaseModel):
    id: str
    user_id: str
    content: str
    created_at: str


@api_router.post("/submissions", response_model=SubmissionOut, tags=["submissions"])
async def create_submission(body: SubmissionIn, user: CurrentUser = Depends(get_current_user)):
    _require_supabase_config()

    url = f"{settings.supabase_url.rstrip('/')}/rest/v1/submissions"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(
            url,
            headers=_service_headers(prefer_return=True),
            json={"user_id": user.id, "content": body.content},
        )

    if r.status_code >= 400:
        raise HTTPException(status_code=400, detail=f"Failed to create submission: {r.text}")

    rows = r.json() or []
    if not rows:
        raise HTTPException(status_code=500, detail="Supabase did not return created submission")

    row = rows[0]
    await _audit_log("submission.created", actor_user_id=user.id, entity_type="submission", entity_id=row["id"])
    return row


@api_router.get("/submissions/{submission_id}", response_model=SubmissionOut, tags=["submissions"])
async def get_submission(submission_id: str, user: CurrentUser = Depends(get_current_user)):
    _require_supabase_config()

    url = f"{settings.supabase_url.rstrip('/')}/rest/v1/submissions?id=eq.{submission_id}&select=*"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, headers=_service_headers(prefer_return=False))

    if r.status_code >= 400:
        raise HTTPException(status_code=400, detail=f"Failed to fetch submission: {r.text}")

    rows = r.json() or []
    if not rows:
        raise HTTPException(status_code=404, detail="Not found")

    row = rows[0]
    if row.get("user_id") != user.id and user.role != "admin":
        raise HTTPException(status_code=404, detail="Not found")

    return row


# -------------------------
# Reports
# -------------------------

class ReportIn(BaseModel):
    submission_id: str
    report: dict[str, Any]


class ReportOut(BaseModel):
    id: str
    submission_id: str
    user_id: str
    report: dict[str, Any]
    created_at: str


@api_router.post("/reports", response_model=ReportOut, tags=["reports"])
async def create_report(body: ReportIn, user: CurrentUser = Depends(get_current_user)):
    _require_supabase_config()

    sub_url = f"{settings.supabase_url.rstrip('/')}/rest/v1/submissions?id=eq.{body.submission_id}&select=id,user_id"
    async with httpx.AsyncClient(timeout=10) as client:
        sub_r = await client.get(sub_url, headers=_service_headers(prefer_return=False))

    if sub_r.status_code >= 400:
        raise HTTPException(status_code=400, detail=f"Failed to validate submission: {sub_r.text}")

    subs = sub_r.json() or []
    if not subs:
        raise HTTPException(status_code=404, detail="Submission not found")

    owner_id = subs[0]["user_id"]
    if owner_id != user.id and user.role != "admin":
        raise HTTPException(status_code=404, detail="Submission not found")

    url = f"{settings.supabase_url.rstrip('/')}/rest/v1/reports"
    payload = {"submission_id": body.submission_id, "user_id": owner_id, "report": body.report}

    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(url, headers=_service_headers(prefer_return=True), json=payload)

    if r.status_code >= 400:
        raise HTTPException(status_code=400, detail=f"Failed to create report: {r.text}")

    rows = r.json() or []
    if not rows:
        raise HTTPException(status_code=500, detail="Supabase did not return created report")

    row = rows[0]
    await _audit_log(
        "report.created",
        actor_user_id=user.id,
        entity_type="report",
        entity_id=row["id"],
        metadata={"submission_id": body.submission_id},
    )
    return row


# -------------------------
# Routers
# -------------------------

api_router.include_router(files_router)
api_router.include_router(ingestion_router)
api_router.include_router(results_router)
api_router.include_router(prof_ingestion_router)
api_router.include_router(prof_results_router)
api_router.include_router(media_router)
api_router.include_router(phase7_router)
api_router.include_router(progress_router)
api_router.include_router(admin_router)
router.get("/results/{file_id}")