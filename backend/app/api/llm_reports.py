from __future__ import annotations
import httpx
from fastapi import APIRouter, Depends, HTTPException
from app.core.config import settings
from app.core.deps import CurrentUser, get_current_user, require_roles

router = APIRouter(tags=["llm-reports"])

def _llm_headers():
    if not settings.llm_service_secret:
        raise HTTPException(status_code=500, detail="LLM_SERVICE_SECRET not set")
    return {"x-ai-secret": settings.llm_service_secret}

@router.post("/llm/student/report")
async def llm_student_report(body: dict, user: CurrentUser = Depends(require_roles("student","admin"))):
    if not settings.llm_service_url:
        raise HTTPException(status_code=500, detail="LLM_SERVICE_URL not set")

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{settings.llm_service_url}/llm/student/report", json=body, headers=_llm_headers())
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"llm-service failed: {r.text}")
    return r.json()

@router.post("/llm/professor/report")
async def llm_prof_report(body: dict, user: CurrentUser = Depends(require_roles("professor","admin"))):
    if not settings.llm_service_url:
        raise HTTPException(status_code=500, detail="LLM_SERVICE_URL not set")

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{settings.llm_service_url}/llm/professor/report", json=body, headers=_llm_headers())
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"llm-service failed: {r.text}")
    return r.json()