from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends, HTTPException

from app.core.config import settings
from app.core.deps import CurrentUser, require_roles
from app.rag.payloads import inject_rag_fields, rag_meta_from_grounding_fields
from app.rag.retrieval.context_builder import (
    build_professor_rag_payload,
    build_student_rag_payload,
)

router = APIRouter(tags=["llm-reports"])


def _llm_headers() -> dict[str, str]:
    if not settings.llm_service_secret:
        raise HTTPException(status_code=500, detail="LLM_SERVICE_SECRET not set")
    return {"x-ai-secret": settings.llm_service_secret}

@router.post("/llm/student/report")
async def llm_student_report(
    body: dict,
    user: CurrentUser = Depends(require_roles("student", "admin")),
):
    if not settings.llm_service_url:
        raise HTTPException(status_code=500, detail="LLM_SERVICE_URL not set")

    outbound_body = dict(body)

    if settings.rag_enabled:
        rag_payload = build_student_rag_payload(outbound_body)
        outbound_body = inject_rag_fields(outbound_body, rag_payload)

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{settings.llm_service_url}/llm/student/report",
            json=outbound_body,
            headers=_llm_headers(),
        )

    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"llm-service failed: {r.text}")

    response_data = r.json()

    if settings.rag_enabled:
        response_data["rag_meta"] = rag_meta_from_grounding_fields(outbound_body)

    return response_data


@router.post("/llm/professor/report")
async def llm_prof_report(
    body: dict,
    user: CurrentUser = Depends(require_roles("professor", "admin")),
):
    if not settings.llm_service_url:
        raise HTTPException(status_code=500, detail="LLM_SERVICE_URL not set")

    outbound_body = dict(body)

    if settings.rag_enabled:
        rag_payload = build_professor_rag_payload(outbound_body)
        outbound_body = inject_rag_fields(outbound_body, rag_payload)

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{settings.llm_service_url}/llm/professor/report",
            json=outbound_body,
            headers=_llm_headers(),
        )

    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"llm-service failed: {r.text}")

    response_data = r.json()

    if settings.rag_enabled:
        response_data["rag_meta"] = rag_meta_from_grounding_fields(outbound_body)

    return response_data
