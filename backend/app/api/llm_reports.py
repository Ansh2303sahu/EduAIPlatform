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
_LLM_PROXY_TIMEOUT = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=30.0)


def _llm_headers() -> dict[str, str]:
    if not settings.llm_service_secret:
        raise HTTPException(status_code=500, detail="LLM_SERVICE_SECRET not set")
    return {"x-ai-secret": settings.llm_service_secret}


def _describe_http_error(exc: BaseException) -> str:
    text = str(exc).strip()
    return text or repr(exc)


def _response_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except Exception:
        payload = None

    if isinstance(payload, dict):
        for key in ("detail", "message", "error"):
            value = payload.get(key)
            if value:
                return str(value).strip()

    try:
        text = response.text
    except Exception:
        text = ""
    return str(text or "").strip()


async def _post_llm_service(path: str, body: dict) -> dict:
    base_url = str(settings.llm_service_url or "").rstrip("/")
    if not base_url:
        raise HTTPException(status_code=500, detail="LLM_SERVICE_URL not set")

    url = f"{base_url}{path}"
    try:
        async with httpx.AsyncClient(timeout=_LLM_PROXY_TIMEOUT) as client:
            response = await client.post(
                url,
                json=body,
                headers=_llm_headers(),
            )
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                f"llm-service timed out calling {path}: "
                f"{type(exc).__name__}: {_describe_http_error(exc)}"
            ),
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                f"llm-service unreachable calling {path}: "
                f"{type(exc).__name__}: {_describe_http_error(exc)}"
            ),
        ) from exc

    if response.status_code >= 400:
        detail = _response_detail(response)
        raise HTTPException(
            status_code=502,
            detail=f"llm-service failed calling {path}: HTTP {response.status_code}: {detail}",
        )

    try:
        return response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"llm-service returned invalid JSON calling {path}",
        ) from exc

@router.post("/llm/student/report")
async def llm_student_report(
    body: dict,
    user: CurrentUser = Depends(require_roles("student", "admin")),
):
    outbound_body = dict(body)

    if settings.rag_enabled:
        rag_payload = build_student_rag_payload(outbound_body)
        outbound_body = inject_rag_fields(outbound_body, rag_payload)

    response_data = await _post_llm_service("/llm/student/report", outbound_body)

    if settings.rag_enabled:
        response_data["rag_meta"] = rag_meta_from_grounding_fields(outbound_body)

    return response_data


@router.post("/llm/professor/report")
async def llm_prof_report(
    body: dict,
    user: CurrentUser = Depends(require_roles("professor", "admin")),
):
    outbound_body = dict(body)

    if settings.rag_enabled:
        rag_payload = build_professor_rag_payload(outbound_body)
        outbound_body = inject_rag_fields(outbound_body, rag_payload)

    response_data = await _post_llm_service("/llm/professor/report", outbound_body)

    if settings.rag_enabled:
        response_data["rag_meta"] = rag_meta_from_grounding_fields(outbound_body)

    return response_data
