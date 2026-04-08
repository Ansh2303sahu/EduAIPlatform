from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from io import BytesIO

from app.core.deps import CurrentUser, get_current_user
from app.genai.service import GenAIService
from app.genai.schemas import AIReportGenerateIn, AIReportResponse, AuditResponse, CompareResponse, ExplainResponse

router = APIRouter(prefix="/ai", tags=["ai"])
_service = GenAIService()


def _require_role(user: CurrentUser, *allowed: str) -> None:
    if user.role not in allowed:
        raise HTTPException(status_code=403, detail="Forbidden")


def _resolve_role(user: CurrentUser, role: str | None) -> Literal["student", "professor"]:
    if role in {"student", "professor"}:
        return role
    if user.role == "professor":
        return "professor"
    return "student"


@router.post("/generate/student-report", response_model=AIReportResponse)
async def generate_student_report(
    body: AIReportGenerateIn,
    user: CurrentUser = Depends(get_current_user),
):
    _require_role(user, "student", "admin")
    return await _service.generate_student_report(body, user)


@router.post("/generate/professor-report", response_model=AIReportResponse)
async def generate_professor_report(
    body: AIReportGenerateIn,
    user: CurrentUser = Depends(get_current_user),
):
    _require_role(user, "professor", "admin")
    return await _service.generate_professor_report(body, user)


@router.get("/explain/{file_id}", response_model=ExplainResponse)
async def explain_report(
    file_id: str,
    role: str | None = Query(default=None),
    user: CurrentUser = Depends(get_current_user),
):
    resolved = _resolve_role(user, role)
    if resolved == "professor":
        _require_role(user, "professor", "admin")
    return await _service.explain(file_id=file_id, role=resolved, user=user)


@router.get("/compare/{file_id}", response_model=CompareResponse)
async def compare_report(
    file_id: str,
    role: str | None = Query(default=None),
    user: CurrentUser = Depends(get_current_user),
):
    resolved = _resolve_role(user, role)
    if resolved == "professor":
        _require_role(user, "professor", "admin")
    return await _service.compare(file_id=file_id, role=resolved, user=user)


@router.get("/audit/{file_id}", response_model=AuditResponse)
async def audit_report(
    file_id: str,
    role: str | None = Query(default=None),
    include_pdf: bool = Query(default=False),
    user: CurrentUser = Depends(get_current_user),
):
    resolved = _resolve_role(user, role)
    if resolved == "professor":
        _require_role(user, "professor", "admin")
    return await _service.audit(file_id=file_id, role=resolved, user=user, include_pdf=include_pdf)


@router.get("/pdf/{file_id}")
async def pdf_report(
    file_id: str,
    role: str | None = Query(default=None),
    user: CurrentUser = Depends(get_current_user),
):
    resolved = _resolve_role(user, role)
    if resolved == "professor":
        _require_role(user, "professor", "admin")
    pdf_bytes, filename = await _service.pdf(file_id=file_id, role=resolved, user=user)
    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
