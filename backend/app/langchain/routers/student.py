"""
LangChain student generation router.

Primary public path:
- POST /api/langchain/student/generate

Compatibility alias:
- POST /api/phase10/student/generate
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Mapping

from fastapi import APIRouter, Depends

from app.core.config import settings
from app.core.deps import CurrentUser, require_roles
from app.langchain.schemas import (
    LangChainRagMeta,
    LangChainResponseMeta,
    LangChainStoredSummary,
    LangChainStudentResponse,
    Phase10GenerateIn,
)
from app.langchain.services.student_pipeline import StudentPipeline
from app.services.report_generation_support import (
    build_ingestion_bundle as _build_ingestion_bundle,
    call_ai_student_multimodal as _call_ai_student_multimodal,
    detect_submission_kind as _detect_submission_kind,
    get_rows as _get_rows,
    load_file as _load_file,
    normalize_report_row as _normalize_report_row,
    post_row as _post_row,
    rate_limit as _rate_limit,
    sha256_json as _sha256_json,
    uuid_or_none as _uuid_or_none,
)

router = APIRouter(prefix="/langchain", tags=["langchain"])
legacy_router = APIRouter(prefix="/phase10", tags=["phase10"])
logger = logging.getLogger("langchain.router.student")

_pipeline = StudentPipeline()


@router.post("/student/generate", response_model=LangChainStudentResponse)
async def generate_student(
    body: Phase10GenerateIn,
    user: CurrentUser = Depends(require_roles("student", "admin")),
):
    return await _generate_student(body, user)


@legacy_router.post(
    "/student/generate",
    response_model=LangChainStudentResponse,
    include_in_schema=False,
)
async def generate_student_legacy(
    body: Phase10GenerateIn,
    user: CurrentUser = Depends(require_roles("student", "admin")),
):
    return await _generate_student(body, user)


async def _generate_student(
    body: Phase10GenerateIn,
    user: CurrentUser,
) -> LangChainStudentResponse:
    _rate_limit(str(user.id))

    request_id = str(uuid.uuid4())
    t_all = time.perf_counter()

    file_row = await _load_file(body.file_id, user)
    ingestion = await _build_ingestion_bundle(body.file_id, user)
    submission_kind = _detect_submission_kind(ingestion)
    analysis_type = (
        "student_project_review"
        if submission_kind == "project"
        else "student_academic_review"
    )

    input_hash = _sha256_json(ingestion)
    prompt_hash = _sha256_json(
        {
            "role": "student",
            "template": "langchain_student_v1",
            "analysis_type": analysis_type,
            "rag_enabled": settings.rag_enabled,
        }
    )

    if not body.force:
        existing = await _get_rows(
            "ai_reports"
            f"?file_id=eq.{body.file_id}&role=eq.student"
            f"&input_hash=eq.{input_hash}&prompt_hash=eq.{prompt_hash}"
            "&select=*&order=created_at.desc&limit=1"
        )
        if existing:
            normalized = _normalize_report_row("student", existing[0]) or {}
            response = _build_cached_student_response(
                request_id=request_id,
                stored_row=normalized,
            )
            return LangChainStudentResponse.model_validate(response)

    ml = await _call_ai_student_multimodal(user, ingestion)

    result = await _pipeline.run(
        file_id=body.file_id,
        submission_id=str(file_row.get("submission_id") or ""),
        ingestion_dict=ingestion,
        ml_dict=ml,
        submission_kind=submission_kind,
        user_id=str(user.id),
        file_metadata={
            "file_id": body.file_id,
            "submission_id": file_row.get("submission_id"),
            "mime_type": file_row.get("mime_type"),
            "created_at": file_row.get("created_at"),
        },
    )

    total_ms = int((time.perf_counter() - t_all) * 1000)
    stored = await _post_row(
        "ai_reports",
        {
            "file_id": _uuid_or_none(body.file_id),
            "submission_id": _uuid_or_none(file_row.get("submission_id")),
            "role": "student",
            "report_json": result["report"],
            "report_hash": _sha256_json(result["report"]),
            "prompt_hash": prompt_hash,
            "input_hash": input_hash,
            "model_versions": result.get("storage_payload", {}).get("model_versions", {}),
            **result["storage_fields"],
            "needs_review": bool(result["report"].get("safety", {}).get("needs_review", False)),
        },
    )

    logger.info(
        "langchain student generate complete file_id=%s request_id=%s model=%s total_ms=%d",
        body.file_id,
        request_id,
        result.get("model_used"),
        total_ms,
    )

    response = {
        "cached": False,
        "request_id": request_id,
        "ml": result.get("ml", {}),
        "report": result["report"],
        "rag_meta": _public_rag_meta(result.get("rag_meta")),
        "meta": _public_meta_from_result("student", result),
        "stored": _public_stored_summary(stored),
    }
    return LangChainStudentResponse.model_validate(response)


def _build_cached_student_response(
    *,
    request_id: str,
    stored_row: Mapping[str, Any],
) -> dict[str, Any]:
    report = stored_row.get("report_json") or {}
    return {
        "cached": True,
        "request_id": request_id,
        "ml": {},
        "report": report,
        "rag_meta": _public_rag_meta_from_row(stored_row),
        "meta": {
            "role": "student",
            "needs_review": bool((report.get("safety") or {}).get("needs_review", False)),
            "model_used": str(
                (((stored_row.get("model_versions") or {}).get("llm_model_used")) or "")
            ),
            "fallback_used": False,
            "decision_source": "llm",
            "execution_mode": "normal",
            "discrepancy_flag": None,
        },
        "stored": _public_stored_summary(stored_row),
    }


def _public_meta_from_result(role: str, result: Mapping[str, Any]) -> LangChainResponseMeta:
    report = result.get("report") or {}
    return LangChainResponseMeta(
        role=role,  # type: ignore[arg-type]
        needs_review=bool((report.get("safety") or {}).get("needs_review", False)),
        model_used=str(result.get("model_used") or ""),
        fallback_used=bool(result.get("fallback_used", False)),
        decision_source=str(result.get("decision_source") or "llm"),
        execution_mode=str(result.get("execution_mode") or "normal"),
        discrepancy_flag=bool(result.get("discrepancy_flag", False)),
    )


def _normalize_citations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _public_rag_meta(rag_meta: Mapping[str, Any] | None) -> LangChainRagMeta:
    rag = dict(rag_meta or {})
    citations = _normalize_citations(rag.get("citations"))
    chunk_count = rag.get("chunk_count")
    if not isinstance(chunk_count, int):
        retrieved = rag.get("retrieved_chunks")
        chunk_count = len(retrieved) if isinstance(retrieved, list) else 0

    return LangChainRagMeta(
        enabled=bool(rag.get("enabled", False)),
        confidence_score=float(rag.get("confidence_score", 0.0) or 0.0),
        confidence_label=str(rag.get("confidence_label") or "low"),
        safe_review=bool(rag.get("safe_review", False)),
        chunk_count=max(0, int(chunk_count)),
        weak_retrieval=bool(rag.get("weak_retrieval", False)),
        citations=citations,
    )


def _public_rag_meta_from_row(row: Mapping[str, Any]) -> LangChainRagMeta:
    retrieved_chunks = row.get("retrieved_chunks")
    citations = _normalize_citations(row.get("citations"))
    confidence_label = str(row.get("retrieval_confidence_label") or "low")
    safe_review = bool(row.get("safe_review", False))
    chunk_count = len(retrieved_chunks) if isinstance(retrieved_chunks, list) else 0
    weak_retrieval = bool(
        safe_review
        or confidence_label.lower() == "low"
        or chunk_count == 0
        or len(citations) == 0
    )
    return LangChainRagMeta(
        enabled=bool(chunk_count or citations),
        confidence_score=float(row.get("retrieval_confidence", 0.0) or 0.0),
        confidence_label=confidence_label,
        safe_review=safe_review,
        chunk_count=chunk_count,
        weak_retrieval=weak_retrieval,
        citations=citations,
    )


def _public_stored_summary(row: Mapping[str, Any] | None) -> LangChainStoredSummary | None:
    if not isinstance(row, Mapping):
        return None
    return LangChainStoredSummary(
        id=str(row.get("id") or ""),
        file_id=str(row.get("file_id") or ""),
        submission_id=str(row.get("submission_id")) if row.get("submission_id") is not None else None,
        role=str(row.get("role") or ""),
        needs_review=bool(row.get("needs_review", False)),
        created_at=str(row.get("created_at")) if row.get("created_at") is not None else None,
    )
