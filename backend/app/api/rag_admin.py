from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.deps import CurrentUser, require_admin_mfa
from app.rag.ingestion.pipeline import ingest_knowledge_document
from app.rag.retrieval.professor_retriever import retrieve_professor_context
from app.rag.retrieval.student_retriever import retrieve_student_context

router = APIRouter(prefix="/rag/admin", tags=["rag-admin"])


class RAGIngestIn(BaseModel):
    file_path: str = Field(..., min_length=1)
    audience: str = Field(..., pattern="^(student|professor)$")
    category: str = Field(..., min_length=1)
    uploaded_by: str = Field(..., min_length=1)

    document_title: str | None = None
    document_id: str | None = None
    version: str = "v1"
    source_priority: int = Field(default=50, ge=0, le=100)
    is_official: bool = False
    parent_doc_id: str | None = None
    effective_date: str | None = None
    status: str = Field(default="active", pattern="^(active|inactive|draft)$")


class RAGRetrievalDebugIn(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=4, ge=1, le=12)
    category: str | None = None
    version: str | None = None
    submission_type: str | None = None
    analysis_type: str | None = None
    preferred_categories: list[str] = Field(default_factory=list)
    official_bias: float = Field(default=0.0, ge=0.0, le=1.0)
    ml_signals: dict = Field(default_factory=dict)
    official_only: bool | None = None
    text_excerpt: str | None = None
    title_hint: str | None = None
    keywords: list[str] = Field(default_factory=list)
    mode: str | None = None
    degraded_input: bool = False
    include_eligibility: bool = True


@router.post("/ingest")
async def ingest_rag_document(
    body: RAGIngestIn,
    admin: CurrentUser = Depends(require_admin_mfa()),
):
    path = Path(body.file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Knowledge file not found: {body.file_path}")

    try:
        result = ingest_knowledge_document(
            file_path=body.file_path,
            audience=body.audience,
            category=body.category,
            uploaded_by=body.uploaded_by,
            document_title=body.document_title,
            document_id=body.document_id,
            version=body.version,
            source_priority=body.source_priority,
            is_official=body.is_official,
            parent_doc_id=body.parent_doc_id,
            effective_date=body.effective_date,
            status=body.status,
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG ingestion failed: {e}")

@router.get("/coverage/{audience}")
async def rag_coverage(
    audience: str,
    admin: CurrentUser = Depends(require_admin_mfa()),
):
    """Return a diagnostic summary of what is indexed in the vector store.

    Reports total chunks, active vs inactive, per-category breakdown, and
    per-document chunk counts.  Read-only — does not modify the store.
    """
    if audience not in ("student", "professor"):
        raise HTTPException(status_code=400, detail="audience must be 'student' or 'professor'")
    try:
        from app.rag.debug_coverage import coverage_report
        return coverage_report(audience)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Coverage report failed: {e}")


@router.post("/retrieval-debug/{audience}")
async def rag_retrieval_debug(
    audience: str,
    body: RAGRetrievalDebugIn,
    admin: CurrentUser = Depends(require_admin_mfa()),
):
    if audience not in ("student", "professor"):
        raise HTTPException(status_code=400, detail="audience must be 'student' or 'professor'")

    try:
        if audience == "student":
            result = retrieve_student_context(
                query=body.query,
                top_k=body.top_k,
                category=body.category,
                version=body.version,
                ml_signals=body.ml_signals,
                submission_type=body.submission_type,
                analysis_type=body.analysis_type,
                preferred_categories=body.preferred_categories,
                official_bias=body.official_bias,
                text_excerpt=body.text_excerpt,
                keywords=body.keywords,
                title_hint=body.title_hint,
                mode=body.mode,
                degraded_input=body.degraded_input,
            )
        else:
            result = retrieve_professor_context(
                query=body.query,
                top_k=body.top_k,
                category=body.category,
                version=body.version,
                ml_signals=body.ml_signals,
                submission_type=body.submission_type,
                official_only=body.official_only,
                analysis_type=body.analysis_type,
                preferred_categories=body.preferred_categories,
                official_bias=body.official_bias,
                text_excerpt=body.text_excerpt,
                keywords=body.keywords,
                title_hint=body.title_hint,
                mode=body.mode,
                degraded_input=body.degraded_input,
            )

        response = {"result": result.model_dump()}
        if body.include_eligibility:
            from app.rag.debug_coverage import eligibility_report

            response["eligibility_summary"] = eligibility_report(
                audience,
                (result.trace.applied_filters or {}),
            )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval debug failed: {e}")
