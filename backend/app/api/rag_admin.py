from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.deps import CurrentUser, require_admin_mfa
from app.rag.ingestion.pipeline import ingest_knowledge_document

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