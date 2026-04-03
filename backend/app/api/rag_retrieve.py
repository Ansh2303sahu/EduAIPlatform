from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.deps import CurrentUser, get_current_user
from app.rag.retrieval.student_retriever import retrieve_student_context
from app.rag.retrieval.professor_retriever import retrieve_professor_context
from app.rag.security.role_guard import enforce_audience_role_match

router = APIRouter(prefix="/rag", tags=["rag"])


class RAGRetrieveIn(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=4, ge=1, le=10)
    category: str | None = None
    version: str | None = None
    submission_type: str | None = None
    analysis_type: str | None = None
    preferred_categories: list[str] = Field(default_factory=list)
    official_bias: float = Field(default=0.0, ge=0.0, le=1.0)
    ml_signals: dict = Field(default_factory=dict)
    official_only: bool | None = None


@router.post("/student/retrieve")
async def rag_student_retrieve(
    body: RAGRetrieveIn,
    user: CurrentUser = Depends(get_current_user),
):
    try:
        enforce_audience_role_match(user.role, "student")
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
        )
        return result.model_dump()
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Student RAG retrieval failed: {e}")


@router.post("/professor/retrieve")
async def rag_professor_retrieve(
    body: RAGRetrieveIn,
    user: CurrentUser = Depends(get_current_user),
):
    try:
        enforce_audience_role_match(user.role, "professor")
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
        )
        return result.model_dump()
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Professor RAG retrieval failed: {e}")
