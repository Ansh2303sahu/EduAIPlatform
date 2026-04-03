"""
FastAPI request/response schemas for Phase 10 routers.

Two sets of output schemas live here:

1. **Phase 7-compatible schemas** (``Phase10StudentReport``, ``Phase10ProfessorReport``)
   — used by ``parsers/validators.py``, ``parsers/normalizer.py``, and the existing
   tests.  These mirror the llm-service output shape and must not change.

2. **Phase 10-native schemas** (``Phase10StudentOut``, ``Phase10ProfessorOut``)
   — streamlined, flat-typed schemas used by the new LangChain pipeline.
   Strengths/improvement_plan/moderation_notes are ``list[str]`` rather than nested
   objects, and RAG citations are surfaced as typed ``CitationOut`` items.  These
   schemas have safe defaults on every optional field so a partial LLM response
   never causes a crash.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ===========================================================================
# Phase 7-compatible sub-schemas (used by validators.py / normalizer.py)
# ===========================================================================

class _Issue(BaseModel):
    title: str
    evidence: str
    severity: Literal["low", "med", "high"] = "low"


class _Strength(BaseModel):
    title: str
    evidence: str


class _ImprovementAction(BaseModel):
    action: str
    why: str
    how: str
    priority: int = Field(ge=1, le=10, default=5)


class _ChecklistItem(BaseModel):
    item: str
    done: bool = False


class _ArchitectureReview(BaseModel):
    overview: str = "Not assessed."
    backend: str = "Not assessed."
    frontend: str = "Not assessed."
    database: str = "Not assessed."
    security: str = "Not assessed."


class _ImplementationReview(BaseModel):
    features_built: List[str] = Field(default_factory=list)
    technical_quality: str = "Not assessed."
    integration_quality: str = "Not assessed."


class _EvaluationReview(BaseModel):
    testing_present: str = "Not assessed."
    limitations: str = "Not assessed."
    academic_quality: str = "Not assessed."


class _ConfidenceSummary(BaseModel):
    mode: Literal["normal", "restricted"] = "normal"
    overall: float = Field(default=0.0, ge=0.0, le=1.0)


class _ModelAgreement(BaseModel):
    ml_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    llm_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    final_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class _Safety(BaseModel):
    needs_review: bool = False
    reason: str = ""


class _RagMeta(BaseModel):
    enabled: bool = False
    confidence_score: float = 0.0
    confidence_label: str = "low"
    safe_review: bool = False
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    trace: Optional[Dict[str, Any]] = None


class _RubricRow(BaseModel):
    criterion: str
    band: str
    justification: str


class _ModerationNote(BaseModel):
    risk: str
    note: str


# ===========================================================================
# Phase 10 request schemas
# ===========================================================================

class Phase10GenerateIn(BaseModel):
    """Common fields for both student and professor generation requests."""

    file_id: str = Field(..., min_length=1)
    force: bool = False


# ===========================================================================
# Phase 7-compatible response schemas (backward compat — DO NOT CHANGE)
# ===========================================================================

class Phase10StudentReport(BaseModel):
    """Full student report shape returned by Phase 10 (same shape as Phase 7)."""

    model_config = {"protected_namespaces": ()}

    summary: str
    issues: List[_Issue] = Field(default_factory=list)
    strengths: List[_Strength] = Field(default_factory=list)
    architecture_review: _ArchitectureReview = Field(default_factory=_ArchitectureReview)
    implementation_review: _ImplementationReview = Field(default_factory=_ImplementationReview)
    evaluation_review: _EvaluationReview = Field(default_factory=_EvaluationReview)
    improvement_plan: List[_ImprovementAction] = Field(default_factory=list)
    checklist: List[_ChecklistItem] = Field(default_factory=list)
    confidence: _ConfidenceSummary = Field(default_factory=_ConfidenceSummary)
    model_agreement: _ModelAgreement = Field(default_factory=_ModelAgreement)
    safety: _Safety = Field(default_factory=_Safety)
    rag_meta: Optional[_RagMeta] = None


class Phase10ProfessorReport(BaseModel):
    """Full professor report shape returned by Phase 10."""

    rubric_breakdown: List[_RubricRow] = Field(default_factory=list)
    feedback_explanation: str = ""
    moderation_notes: List[_ModerationNote] = Field(default_factory=list)
    safety: _Safety = Field(default_factory=_Safety)
    rag_meta: Optional[_RagMeta] = None


class Phase10StudentResponse(BaseModel):
    """Envelope returned by POST /phase10/student/generate."""

    cached: bool = False
    request_id: str
    ml: Dict[str, Any] = Field(default_factory=dict)
    report: Phase10StudentReport
    rag_meta: Dict[str, Any] = Field(default_factory=dict)
    stored: Optional[Dict[str, Any]] = None


class Phase10ProfessorResponse(BaseModel):
    """Envelope returned by POST /phase10/professor/generate."""

    cached: bool = False
    request_id: str
    ml: Dict[str, Any] = Field(default_factory=dict)
    report: Phase10ProfessorReport
    rag_meta: Dict[str, Any] = Field(default_factory=dict)
    stored: Optional[Dict[str, Any]] = None


class LangChainRagMeta(BaseModel):
    """Public-safe RAG metadata for LangChain generation endpoints."""

    enabled: bool = False
    confidence_score: float = 0.0
    confidence_label: str = "low"
    safe_review: bool = False
    chunk_count: int = 0
    weak_retrieval: bool = False
    citations: List[Dict[str, Any]] = Field(default_factory=list)


class LangChainStoredSummary(BaseModel):
    """Minimal stored-row summary safe to return from LangChain routers."""

    id: str = ""
    file_id: str = ""
    submission_id: Optional[str] = None
    role: str = ""
    needs_review: bool = False
    created_at: Optional[str] = None


class LangChainResponseMeta(BaseModel):
    """Public metadata for a LangChain generation run."""

    role: Literal["student", "professor"]
    needs_review: bool = False
    model_used: str = ""
    fallback_used: bool = False
    decision_source: str = "llm"
    execution_mode: str = "normal"
    discrepancy_flag: Optional[bool] = None


class LangChainStudentResponse(BaseModel):
    """Public response envelope for POST /api/langchain/student/generate."""

    cached: bool = False
    request_id: str
    ml: Dict[str, Any] = Field(default_factory=dict)
    report: Phase10StudentReport
    rag_meta: LangChainRagMeta = Field(default_factory=LangChainRagMeta)
    meta: LangChainResponseMeta
    stored: Optional[LangChainStoredSummary] = None


class LangChainProfessorResponse(BaseModel):
    """Public response envelope for POST /api/langchain/professor/generate."""

    cached: bool = False
    request_id: str
    ml: Dict[str, Any] = Field(default_factory=dict)
    report: Phase10ProfessorReport
    rag_meta: LangChainRagMeta = Field(default_factory=LangChainRagMeta)
    meta: LangChainResponseMeta
    stored: Optional[LangChainStoredSummary] = None


# ===========================================================================
# Phase 10-native sub-schemas
# ===========================================================================

class CitationOut(BaseModel):
    """A single RAG-grounded citation surfaced in the Phase 10-native report."""

    source: str = ""
    snippet: str = ""
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)


class IssueOut(BaseModel):
    """A single identified issue in a student submission."""

    title: str = ""
    description: str = ""
    severity: Literal["low", "med", "high"] = "low"


class StudentChecklistItem(BaseModel):
    """One item in a student's improvement checklist."""

    item: str = ""
    done: bool = False


class RubricBandOut(BaseModel):
    """One rubric criterion row in a professor evaluation."""

    criterion: str = ""
    band: str = ""
    justification: str = ""


# ===========================================================================
# Phase 10-native output schemas
# ===========================================================================

class Phase10StudentOut(BaseModel):
    """
    Streamlined student report shape native to Phase 10.

    All list fields default to empty so a partial LLM response never raises
    ValidationError during pipeline assembly.  ``safe_mode`` and
    ``fallback_used`` are surfaced as top-level booleans rather than being
    buried in a nested ``safety`` object.
    """

    model_config = {"protected_namespaces": ()}

    summary: str = ""
    strengths: List[str] = Field(default_factory=list)
    issues: List[IssueOut] = Field(default_factory=list)
    improvement_plan: List[str] = Field(default_factory=list)
    checklist: List[StudentChecklistItem] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    citations: List[CitationOut] = Field(default_factory=list)
    safe_mode: bool = False
    fallback_used: bool = False


class Phase10ProfessorOut(BaseModel):
    """
    Streamlined professor report shape native to Phase 10.

    ``moderation_notes`` and ``feedback_reasoning`` are plain ``list[str]``
    rather than nested typed objects so the LLM can return bullet-point arrays
    directly without additional coercion.
    """

    rubric_breakdown: List[RubricBandOut] = Field(default_factory=list)
    feedback_reasoning: List[str] = Field(default_factory=list)
    moderation_notes: List[str] = Field(default_factory=list)
    needs_review: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    citations: List[CitationOut] = Field(default_factory=list)
    safe_mode: bool = False
    fallback_used: bool = False
    discrepancy_flag: bool = False
