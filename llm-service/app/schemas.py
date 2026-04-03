from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal, List, Optional, Any, Dict

Severity = Literal["low", "med", "high"]

# -----------------------------
# Student feedback structures
# -----------------------------

class Issue(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    evidence: str = Field(min_length=1, max_length=2000)
    severity: Severity


class Strength(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    evidence: str = Field(min_length=1, max_length=2000)


class ImprovementAction(BaseModel):
    action: str = Field(min_length=1, max_length=300)
    why: str = Field(min_length=1, max_length=800)
    how: str = Field(min_length=1, max_length=800)
    priority: int = Field(ge=1, le=10)


class ChecklistItem(BaseModel):
    item: str = Field(min_length=1, max_length=200)
    done: bool = False


class ArchitectureReview(BaseModel):
    overview: str = Field(default="Not assessed.", min_length=1, max_length=1200)
    backend: str = Field(default="Not assessed.", min_length=1, max_length=1200)
    frontend: str = Field(default="Not assessed.", min_length=1, max_length=1200)
    database: str = Field(default="Not assessed.", min_length=1, max_length=1200)
    security: str = Field(default="Not assessed.", min_length=1, max_length=1200)


class ImplementationReview(BaseModel):
    features_built: List[str] = Field(default_factory=list)
    technical_quality: str = Field(default="Not assessed.", min_length=1, max_length=1200)
    integration_quality: str = Field(default="Not assessed.", min_length=1, max_length=1200)


class EvaluationReview(BaseModel):
    testing_present: str = Field(default="Not assessed.", min_length=1, max_length=1200)
    limitations: str = Field(default="Not assessed.", min_length=1, max_length=1200)
    academic_quality: str = Field(default="Not assessed.", min_length=1, max_length=1200)


class ConfidenceSummary(BaseModel):
    mode: Literal["normal", "restricted"] = "normal"
    overall: float = Field(default=0.0, ge=0.0, le=1.0)


class RAGCitation(BaseModel):
    index: Optional[int] = None
    title: str
    section: Optional[str] = None
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    category: Optional[str] = None
    score: Optional[float] = None


class RAGMetaOut(BaseModel):
    enabled: bool = False
    confidence_score: float = 0.0
    confidence_label: str = "low"
    safe_review: bool = False
    citations: List[RAGCitation] = Field(default_factory=list)
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    trace: Optional[Dict[str, Any]] = None


class ModelAgreement(BaseModel):
    ml_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    llm_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    final_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class Safety(BaseModel):
    needs_review: bool = False
    reason: str = ""


class StudentReportOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    summary: str = Field(min_length=1, max_length=1200)
    issues: List[Issue] = Field(default_factory=list)
    strengths: List[Strength] = Field(default_factory=list)
    architecture_review: ArchitectureReview = Field(default_factory=ArchitectureReview)
    implementation_review: ImplementationReview = Field(default_factory=ImplementationReview)
    evaluation_review: EvaluationReview = Field(default_factory=EvaluationReview)
    improvement_plan: List[ImprovementAction] = Field(default_factory=list)
    checklist: List[ChecklistItem] = Field(default_factory=list)
    confidence: ConfidenceSummary = Field(default_factory=ConfidenceSummary)
    model_agreement: ModelAgreement = Field(default_factory=ModelAgreement)
    safety: Safety = Field(default_factory=Safety)
    rag_meta: Optional[RAGMetaOut] = None


# -----------------------------
# Professor feedback structures
# -----------------------------

class RubricRow(BaseModel):
    criterion: str = Field(min_length=1, max_length=200)
    band: str = Field(min_length=1, max_length=80)
    justification: str = Field(min_length=1, max_length=1200)


class ModerationNote(BaseModel):
    risk: str = Field(min_length=1, max_length=120)
    note: str = Field(min_length=1, max_length=800)


class ProfessorReportOut(BaseModel):
    rubric_breakdown: List[RubricRow]
    feedback_explanation: str = Field(min_length=1, max_length=1600)
    moderation_notes: List[ModerationNote]
    safety: Safety
    rag_meta: Optional[RAGMetaOut] = None

# -----------------------------
# Ingestion bundle
# -----------------------------

class IngestionBundle(BaseModel):
    text_content: str = ""
    ocr_text: str = ""
    audio_transcript: str = ""
    tables_json: Optional[Dict[str, Any]] = None


# -----------------------------
# ML signals
# -----------------------------

class MLStudentSignals(BaseModel):
    feedback_category: str
    quality_band: Literal["low", "med", "high"]
    confidence_0_to_4: int = Field(ge=0, le=4)


class MLProfessorSignals(BaseModel):
    rubric_band: str
    argument_depth: Literal["low", "med", "high"]
    moderation_consistency: Literal["low", "med", "high"]


# -----------------------------
# RAG grounding structures
# -----------------------------

class RAGPayload(BaseModel):
    enabled: bool = True
    audience: Optional[str] = None
    query: Optional[str] = None
    context: Optional[str] = None
    citations: Optional[List[RAGCitation]] = None
    confidence_score: Optional[float] = None
    confidence_label: Optional[str] = None
    safe_review: Optional[bool] = None
    trace: Optional[Dict[str, Any]] = None
    instruction: Optional[str] = None


# -----------------------------
# Student LLM request
# -----------------------------

class StudentReportIn(BaseModel):
    submission_id: str
    ingestion: IngestionBundle
    ml: MLStudentSignals
    mode: Optional[str] = None
    safety_flags: Optional[Dict[str, Any]] = None
    query: Optional[str] = None
    top_k: Optional[int] = None
    analysis_type: Optional[str] = None
    analysis_focus: Optional[List[str]] = None
    submission_type: Optional[str] = None
    grounding_retrieved_chunks: Optional[List[Dict[str, Any]]] = None

    # RAG fields (optional)
    rag: Optional[RAGPayload] = None
    grounding_context: Optional[str] = None
    grounding_citations: Optional[List[RAGCitation]] = None
    grounding_instruction: Optional[str] = None

    retrieval_confidence_score: Optional[float] = None
    retrieval_confidence_label: Optional[str] = None
    retrieval_safe_review: Optional[bool] = None
    retrieval_trace: Optional[Dict[str, Any]] = None


# -----------------------------
# Professor LLM request
# -----------------------------

class ProfessorReportIn(BaseModel):
    submission_id: str
    ingestion: IngestionBundle
    ml: MLProfessorSignals
    mode: Optional[str] = None
    safety_flags: Optional[Dict[str, Any]] = None
    query: Optional[str] = None
    top_k: Optional[int] = None
    analysis_type: Optional[str] = None
    analysis_focus: Optional[List[str]] = None
    submission_type: Optional[str] = None
    grounding_retrieved_chunks: Optional[List[Dict[str, Any]]] = None

    # RAG fields (optional)
    rag: Optional[RAGPayload] = None
    grounding_context: Optional[str] = None
    grounding_citations: Optional[List[RAGCitation]] = None
    grounding_instruction: Optional[str] = None

    retrieval_confidence_score: Optional[float] = None
    retrieval_confidence_label: Optional[str] = None
    retrieval_safe_review: Optional[bool] = None
    retrieval_trace: Optional[Dict[str, Any]] = None







