from pydantic import BaseModel, Field
from typing import Literal, List, Optional, Any, Dict

Severity = Literal["low", "med", "high"]

class Issue(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    evidence: str = Field(min_length=1, max_length=2000)
    severity: Severity

class ImprovementAction(BaseModel):
    action: str = Field(min_length=1, max_length=300)
    why: str = Field(min_length=1, max_length=800)
    how: str = Field(min_length=1, max_length=800)
    priority: int = Field(ge=1, le=10)

class ChecklistItem(BaseModel):
    item: str = Field(min_length=1, max_length=200)
    done: bool = False

class ModelAgreement(BaseModel):
    ml_confidence: float = Field(ge=0.0, le=1.0)
    llm_confidence: float = Field(ge=0.0, le=1.0)
    final_confidence: float = Field(ge=0.0, le=1.0)

class Safety(BaseModel):
    needs_review: bool = False
    reason: str = ""

class StudentReportOut(BaseModel):
    summary: str = Field(min_length=1, max_length=1200)
    issues: List[Issue]
    improvement_plan: List[ImprovementAction]
    checklist: List[ChecklistItem]
    model_agreement: ModelAgreement
    safety: Safety

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

class IngestionBundle(BaseModel):
    text_content: str = ""
    ocr_text: str = ""
    audio_transcript: str = ""
    tables_json: Optional[Dict[str, Any]] = None

class MLStudentSignals(BaseModel):
    feedback_category: str
    quality_band: Literal["low", "med", "high"]
    confidence_0_to_4: int = Field(ge=0, le=4)

class MLProfessorSignals(BaseModel):
    rubric_band: str
    argument_depth: Literal["low", "med", "high"]
    moderation_consistency: Literal["low", "med", "high"]

class StudentReportIn(BaseModel):
    submission_id: str
    ingestion: IngestionBundle
    ml: MLStudentSignals

class ProfessorReportIn(BaseModel):
    submission_id: str
    ingestion: IngestionBundle
    ml: MLProfessorSignals