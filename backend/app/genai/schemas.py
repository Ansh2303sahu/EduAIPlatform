"""Public and internal schemas for the Phase 15/16 generative AI layer."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


ConfidenceBand = Literal["low", "medium", "high"]
ReportType = Literal["student", "professor"]


class EvidenceReference(BaseModel):
    """Compact evidence reference grounded in RAG output or ML evidence."""

    source: str = ""
    chunk_id: str = ""
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    excerpt: str = ""
    category: str = ""


class StructuredFeedbackBlock(BaseModel):
    """Shared structured feedback fields required across Phase 15/16 outputs."""

    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_references: list[EvidenceReference] = Field(default_factory=list)


class ImprovementAction(BaseModel):
    """A concrete improvement action for the learner."""

    title: str
    rationale: str
    steps: list[str] = Field(default_factory=list)
    priority: Literal["low", "medium", "high"] = "medium"


class LearningMilestone(BaseModel):
    """A learning-path milestone."""

    title: str
    objective: str
    activities: list[str] = Field(default_factory=list)


class ImprovementPlan(StructuredFeedbackBlock):
    """A structured improvement plan for a learner or reviewed work."""

    actions: list[ImprovementAction] = Field(default_factory=list)
    timeline: str = ""


class LearningPath(StructuredFeedbackBlock):
    """A structured learning path tailored to the submission."""

    milestones: list[LearningMilestone] = Field(default_factory=list)
    recommended_practice: list[str] = Field(default_factory=list)


class ReportConfidence(BaseModel):
    """Calibrated confidence summary for UI and audit surfaces."""

    score: float = Field(default=0.0, ge=0.0, le=1.0)
    band: ConfidenceBand = "low"
    rationale: str = ""


class SafetySummary(BaseModel):
    """Conservative safety metadata retained for compatibility and audit."""

    needs_review: bool = False
    reason: str = ""


class FeedbackCard(BaseModel):
    title: str = ""
    detail: str = ""
    severity: Literal["low", "medium", "high"] | None = None


class SectionFeedback(BaseModel):
    section_name: str = ""
    what_works: str = ""
    what_needs_improvement: str = ""
    recommended_fix: str = ""


class PriorityIssue(BaseModel):
    title: str = ""
    why_it_matters: str = ""
    how_to_fix_it: str = ""


class SectionObservation(BaseModel):
    section_name: str = ""
    observation: str = ""
    concern: str = ""
    next_step: str = ""


class StudentReport(StructuredFeedbackBlock):
    """Structured student-facing report."""

    report_type: Literal["student"] = "student"
    summary: str = ""
    overall_judgment: str = ""
    issues: list[dict[str, Any]] = Field(default_factory=list)
    strength_cards: list[FeedbackCard] = Field(default_factory=list)
    weakness_cards: list[FeedbackCard] = Field(default_factory=list)
    section_feedback: list[SectionFeedback] = Field(default_factory=list)
    priority_issue: PriorityIssue = Field(default_factory=PriorityIssue)
    improvement_plan: ImprovementPlan = Field(default_factory=ImprovementPlan)
    learning_path: LearningPath = Field(default_factory=LearningPath)
    confidence: ReportConfidence = Field(default_factory=ReportConfidence)
    confidence_explanation: str = ""
    evidence_coverage: str = ""
    reasoning_summary: list[str] = Field(default_factory=list)
    counterfactual_explanation: str = ""
    safety: SafetySummary = Field(default_factory=SafetySummary)


class ProfessorModerationReport(StructuredFeedbackBlock):
    """Structured professor/moderation report."""

    report_type: Literal["professor"] = "professor"
    summary: str = ""
    evaluator_overview: str = ""
    feedback_explanation: str = ""
    moderation_notes: list[str] = Field(default_factory=list)
    rubric_alignment: list[str] = Field(default_factory=list)
    strength_cards: list[FeedbackCard] = Field(default_factory=list)
    concern_cards: list[FeedbackCard] = Field(default_factory=list)
    section_observations: list[SectionObservation] = Field(default_factory=list)
    marking_considerations: list[str] = Field(default_factory=list)
    action_recommendations: list[str] = Field(default_factory=list)
    confidence: ReportConfidence = Field(default_factory=ReportConfidence)
    confidence_explanation: str = ""
    evidence_coverage: str = ""
    reasoning_summary: list[str] = Field(default_factory=list)
    counterfactual_explanation: str = ""
    safety: SafetySummary = Field(default_factory=SafetySummary)


class FeatureContribution(BaseModel):
    """Approximate feature importance entry."""

    feature: str
    contribution: float = Field(ge=0.0, le=1.0)
    direction: Literal["positive", "negative", "mixed"] = "mixed"
    explanation: str = ""


class ReasoningStep(BaseModel):
    """Compact reasoning step for the explanation tab."""

    step: str
    detail: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class RagTraceEntry(BaseModel):
    """Compact RAG trace entry surfaced to UI and audit tooling."""

    source: str = ""
    chunk_id: str = ""
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    category: str = ""


class ExplanationTab(BaseModel):
    """Explanation tab contract for the frontend."""

    reasoning_steps: list[ReasoningStep] = Field(default_factory=list)
    feature_importance: list[FeatureContribution] = Field(default_factory=list)
    confidence_band: ConfidenceBand = "low"
    confidence_summary: str = ""
    hybrid_summary: str = ""
    counterfactual_explanation: str = ""


class SourcesTab(BaseModel):
    """Sources tab contract for the frontend."""

    evidence_references: list[EvidenceReference] = Field(default_factory=list)
    rag_trace: list[RagTraceEntry] = Field(default_factory=list)
    source_summary: str = ""


class BiasSignal(BaseModel):
    """Bias/trust signal for the fairness tab."""

    name: str
    severity: Literal["low", "medium", "high"] = "low"
    detail: str = ""


class MultiModelAgreement(BaseModel):
    """Agreement summary across primary and validator models."""

    primary_model: str = ""
    validator_model: str = ""
    agreement_score: float = Field(default=0.0, ge=0.0, le=1.0)
    agreement_band: ConfidenceBand = "low"
    summary: str = ""


class ConsistencyFinding(BaseModel):
    """Consistency checker output."""

    issue: str
    severity: Literal["low", "medium", "high"] = "low"
    detail: str = ""


class FairnessTab(BaseModel):
    """Fairness/trust tab contract for the frontend."""

    bias_signals: list[BiasSignal] = Field(default_factory=list)
    multi_model_agreement: MultiModelAgreement = Field(default_factory=MultiModelAgreement)
    counterfactual_explanation: str = ""
    consistency_findings: list[ConsistencyFinding] = Field(default_factory=list)


class AuditTab(BaseModel):
    """Compact audit/versioning payload."""

    model_config = {"protected_namespaces": ()}

    request_id: str = ""
    execution_id: str = ""
    role: ReportType
    graph_version: str = ""
    prompt_version: str = ""
    output_version: str = ""
    model_version: str = ""
    validator_model_version: str = ""
    timestamp: datetime
    final_status: str = ""
    pdf_base64: str | None = None


class StoredSummary(BaseModel):
    """Stored-row summary returned to the client."""

    id: str = ""
    file_id: str = ""
    submission_id: str | None = None
    role: str = ""
    created_at: str | None = None
    needs_review: bool = False


class PredictionTab(BaseModel):
    """Prediction tab contract for the frontend."""

    report_type: ReportType
    report: StudentReport | ProfessorModerationReport


class AIReportResponse(BaseModel):
    """Unified frontend-friendly response contract."""

    request_id: str
    prediction: PredictionTab
    explanation: ExplanationTab
    sources: SourcesTab
    warnings: list[str] = Field(default_factory=list)
    fairness: FairnessTab = Field(default_factory=FairnessTab)
    audit: AuditTab
    stored: StoredSummary | None = None


class AIReportGenerateIn(BaseModel):
    """Request body for student/professor report generation."""

    file_id: str = Field(min_length=1)
    force: bool = False


class ExplainResponse(BaseModel):
    """Response for GET /ai/explain/{file_id}."""

    file_id: str
    role: ReportType
    explanation: ExplanationTab


class CompareResponse(BaseModel):
    """Response for GET /ai/compare/{file_id}."""

    file_id: str
    role: ReportType
    fairness: FairnessTab
    comparison_summary: str = ""
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_band: ConfidenceBand = "low"
    evidence_reference_count: int = Field(default=0, ge=0)
    warning_count: int = Field(default=0, ge=0)


class AuditResponse(BaseModel):
    """Response for GET /ai/audit/{file_id}."""

    file_id: str
    role: ReportType
    audit: AuditTab
    warnings: list[str] = Field(default_factory=list)


class AICheckSummary(BaseModel):
    """Top-level summary shown in the frontend AI check panel."""

    selected_pipeline: str = "none"
    status: str = "not_available"
    created_at: str | None = None
    needs_review: bool = False
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_band: ConfidenceBand | None = None
    report_summary: str = ""


class AICheckLangChain(BaseModel):
    """Compact LangChain execution view for the frontend."""

    model_config = {"protected_namespaces": ()}

    available: bool = False
    pipeline: str = ""
    chain_name: str = ""
    chain_version: str = ""
    prompt_version: str = ""
    schema_version: str = ""
    provider: str = ""
    model_used: str = ""
    primary_model: str = ""
    fallback_model: str = ""
    fallback_used: bool = False
    execution_mode: str = ""
    decision_source: str = ""
    discrepancy_flag: bool | None = None
    retrieval_mode: str = ""
    retrieved_chunk_count: int = 0
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    summary: str = ""


class AICheckLangGraph(BaseModel):
    """Compact LangGraph execution view for the frontend."""

    available: bool = False
    pipeline: str = ""
    graph_name: str = ""
    graph_version: str = ""
    prompt_version: str = ""
    output_version: str = ""
    final_status: str = ""
    safe_mode: bool = False
    total_steps: int = 0
    total_latency_ms: float = 0.0
    node_count: int = 0
    decision_count: int = 0
    failure_count: int = 0
    trace_summary: str = ""
    warnings: list[str] = Field(default_factory=list)


class AICheckGenAI(BaseModel):
    """GenAI prediction summary extracted from a stored Phase 15/16 row."""

    model_config = {"protected_namespaces": ()}

    available: bool = False
    pipeline: str = ""
    model_version: str = ""
    validator_model_version: str = ""
    final_status: str = ""
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_band: ConfidenceBand | None = None
    report_summary: str = ""
    warning_count: int = 0


class AICheckRAG(BaseModel):
    """Grounding and retrieval summary."""

    enabled: bool = False
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_label: str = ""
    citations_count: int = 0
    retrieved_chunk_count: int = 0
    query: str = ""
    collection_name: str = ""
    safe_review: bool = False
    summary: str = ""


class AICheckML(BaseModel):
    """ML calibration summary."""

    model_config = {"protected_namespaces": ()}

    available: bool = False
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    model_names: list[str] = Field(default_factory=list)
    source: str = ""
    summary: str = ""


class AICheckLLM(BaseModel):
    """LLM routing summary."""

    model_config = {"protected_namespaces": ()}

    available: bool = False
    model_used: str = ""
    primary_model: str = ""
    fallback_model: str = ""
    route: str = ""
    source: str = ""


class AICheckMCP(BaseModel):
    """MCP capability and graph-usage summary."""

    enabled: bool = False
    orchestration_enabled: bool = False
    llm_enabled: bool = False
    graph_used: bool = False
    tool_call_count: int = 0
    visible_tools: list[str] = Field(default_factory=list)
    summary: str = ""


class AICheckN8NIntegrations(BaseModel):
    """Per-integration n8n wiring flags."""

    assessment: bool = False
    file_upload: bool = False
    low_confidence: bool = False
    pipeline_failure: bool = False


class AICheckN8N(BaseModel):
    """n8n workflow bridge summary."""

    configured: bool = False
    generation_bridge_active: bool = False
    integrations: AICheckN8NIntegrations = Field(default_factory=AICheckN8NIntegrations)
    summary: str = ""


class AICheckResponse(BaseModel):
    """Unified AI check payload for the frontend."""

    file_id: str
    role: ReportType
    summary: AICheckSummary = Field(default_factory=AICheckSummary)
    langchain: AICheckLangChain = Field(default_factory=AICheckLangChain)
    langgraph: AICheckLangGraph = Field(default_factory=AICheckLangGraph)
    genai: AICheckGenAI = Field(default_factory=AICheckGenAI)
    rag: AICheckRAG = Field(default_factory=AICheckRAG)
    ml: AICheckML = Field(default_factory=AICheckML)
    llm: AICheckLLM = Field(default_factory=AICheckLLM)
    mcp: AICheckMCP = Field(default_factory=AICheckMCP)
    n8n: AICheckN8N = Field(default_factory=AICheckN8N)


class PlanSection(BaseModel):
    """Internal plan section for the planner node."""

    title: str
    objective: str


class CritiquePoint(BaseModel):
    """Internal critique point returned by the critic node."""

    issue: str
    severity: Literal["low", "medium", "high"] = "medium"
    recommendation: str


class CritiqueReport(BaseModel):
    """Internal structured critique for the self-critique loop."""

    overall_quality: Literal["strong", "acceptable", "weak"] = "acceptable"
    concerns: list[CritiquePoint] = Field(default_factory=list)
    tone_bias_flags: list[str] = Field(default_factory=list)
    contradiction_flags: list[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    refinement_required: bool = False
    validator_notes: list[str] = Field(default_factory=list)
