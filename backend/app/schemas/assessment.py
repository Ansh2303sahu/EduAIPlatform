"""Phase 14 — Multi-Model Assessment Layer: Pydantic schemas.

Provider responsibilities
-------------------------
  OpenAI GPT-4o       → primary rubric assessment (grades, issues, strengths)
  Claude Sonnet       → consistency review (logic coherence, hallucination check)
  Gemini Pro          → conditional multimodal extraction (diagrams, figures)

Schema groups
-------------
  Outbound (backend → n8n)  : RubricContextOut
  Inbound  (n8n → backend)  : AssessmentResultIn, AssessmentEscalateIn, AssessmentMetricIn
  Internal                  : OpenAIAssessmentResult, ClaudeReviewResult,
                              GeminiExtractionResult, AssessmentGateDecision,
                              MultiModelAssessmentResult, AssessmentAuditRecord,
                              AssessmentEscalationRecord
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Sub-schemas reused across providers
# ---------------------------------------------------------------------------

class RubricCriterion(BaseModel):
    criterion: str = Field(..., min_length=1, max_length=200)
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    description: str = Field(default="", max_length=1000)


class RubricScore(BaseModel):
    criterion: str
    band: str  # e.g. "Distinction", "Merit", "Pass", "Fail"
    score: float = Field(ge=0.0, le=100.0)
    justification: str = Field(default="", max_length=2000)
    evidence_quotes: list[str] = Field(default_factory=list)


class AssessmentIssue(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    evidence: str = Field(..., min_length=1, max_length=2000)
    severity: Literal["low", "med", "high"]


class AssessmentStrength(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    evidence: str = Field(..., min_length=1, max_length=2000)


class ImprovementAction(BaseModel):
    action: str = Field(..., min_length=1, max_length=300)
    why: str = Field(..., min_length=1, max_length=800)
    how: str = Field(..., min_length=1, max_length=800)
    priority: int = Field(ge=1, le=10)


class ProviderUsageStats(BaseModel):
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    cost_usd: float = Field(default=0.0, ge=0.0)


# ---------------------------------------------------------------------------
# Rubric context — backend → n8n so providers know what to assess
# ---------------------------------------------------------------------------

class RubricContextOut(BaseModel):
    """Payload returned by POST /internal/assessment/rubric-context.

    n8n passes this to each provider as part of the system prompt context.
    It intentionally excludes raw file content (which is too large for HTTP).
    Only extracted-text summaries and rubric metadata are included.
    """

    file_id: str
    submission_id: str = ""
    user_id: str
    role: Literal["student", "professor"]
    analysis_type: str = ""

    # Rubric definition fetched from professor's rubric store
    rubric_criteria: list[RubricCriterion] = Field(default_factory=list)
    rubric_name: str = ""
    rubric_version: str = "1.0"

    # Extracted content summary from Phase 12 ingestion (safe to pass over HTTP)
    content_summary: str = Field(default="", max_length=8000)
    word_count: int = 0
    language: str = "en"

    # ML signals from Phase 12 (passed through for provider context)
    ml_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    ml_prediction_label: str = ""

    # Whether the submission contains non-text media (triggers Gemini)
    has_images: bool = False
    has_tables: bool = False
    has_code_blocks: bool = False
    media_file_ids: list[str] = Field(default_factory=list)

    # Phase 12 single-model draft (used by Claude as a review baseline)
    draft_summary: str = Field(default="", max_length=4000)
    draft_issues: list[dict[str, Any]] = Field(default_factory=list)

    # Metadata
    schema_version: str = "14.1"
    context_built_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# OpenAI GPT-4o — primary assessor result
# ---------------------------------------------------------------------------

class OpenAIAssessmentResult(BaseModel):
    """Structured output from the OpenAI GPT-4o primary assessment call.

    n8n extracts this from the OpenAI JSON-mode response and forwards it
    to the backend inside AssessmentResultIn.openai_result.
    """

    model_config = {"protected_namespaces": ()}

    model_id: str = Field(default="gpt-4o")
    rubric_scores: list[RubricScore] = Field(default_factory=list)
    overall_grade: str = ""  # e.g. "Merit", "72/100"
    overall_score: float = Field(default=0.0, ge=0.0, le=100.0)
    summary: str = Field(default="", max_length=4000)
    strengths: list[AssessmentStrength] = Field(default_factory=list)
    issues: list[AssessmentIssue] = Field(default_factory=list)
    improvement_plan: list[ImprovementAction] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_human_review: bool = False
    safety_flags: list[str] = Field(default_factory=list)
    usage: ProviderUsageStats = Field(
        default_factory=lambda: ProviderUsageStats(model="gpt-4o")
    )
    raw_response_hash: str = ""  # SHA-256 of the raw JSON for audit
    assessed_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Claude Sonnet — consistency reviewer result
# ---------------------------------------------------------------------------

class ClaudeCorrection(BaseModel):
    field_path: str  # e.g. "rubric_scores[0].band"
    original_value: Any
    suggested_value: Any
    reason: str


class ClaudeReviewResult(BaseModel):
    """Output from the Claude Sonnet consistency-review pass.

    Claude receives the OpenAI result + rubric context and checks for:
      - Logical inconsistencies (score vs. justification mismatch)
      - Unsupported claims not grounded in the submission text
      - Internal contradictions between strengths/issues
      - Potential hallucinations in evidence_quotes
    """

    model_config = {"protected_namespaces": ()}

    model_id: str = Field(default="claude-sonnet-4-6")
    consistent: bool = True
    reviewer_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    concerns: list[str] = Field(default_factory=list)
    corrections: list[ClaudeCorrection] = Field(default_factory=list)
    flagged_for_hitl: bool = False
    hitl_reason: str = ""
    overall_verdict: Literal["approved", "needs_correction", "escalate"] = "approved"
    usage: ProviderUsageStats = Field(
        default_factory=lambda: ProviderUsageStats(model="claude-sonnet-4-6")
    )
    reviewed_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Gemini Pro — conditional multimodal extractor result
# ---------------------------------------------------------------------------

class FigureAnalysis(BaseModel):
    figure_id: str
    figure_type: Literal["diagram", "chart", "table", "image", "code", "other"]
    description: str = Field(default="", max_length=2000)
    assessment_relevance: str = Field(default="", max_length=1000)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)


class GeminiExtractionResult(BaseModel):
    """Output from the Gemini 1.5 Pro multimodal extraction pass.

    Only triggered when the submission contains non-text media.
    Provides figure/diagram analysis that OpenAI text-only cannot access.
    """

    model_config = {"protected_namespaces": ()}

    model_id: str = Field(default="gemini-1.5-pro")
    multimodal_used: bool = False
    skip_reason: str = ""  # populated when multimodal_used=False

    figures: list[FigureAnalysis] = Field(default_factory=list)
    additional_context: str = Field(default="", max_length=4000)
    media_quality_overall: float = Field(default=0.0, ge=0.0, le=1.0)

    # Additional issues/strengths discovered from visual content
    visual_issues: list[AssessmentIssue] = Field(default_factory=list)
    visual_strengths: list[AssessmentStrength] = Field(default_factory=list)

    usage: ProviderUsageStats = Field(
        default_factory=lambda: ProviderUsageStats(model="gemini-1.5-pro")
    )
    extracted_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Gate decision — computed by n8n after all providers complete
# ---------------------------------------------------------------------------

class AssessmentGateDecision(BaseModel):
    """Gating result computed by n8n before persisting the final assessment.

    Gate criteria (any → escalate):
      - OpenAI confidence < 0.40
      - Claude flags inconsistency (overall_verdict == "escalate")
      - OpenAI needs_human_review == True
      - safety_flags non-empty
    """

    pass_gate: bool
    escalate: bool
    hitl_required: bool
    escalation_reasons: list[str] = Field(default_factory=list)
    final_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_sources: dict[str, float] = Field(default_factory=dict)
    decided_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Merged final result — what the backend ultimately persists
# ---------------------------------------------------------------------------

class MultiModelAssessmentResult(BaseModel):
    """Canonical merged assessment result after all providers + gate pass.

    Stored as the Phase 14 ai_assessments record.  The gate logic in n8n
    applies Claude corrections to the OpenAI result before passing this here.
    """

    assessment_id: str = Field(default_factory=lambda: str(uuid4()))
    file_id: str
    submission_id: str = ""
    user_id: str
    role: Literal["student", "professor"]

    # Merged report fields (OpenAI base + Claude corrections applied)
    overall_grade: str = ""
    overall_score: float = Field(default=0.0, ge=0.0, le=100.0)
    summary: str = ""
    rubric_scores: list[RubricScore] = Field(default_factory=list)
    strengths: list[AssessmentStrength] = Field(default_factory=list)
    issues: list[AssessmentIssue] = Field(default_factory=list)
    improvement_plan: list[ImprovementAction] = Field(default_factory=list)

    # Multimodal additions (empty when Gemini skipped)
    figures: list[FigureAnalysis] = Field(default_factory=list)
    visual_context: str = ""

    # Gate output
    gate: AssessmentGateDecision

    # Provider usage for billing/audit
    provider_stats: dict[str, ProviderUsageStats] = Field(default_factory=dict)
    total_cost_usd: float = Field(default=0.0, ge=0.0)

    # Provenance
    workflow_version: str = "v2"
    schema_version: str = "14.1"
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Audit record
# ---------------------------------------------------------------------------

class AssessmentAuditRecord(BaseModel):
    """Immutable audit trail written for every multi-model assessment attempt.

    Written regardless of success/failure so we can reconstruct what happened.
    """

    audit_id: str = Field(default_factory=lambda: str(uuid4()))
    assessment_id: str = ""
    file_id: str
    submission_id: str = ""
    user_id: str
    role: str

    # Providers invoked
    openai_invoked: bool = False
    claude_invoked: bool = False
    gemini_invoked: bool = False

    # Outcome
    gate_passed: bool = False
    hitl_triggered: bool = False
    final_status: Literal["completed", "escalated", "failed", "hitl_pending"] = "failed"

    # Provider latencies (ms)
    openai_latency_ms: int = 0
    claude_latency_ms: int = 0
    gemini_latency_ms: int = 0
    total_latency_ms: int = 0

    # Token totals
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_usd: float = 0.0

    # Workflow metadata
    workflow_version: str = "v2"
    n8n_execution_id: str = ""
    correlation_id: str = ""

    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Escalation record
# ---------------------------------------------------------------------------

class AssessmentEscalationRecord(BaseModel):
    """Written when the gate triggers HITL escalation."""

    escalation_id: str = Field(default_factory=lambda: str(uuid4()))
    assessment_id: str = ""
    file_id: str
    submission_id: str = ""
    user_id: str
    role: str

    reasons: list[str] = Field(default_factory=list)
    openai_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    claude_verdict: str = ""
    severity: Literal["low", "medium", "high", "critical"] = "medium"

    status: Literal["pending", "approved", "rejected", "timeout"] = "pending"
    decided_by: str = ""
    decision_notes: str = ""
    resolved_at: str = ""

    n8n_resume_url: str = ""
    n8n_execution_id: str = ""
    workflow_version: str = "v2"

    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Inbound payloads — what n8n POSTs to /internal/assessment/*
# ---------------------------------------------------------------------------

class RubricContextIn(BaseModel):
    """Request body for POST /internal/assessment/rubric-context."""

    file_id: str = Field(..., min_length=1)
    submission_id: str = ""
    user_id: str = Field(..., min_length=1)
    role: Literal["student", "professor"]
    include_draft: bool = True  # whether to include Phase 12 draft output


class AssessmentResultIn(BaseModel):
    """Request body for POST /internal/assessment/submit-result.

    n8n POSTs this after all providers complete and the gate passes.
    """

    event_id: str = Field(..., min_length=1, description="Idempotency key")
    n8n_execution_id: str = ""
    correlation_id: str = ""

    file_id: str = Field(..., min_length=1)
    submission_id: str = ""
    user_id: str = Field(..., min_length=1)
    role: Literal["student", "professor"]

    openai_result: OpenAIAssessmentResult
    claude_review: ClaudeReviewResult
    gemini_extraction: GeminiExtractionResult | None = None
    gate_decision: AssessmentGateDecision

    # Merged final report (built by n8n merge Code node)
    final_report: dict[str, Any] = Field(default_factory=dict)

    workflow_version: str = "v2"
    schema_version: str = "14.1"


class AssessmentEscalateIn(BaseModel):
    """Request body for POST /internal/assessment/escalate."""

    event_id: str = Field(..., min_length=1)
    n8n_execution_id: str = ""
    n8n_resume_url: str = ""
    correlation_id: str = ""

    file_id: str = Field(..., min_length=1)
    submission_id: str = ""
    user_id: str = Field(..., min_length=1)
    role: str = ""

    reasons: list[str] = Field(default_factory=list)
    openai_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    claude_verdict: str = ""
    severity: Literal["low", "medium", "high", "critical"] = "medium"


class AssessmentMetricIn(BaseModel):
    """Request body for POST /internal/assessment/metric."""

    metric: str = Field(..., min_length=1, max_length=128, pattern=r"^[a-z0-9_.]+$")
    value: int = Field(default=1, ge=1)
    labels: dict[str, str] = Field(default_factory=dict)


class AssessmentValidateIn(BaseModel):
    """Optional request body for POST /internal/assessment/validate-result.

    n8n can call this before submit-result to pre-validate the merged payload
    without persisting it, catching schema errors early.
    """

    openai_result: OpenAIAssessmentResult
    claude_review: ClaudeReviewResult
    gemini_extraction: GeminiExtractionResult | None = None
    gate_decision: AssessmentGateDecision
    final_report: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Event payload — emitted by Phase 12 generation node to trigger Phase 14
# ---------------------------------------------------------------------------

class AssessmentRequestedPayload(BaseModel):
    """Payload for the ai.assessment.requested event.

    Phase 12 generation node emits this after producing the draft report.
    n8n picks it up and starts the multi-model assessment workflow.
    """

    file_id: str
    user_id: str
    role: str
    submission_id: str = ""
    draft_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    has_images: bool = False
    has_tables: bool = False
    has_code_blocks: bool = False
    pipeline: str = "phase12_langgraph"
    workflow_version: str = "v2"
