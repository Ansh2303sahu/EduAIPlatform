"""
Phase 11.3 MCP bounded workflow schemas.

The orchestration layer keeps a stable, explicit envelope rather than exposing
an open-ended agent loop. Requests are validated against workflow-specific
payload models, and responses always include per-step MCP envelopes.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter

from app.mcp.schemas import MCPFailureEnvelope, MCPSuccessEnvelope


class WorkflowStepResult(BaseModel):
    model_config = {"extra": "forbid"}

    step_name: str = Field(..., min_length=1, max_length=128)
    tool_name: str = Field(..., min_length=1, max_length=128)
    index: int = Field(..., ge=1, le=20)
    critical: bool
    envelope: MCPSuccessEnvelope | MCPFailureEnvelope


class WorkflowSkippedStep(BaseModel):
    model_config = {"extra": "forbid"}

    step_name: str = Field(..., min_length=1, max_length=128)
    tool_name: str = Field(..., min_length=1, max_length=128)
    reason: str = Field(..., min_length=1, max_length=256)


class WorkflowExplainabilityMeta(BaseModel):
    model_config = {"extra": "forbid"}

    deterministic_plan: bool = True
    policy_controlled: bool = True
    continue_on_non_critical_failure: bool = False
    max_steps_applied: int = Field(..., ge=1, le=20)
    executed_tool_order: list[str] = Field(default_factory=list)
    cache_hit_steps: int = Field(default=0, ge=0)
    llm_used_steps: int = Field(default=0, ge=0)
    fallback_steps: int = Field(default=0, ge=0)
    partial_completion: bool = False
    step_failures: int = Field(default=0, ge=0)
    stopped_reason: str = ""


class StudentWorkflowPayload(BaseModel):
    model_config = {"extra": "forbid"}

    text: str = Field(..., min_length=1, max_length=8_000)
    max_sentences: int = Field(default=3, ge=1, le=10)
    focus_mode: Literal["overview", "methodology", "findings", "conclusion"] = (
        "overview"
    )
    preserve_key_terms: bool = True
    submission_type: Literal["essay", "report", "dissertation", "project_report"] = (
        "essay"
    )
    expected_sections: list[str] = Field(default_factory=list, max_length=20)


class ProfessorWorkflowPayload(BaseModel):
    model_config = {"extra": "forbid"}

    submission_text: str = Field(..., min_length=1, max_length=8_000)
    rubric_criteria: list[str] = Field(..., min_length=1, max_length=20)
    grading_scale: Literal["uk_honours", "us_letter", "percentage", "pass_fail"] = (
        "uk_honours"
    )
    strictness: Literal["lenient", "standard", "strict"] = "standard"
    max_evidence_quotes: int = Field(default=2, ge=0, le=5)
    scores: list[float] = Field(..., min_length=1, max_length=50)
    feedback_items: list[str] = Field(default_factory=list, max_length=50)
    expected_band_labels: list[str] = Field(default_factory=list, max_length=10)
    final_summary: str | None = Field(default=None, max_length=2_000)


class BaseWorkflowRequest(BaseModel):
    model_config = {"extra": "forbid"}

    correlation_id: str | None = Field(default=None, max_length=128)
    file_id: str | None = Field(default=None, max_length=256)
    submission_id: str | None = Field(default=None, max_length=256)
    max_steps: int | None = Field(default=None, ge=1, le=20)
    continue_on_non_critical_failure: bool | None = None


class StudentReviewAssistRequest(BaseWorkflowRequest):
    workflow_name: Literal["student_review_assist"]
    payload: StudentWorkflowPayload


class ProfessorReviewAssistRequest(BaseWorkflowRequest):
    workflow_name: Literal["professor_review_assist"]
    payload: ProfessorWorkflowPayload


WorkflowRequest = Annotated[
    StudentReviewAssistRequest | ProfessorReviewAssistRequest,
    Field(discriminator="workflow_name"),
]

WORKFLOW_REQUEST_ADAPTER = TypeAdapter(WorkflowRequest)


class WorkflowInfoStep(BaseModel):
    model_config = {"extra": "forbid"}

    step_name: str
    tool_name: str
    critical: bool


class WorkflowInfo(BaseModel):
    model_config = {"extra": "forbid"}

    workflow_name: str
    description: str
    allowed_roles: list[str]
    max_steps: int
    continue_on_non_critical_failure: bool
    steps: list[WorkflowInfoStep]


class WorkflowResponse(BaseModel):
    model_config = {"extra": "forbid"}

    ok: bool
    workflow_name: str
    request_id: str
    correlation_id: str
    final_status: Literal["completed", "partial", "failed", "blocked"]
    executed_steps: list[str] = Field(default_factory=list)
    skipped_steps: list[WorkflowSkippedStep] = Field(default_factory=list)
    step_results: list[WorkflowStepResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    meta: WorkflowExplainabilityMeta
    error_code: str | None = None
    message: str | None = None


class WorkflowHistoryStepOut(BaseModel):
    model_config = {"extra": "forbid"}

    step_index: int
    step_name: str
    tool_name: str
    tool_version: str | None = None
    step_status: Literal["completed", "failed"]
    execution_ms: float
    cache_hit: bool = False
    llm_used: bool = False
    deterministic_fallback: bool = False
    error_code: str | None = None
    warning_count: int = 0


class WorkflowHistoryRunOut(BaseModel):
    model_config = {"extra": "forbid"}

    workflow_run_id: str
    workflow_name: str
    user_id: str
    role: str
    correlation_id: str
    request_id: str
    final_status: str
    blocked_reason: str | None = None
    partial_reason: str | None = None
    executed_steps: list[str] = Field(default_factory=list)
    skipped_steps: list[WorkflowSkippedStep] = Field(default_factory=list)
    step_count: int = 0
    started_at: str
    finished_at: str
    duration_ms: float
    tool_order: list[str] = Field(default_factory=list)
    cache_hits_count: int = 0
    llm_steps_count: int = 0
    fallback_steps_count: int = 0
    ownership_context_present: bool = False
    request_fingerprint: str = ""
    request_meta: dict[str, object] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error_code: str | None = None
    message: str | None = None


class WorkflowHistoryDetailOut(BaseModel):
    model_config = {"extra": "forbid"}

    run: WorkflowHistoryRunOut
    steps: list[WorkflowHistoryStepOut] = Field(default_factory=list)


class WorkflowHistoryListOut(BaseModel):
    model_config = {"extra": "forbid"}

    items: list[WorkflowHistoryRunOut] = Field(default_factory=list)
    total: int = 0
    limit: int = 50
    offset: int = 0


class WorkflowHistorySummaryOut(BaseModel):
    model_config = {"extra": "forbid"}

    total_runs: int = 0
    success_count: int = 0
    partial_count: int = 0
    failed_count: int = 0
    blocked_count: int = 0
    average_duration_ms: float = 0.0
    most_used_workflows: list[dict[str, object]] = Field(default_factory=list)
    most_common_failed_step: dict[str, object] = Field(default_factory=dict)
