"""Typed Phase 12 LangGraph state built on top of Phase 10 models.

Phase 12 is an orchestration layer only.  This state model intentionally reuses
the existing Phase 10 ``PipelineContext`` and related models for ML, RAG,
validation, and execution metadata rather than duplicating those contracts.

The result is a graph-friendly state object that:
- stays additive to the current architecture
- remains JSON-serializable for audit/log persistence
- keeps student/professor role semantics explicit
- avoids introducing a second storage or report schema
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field

from app.core.deps import CurrentUser
from app.langchain.enums import (
    AnalysisType,
    ChainRole,
    DecisionSource,
    ExecutionMode,
    PipelineMode,
    PipelineRole,
)
from app.langchain.models import ExecutionMetadata, MLContextResult, PipelineContext, RagContext

from .schemas import Phase12ExecutionRequest
from .tracing.graph_trace import (
    GraphDecisionTraceEntry,
    GraphExecutionTraceMetadata,
    GraphFailureTraceEntry,
    GraphNodeTraceEntry,
    GraphRoutingTraceEntry,
    GraphTerminalTraceEntry,
    TraceNodeStatus,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _unit_float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0.0:
        return None
    return max(0.0, min(1.0, number))


def _bucket_float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        bucket = int(float(value))
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, max(0, min(4, bucket)) / 4.0))


def _default_analysis_type(role: str) -> AnalysisType:
    return (
        AnalysisType.STUDENT_ACADEMIC
        if role == "student"
        else AnalysisType.PROFESSOR_ACADEMIC
    )


def _pipeline_role(role: str) -> PipelineRole:
    return PipelineRole.STUDENT if role == "student" else PipelineRole.PROFESSOR


def _chain_role(role: str) -> ChainRole:
    return ChainRole.STUDENT if role == "student" else ChainRole.PROFESSOR


class GraphExecutionStatus(str, Enum):
    """Lifecycle status for a Phase 12 graph run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    BLOCKED = "blocked"


class IngestionStatus(str, Enum):
    """Status of evidence ingestion into the graph state."""

    PENDING = "pending"
    READY = "ready"
    PARTIAL = "partial"
    MISSING = "missing"


class NodeOutcome(str, Enum):
    """Outcome recorded for a graph node execution attempt."""

    STARTED = "started"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
    RETRIED = "retried"


class FailureType(str, Enum):
    """Classify failure reasons without exposing internal stack traces."""

    INGESTION = "ingestion"
    ML = "ml"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    VALIDATION = "validation"
    TOOLING = "tooling"
    STORAGE = "storage"
    POLICY = "policy"
    UNKNOWN = "unknown"


class GraphValidationStatus(str, Enum):
    """Validation status projected from the existing Phase 10 validation result."""

    PENDING = "pending"
    VALID = "valid"
    INVALID = "invalid"
    REPAIRED = "repaired"


class NodeHistoryEntry(BaseModel):
    """Compact, audit-safe record of a node execution event."""

    node_name: str
    outcome: NodeOutcome
    timestamp: datetime = Field(default_factory=_utc_now)
    attempt: int = Field(default=0, ge=0)
    detail: dict[str, Any] = Field(default_factory=dict)


class DecisionHistoryEntry(BaseModel):
    """Compact record of a graph-level decision."""

    node_name: str = ""
    decision: str
    reason: str = ""
    timestamp: datetime = Field(default_factory=_utc_now)
    detail: dict[str, Any] = Field(default_factory=dict)


class Phase12GraphState(BaseModel):
    """Primary typed state container for Phase 12 internal graph orchestration.

    ``PipelineContext`` remains the source of truth for Phase 10-compatible ML,
    retrieval, validation, and report-generation data. This state adds only the
    graph-specific orchestration and audit fields around it.
    """

    model_config = ConfigDict(validate_assignment=True)

    request: Phase12ExecutionRequest
    pipeline_context: PipelineContext

    execution_id: str
    graph_name: str
    graph_version: str = "0.1.0"
    retrieval_policy_version: str = "phase9-rag-v1"
    tool_policy_version: str = "phase11-mcp-v1"

    status: GraphExecutionStatus = GraphExecutionStatus.PENDING
    current_node: str = ""
    final_status: GraphExecutionStatus | None = None

    ingestion_status: IngestionStatus = IngestionStatus.PENDING
    media_metadata: dict[str, Any] = Field(default_factory=dict)
    evidence_summary: str = ""
    evidence_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)

    ml_required: bool = True
    rag_required: bool = True
    retrieval_attempts: int = Field(default=0, ge=0)

    selected_tools: list[str] = Field(default_factory=list)
    tool_outputs: list[dict[str, Any]] = Field(default_factory=list)

    section_plan: list[str] = Field(default_factory=list)
    draft_report: dict[str, Any] = Field(default_factory=dict)
    critique_output: dict[str, Any] = Field(default_factory=dict)
    repaired_report: dict[str, Any] = Field(default_factory=dict)

    warnings: list[str] = Field(default_factory=list)
    failure_reasons: list[str] = Field(default_factory=list)
    retry_counts: dict[str, int] = Field(default_factory=dict)
    node_history: list[NodeHistoryEntry] = Field(default_factory=list)
    decision_history: list[DecisionHistoryEntry] = Field(default_factory=list)

    trace_metadata: GraphExecutionTraceMetadata | None = None
    node_traces: list[GraphNodeTraceEntry] = Field(default_factory=list)
    decision_traces: list[GraphDecisionTraceEntry] = Field(default_factory=list)
    routing_traces: list[GraphRoutingTraceEntry] = Field(default_factory=list)
    failure_traces: list[GraphFailureTraceEntry] = Field(default_factory=list)
    terminal_trace: GraphTerminalTraceEntry | None = None
    audit_trace: list[dict[str, Any]] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=_utc_now)
    completed_at: datetime | None = None

    file_row: dict[str, Any] = Field(default_factory=dict)
    input_hash: str = ""
    prompt_hash: str = ""
    storage_payload: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create(cls, request: Phase12ExecutionRequest) -> "Phase12GraphState":
        """Create a new Phase 12 state around the existing Phase 10 context."""

        execution_id = str(uuid4())
        analysis_type = _default_analysis_type(request.role)
        graph_name = f"phase12_{request.role}_graph"
        execution_meta = ExecutionMetadata(
            request_id=execution_id,
            role=_chain_role(request.role),
            analysis_type=analysis_type.value,
            submission_kind="academic",
            pipeline="phase12_langgraph",
            chain_name=graph_name,
            execution_mode=ExecutionMode.NORMAL,
            decision_source=DecisionSource.HYBRID,
        )
        context = PipelineContext(
            request_id=execution_id,
            file_id=request.file_id,
            submission_id=request.submission_id,
            role=_pipeline_role(request.role),
            analysis_type=analysis_type,
            mode=PipelineMode.NORMAL,
            execution_mode=ExecutionMode.NORMAL,
            submission_kind="academic",
            execution_meta=execution_meta,
        )
        return cls(
            request=request,
            pipeline_context=context,
            execution_id=execution_id,
            graph_name=graph_name,
        )

    @computed_field
    @property
    def prompt_version(self) -> str:
        """Return the role-appropriate prompt version from Phase 10 metadata."""

        meta = self.pipeline_context.execution_meta
        if self.user_role == ChainRole.STUDENT:
            return meta.student_prompt_version
        return meta.professor_prompt_version

    @computed_field
    @property
    def model_version(self) -> str:
        """Return the effective model label without creating a second model contract."""

        meta = self.pipeline_context.execution_meta
        return meta.model_used or meta.primary_model or meta.fallback_model or ""

    @computed_field
    @property
    def user_id(self) -> str:
        return self.request.user_id

    @computed_field
    @property
    def user_role(self) -> ChainRole:
        return _chain_role(self.request.role)

    @computed_field
    @property
    def submission_id(self) -> str:
        return self.pipeline_context.submission_id or self.request.submission_id

    @computed_field
    @property
    def file_id(self) -> str:
        return self.pipeline_context.file_id

    @computed_field
    @property
    def analysis_type(self) -> AnalysisType:
        return self.pipeline_context.analysis_type

    @computed_field
    @property
    def safe_mode(self) -> bool:
        return (
            self.pipeline_context.execution_mode != ExecutionMode.NORMAL
            or self.pipeline_context.mode != PipelineMode.NORMAL
        )

    @computed_field
    @property
    def extracted_text(self) -> str:
        return self.pipeline_context.ingestion.text_content

    @computed_field
    @property
    def ocr_text(self) -> str:
        return self.pipeline_context.ingestion.ocr_text

    @computed_field
    @property
    def transcript_text(self) -> str:
        return self.pipeline_context.ingestion.audio_transcript

    @computed_field
    @property
    def tables_json(self) -> dict[str, Any]:
        return self.pipeline_context.ingestion.tables_json or {}

    @computed_field
    @property
    def ml_completed(self) -> bool:
        return self.pipeline_context.ml_result is not None

    @computed_field
    @property
    def ml_result(self) -> MLContextResult | None:
        return self.pipeline_context.ml_result

    @computed_field
    @property
    def ml_confidence(self) -> float:
        result = self.pipeline_context.ml_result
        if result is not None:
            return float(result.confidence_score)
        raw = self.pipeline_context.ml_raw or {}

        for key in ("confidence_score", "confidence", "probability"):
            score = _unit_float_or_none(raw.get(key))
            if score is not None:
                return score

        bucket_score = _bucket_float_or_none(raw.get("confidence_0_to_4"))
        if bucket_score is not None:
            return bucket_score

        raw_bundle = raw.get("raw") if isinstance(raw.get("raw"), dict) else {}
        confidence_payload = (
            raw_bundle.get("confidence")
            if isinstance(raw_bundle.get("confidence"), dict)
            else {}
        )
        confidence_prediction = (
            confidence_payload.get("prediction")
            if isinstance(confidence_payload.get("prediction"), dict)
            else {}
        )
        score = _unit_float_or_none(confidence_prediction.get("confidence"))
        if score is not None:
            return score

        predictions = raw_bundle.get("predictions") if isinstance(raw_bundle.get("predictions"), dict) else {}
        head_scores = [
            score
            for prediction in predictions.values()
            if isinstance(prediction, dict)
            for score in [_unit_float_or_none(prediction.get("confidence"))]
            if score is not None
        ]
        if head_scores:
            return float(sum(head_scores) / len(head_scores))

        consistency = str(raw.get("moderation_consistency") or "").strip().lower()
        if consistency:
            return {"high": 0.85, "med": 0.6, "medium": 0.6, "low": 0.3}.get(consistency, 0.6)
        return 0.0

    @computed_field
    @property
    def rag_completed(self) -> bool:
        rag = self.pipeline_context.rag
        return bool(
            rag.enabled
            or rag.chunk_count
            or rag.context
            or rag.context_text
            or rag.trace
            or rag.retrieved_chunks
        )

    @computed_field
    @property
    def rag_result(self) -> RagContext:
        return self.pipeline_context.rag

    @computed_field
    @property
    def retrieved_chunks(self) -> list[dict[str, Any]]:
        return list(self.pipeline_context.rag.retrieved_chunks)

    @computed_field
    @property
    def retrieval_scores(self) -> list[float]:
        scores: list[float] = []
        for chunk in self.pipeline_context.rag.retrieved_chunks:
            if not isinstance(chunk, dict):
                continue
            raw = (
                chunk.get("score")
                or chunk.get("similarity")
                or chunk.get("rerank_score")
                or chunk.get("distance")
            )
            if isinstance(raw, (int, float)):
                scores.append(float(raw))
        return scores

    @computed_field
    @property
    def tool_call_count(self) -> int:
        return len(self.tool_outputs)

    @computed_field
    @property
    def final_report(self) -> dict[str, Any]:
        return self.pipeline_context.report

    @computed_field
    @property
    def final_confidence(self) -> float:
        report = self.pipeline_context.report
        if isinstance(report, dict):
            confidence = report.get("confidence") if isinstance(report.get("confidence"), dict) else {}
            agreement = (
                report.get("model_agreement")
                if isinstance(report.get("model_agreement"), dict)
                else {}
            )
            saw_zero_confidence = False
            for value in (
                agreement.get("final_confidence"),
                confidence.get("overall"),
                confidence.get("score"),
                report.get("confidence_score"),
                agreement.get("llm_confidence"),
                agreement.get("ml_confidence"),
            ):
                score = _unit_float_or_none(value)
                if score is not None and score > 0.0:
                    return score
                saw_zero_confidence = saw_zero_confidence or score == 0.0
            if saw_zero_confidence:
                return 0.0

        meta_score = _unit_float_or_none(self.pipeline_context.execution_meta.agreement_score)
        if meta_score is not None and meta_score > 0.0:
            return meta_score

        ml_score = _unit_float_or_none(self.ml_confidence)
        if ml_score is not None and ml_score > 0.0:
            return ml_score

        evidence_score = _unit_float_or_none(self.evidence_quality_score)
        return evidence_score if evidence_score is not None else 0.0

    @computed_field
    @property
    def validation_status(self) -> GraphValidationStatus:
        validation = self.pipeline_context.validation_result
        if validation.valid and validation.repaired:
            return GraphValidationStatus.REPAIRED
        if validation.valid:
            return GraphValidationStatus.VALID
        if validation.errors or validation.warnings:
            return GraphValidationStatus.INVALID
        return GraphValidationStatus.PENDING

    @computed_field
    @property
    def total_steps(self) -> int:
        return len(self.node_traces) or len(self.node_history)

    @computed_field
    @property
    def total_latency_ms(self) -> float:
        if self.completed_at is not None:
            return max(0.0, (self.completed_at - self.started_at).total_seconds() * 1000.0)
        node_latency = sum(entry.latency_ms for entry in self.node_traces)
        return max(0.0, float(node_latency))

    @property
    def role(self) -> str:
        """Compatibility alias used by the existing scaffolding nodes/adapters."""

        return self.user_role.value

    @property
    def correlation_id(self) -> str:
        """Compatibility alias for the current internal request model."""

        return self.request.correlation_id

    @property
    def errors(self) -> list[str]:
        """Compatibility alias for earlier scaffolding code."""

        return self.failure_reasons

    @property
    def trace(self) -> list[dict[str, Any]]:
        """Compatibility alias used by the graph tracing helpers."""

        return self.audit_trace

    @property
    def mcp_results(self) -> list[dict[str, Any]]:
        """Compatibility alias for MCP output accumulation."""

        return self.tool_outputs

    def build_current_user(self) -> CurrentUser:
        """Build a standard backend CurrentUser for adapters and MCP calls."""

        return CurrentUser(
            id=self.request.user_id,
            email=self.request.user_email,
            role=self.request.role,
            raw_claims=self.request.raw_claims,
        )

    def apply_submission_kind(self, submission_kind: str) -> None:
        """Update Phase 10 analysis mode using existing role-aware semantics."""

        normalized = (submission_kind or "academic").strip().lower()
        if self.role == "student":
            analysis_type = (
                AnalysisType.STUDENT_PROJECT
                if normalized in {"project", "code"}
                else AnalysisType.STUDENT_ACADEMIC
            )
        else:
            analysis_type = (
                AnalysisType.PROFESSOR_PROJECT
                if normalized in {"project", "code"}
                else AnalysisType.PROFESSOR_ACADEMIC
            )
        self.pipeline_context.submission_kind = normalized
        self.pipeline_context.analysis_type = analysis_type
        self.pipeline_context.execution_meta.analysis_type = analysis_type.value
        self.pipeline_context.execution_meta.submission_kind = normalized

    def set_current_node(
        self,
        node_name: str,
        *,
        status: GraphExecutionStatus | str | None = None,
    ) -> None:
        """Update the current node and optionally the graph status."""

        self.current_node = node_name
        if status is not None:
            self.status = status if isinstance(status, GraphExecutionStatus) else GraphExecutionStatus(status)
        elif self.status == GraphExecutionStatus.PENDING:
            self.status = GraphExecutionStatus.RUNNING

    def append_node_history(
        self,
        *,
        node_name: str,
        outcome: NodeOutcome | str,
        detail: dict[str, Any] | None = None,
        attempt: int | None = None,
    ) -> None:
        """Record a compact node execution event."""

        self.node_history.append(
            NodeHistoryEntry(
                node_name=node_name,
                outcome=outcome if isinstance(outcome, NodeOutcome) else NodeOutcome(outcome),
                attempt=attempt if attempt is not None else self.retry_counts.get(node_name, 0),
                detail=detail or {},
            )
        )

    def append_decision_history(
        self,
        *,
        decision: str,
        reason: str = "",
        node_name: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Record a graph-level decision in an audit-safe format."""

        self.decision_history.append(
            DecisionHistoryEntry(
                node_name=node_name or self.current_node,
                decision=decision,
                reason=reason,
                detail=detail or {},
            )
        )

    def _append_audit_record(self, *, event_type: str, payload: dict[str, Any]) -> None:
        """Append a compact JSON-safe audit record."""

        self.audit_trace.append(
            {
                "event_type": event_type,
                "timestamp": _utc_now().isoformat(),
                **payload,
            }
        )

    def append_node_trace(
        self,
        *,
        node_name: str,
        status: TraceNodeStatus | str,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        latency_ms: float | None = None,
        retry_count: int | None = None,
        decision_summary: str = "",
        warning_count: int = 0,
        failure_code: str | None = None,
    ) -> GraphNodeTraceEntry:
        """Append a typed node-level trace entry and mirror it into audit trace."""

        start = started_at or _utc_now()
        end = completed_at or start
        computed_latency = latency_ms
        if computed_latency is None:
            computed_latency = max(0.0, (end - start).total_seconds() * 1000.0)
        entry = GraphNodeTraceEntry(
            node_name=node_name,
            status=status if isinstance(status, TraceNodeStatus) else TraceNodeStatus(status),
            started_at=start,
            completed_at=end,
            latency_ms=float(computed_latency),
            retry_count=retry_count if retry_count is not None else self.retry_counts.get(node_name, 0),
            decision_summary=decision_summary,
            warning_count=warning_count,
            failure_code=failure_code,
        )
        self.node_traces.append(entry)
        self._append_audit_record(event_type="node", payload=entry.model_dump(mode="json"))
        return entry

    def append_decision_trace(
        self,
        *,
        source_node: str,
        decision_type: str,
        chosen_branch: str = "",
        reason: str = "",
        confidence: float | None = None,
        safe_mode_triggered: bool = False,
    ) -> GraphDecisionTraceEntry:
        """Append a compact decision trace entry."""

        entry = GraphDecisionTraceEntry(
            source_node=source_node,
            decision_type=decision_type,
            chosen_branch=chosen_branch,
            short_reason=reason,
            confidence=confidence,
            safe_mode_triggered=safe_mode_triggered,
        )
        self.decision_traces.append(entry)
        self._append_audit_record(event_type="decision", payload=entry.model_dump(mode="json"))
        return entry

    def append_routing_trace(
        self,
        *,
        source_node: str,
        chosen_branch: str,
        reason: str = "",
        confidence: float | None = None,
        safe_mode_triggered: bool = False,
    ) -> GraphRoutingTraceEntry:
        """Append a compact routing/branch trace entry."""

        entry = GraphRoutingTraceEntry(
            source_node=source_node,
            chosen_branch=chosen_branch,
            short_reason=reason,
            confidence=confidence,
            safe_mode_triggered=safe_mode_triggered,
        )
        self.routing_traces.append(entry)
        self._append_audit_record(event_type="routing", payload=entry.model_dump(mode="json"))
        return entry

    def append_failure_trace(
        self,
        *,
        node_name: str,
        failure_code: str,
        reason: str,
        retry_count: int | None = None,
        warning_count: int = 0,
    ) -> GraphFailureTraceEntry:
        """Append a compact failure trace entry."""

        entry = GraphFailureTraceEntry(
            node_name=node_name,
            failure_code=failure_code,
            short_reason=reason,
            retry_count=retry_count if retry_count is not None else self.retry_counts.get(node_name, 0),
            warning_count=warning_count,
        )
        self.failure_traces.append(entry)
        self._append_audit_record(event_type="failure", payload=entry.model_dump(mode="json"))
        return entry

    def finalize_trace(
        self,
        *,
        final_status: GraphExecutionStatus | str,
        reason: str = "",
        failure_code: str | None = None,
    ) -> GraphTerminalTraceEntry:
        """Finalize trace metadata and terminal outcome for the graph run."""

        from .tracing.graph_trace import build_execution_trace_metadata

        status_value = (
            final_status if isinstance(final_status, GraphExecutionStatus) else GraphExecutionStatus(final_status)
        )
        self.status = status_value
        self.final_status = status_value
        self.completed_at = self.completed_at or _utc_now()
        terminal = GraphTerminalTraceEntry(
            final_status=status_value.value,
            reason=reason,
            safe_mode=self.safe_mode,
            completed_at=self.completed_at,
            total_steps=self.total_steps,
            total_latency_ms=self.total_latency_ms,
            failure_code=failure_code,
            warning_count=len(self.warnings),
        )
        self.terminal_trace = terminal
        self.trace_metadata = build_execution_trace_metadata(self)
        self._append_audit_record(event_type="terminal", payload=terminal.model_dump(mode="json"))
        return terminal

    def increment_retry_count(self, node_name: str) -> int:
        """Increment and return the retry count for a node."""

        next_count = self.retry_counts.get(node_name, 0) + 1
        self.retry_counts[node_name] = next_count
        return next_count

    def add_warning(self, warning: str) -> None:
        """Add a deduplicated warning message."""

        normalized = warning.strip()
        if normalized and normalized not in self.warnings:
            self.warnings.append(normalized)

    def add_failure_reason(
        self,
        reason: str,
        *,
        failure_type: FailureType | None = None,
    ) -> None:
        """Add a compact failure reason without storing raw exception objects."""

        normalized = reason.strip()
        if not normalized:
            return
        if failure_type is not None:
            normalized = f"{failure_type.value}: {normalized}"
        if normalized not in self.failure_reasons:
            self.failure_reasons.append(normalized)

    def loop_fingerprint(self) -> str:
        """Return a small stable fingerprint for loop/retry detection.

        The fingerprint intentionally avoids raw extracted text and tool payloads.
        It is based on small, audit-safe execution metadata only.
        """

        payload = {
            "graph_name": self.graph_name,
            "graph_version": self.graph_version,
            "role": self.role,
            "analysis_type": self.analysis_type.value,
            "current_node": self.current_node,
            "status": self.status.value,
            "final_status": self.final_status.value if self.final_status else "",
            "safe_mode": self.safe_mode,
            "retrieval_attempts": self.retrieval_attempts,
            "retry_counts": self.retry_counts,
            "selected_tools": self.selected_tools,
            "tool_call_count": self.tool_call_count,
            "warning_count": len(self.warnings),
            "failure_count": len(self.failure_reasons),
            "evidence_lengths": {
                "text": len(self.extracted_text),
                "ocr": len(self.ocr_text),
                "transcript": len(self.transcript_text),
            },
            "report_keys": sorted(self.final_report.keys()),
            "section_plan_size": len(self.section_plan),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]

    def export_compact_trace_summary(self) -> str:
        """Return a short human-readable trace summary without sensitive content."""

        from .tracing.graph_trace import build_compact_trace_summary

        return build_compact_trace_summary(self)

    def export_audit_summary(self) -> dict[str, Any]:
        """Export a compact audit-safe summary without raw evidence text."""

        return {
            "execution_id": self.execution_id,
            "graph_name": self.graph_name,
            "graph_version": self.graph_version,
            "prompt_version": self.prompt_version,
            "retrieval_policy_version": self.retrieval_policy_version,
            "tool_policy_version": self.tool_policy_version,
            "model_version": self.model_version,
            "user_id": self.user_id,
            "user_role": self.user_role.value,
            "submission_id": self.submission_id,
            "file_id": self.file_id,
            "analysis_type": self.analysis_type.value,
            "status": self.status.value,
            "current_node": self.current_node,
            "safe_mode": self.safe_mode,
            "final_status": self.final_status.value if self.final_status else None,
            "total_steps": self.total_steps,
            "total_latency_ms": self.total_latency_ms,
            "ingestion_status": self.ingestion_status.value,
            "evidence_quality_score": self.evidence_quality_score,
            "ml_required": self.ml_required,
            "ml_completed": self.ml_completed,
            "ml_confidence": self.ml_confidence,
            "rag_required": self.rag_required,
            "rag_completed": self.rag_completed,
            "retrieval_attempts": self.retrieval_attempts,
            "retrieved_chunk_count": len(self.retrieved_chunks),
            "selected_tools": list(self.selected_tools),
            "tool_call_count": self.tool_call_count,
            "validation_status": self.validation_status.value,
            "warning_count": len(self.warnings),
            "failure_count": len(self.failure_reasons),
            "node_history_count": len(self.node_history),
            "decision_history_count": len(self.decision_history),
            "node_trace_count": len(self.node_traces),
            "decision_trace_count": len(self.decision_traces),
            "routing_trace_count": len(self.routing_traces),
            "failure_trace_count": len(self.failure_traces),
            "audit_trace_count": len(self.audit_trace),
            "final_confidence": self.final_confidence,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "trace_summary": self.export_compact_trace_summary(),
        }
