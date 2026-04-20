"""Structured tracing models and helpers for Phase 12 LangGraph.

The tracing layer is compact and audit-safe by design. It captures graph
execution metadata, node events, routing/decision summaries, failures, and the
terminal outcome without storing raw prompts, retrieved chunk text, or other
large sensitive blobs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..config import phase12_settings

if TYPE_CHECKING:
    from ..state import Phase12GraphState


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TraceNodeStatus(str, Enum):
    """Status for a traced node execution."""

    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRIED = "retried"


class GraphExecutionTraceMetadata(BaseModel):
    """Top-level execution metadata for a Phase 12 graph run."""

    model_config = {"protected_namespaces": ()}

    execution_id: str
    graph_name: str
    graph_version: str
    prompt_version: str
    retrieval_policy_version: str
    tool_policy_version: str
    model_version: str
    user_role: str
    analysis_type: str
    started_at: datetime
    completed_at: datetime | None = None
    final_status: str | None = None
    safe_mode: bool = False
    total_steps: int = 0
    total_latency_ms: float = 0.0
    schema_version: str = phase12_settings.trace_schema_version


class GraphNodeTraceEntry(BaseModel):
    """Compact node-level trace entry with timing and retry metadata."""

    node_name: str
    status: TraceNodeStatus
    started_at: datetime = Field(default_factory=_utc_now)
    completed_at: datetime | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)
    retry_count: int = Field(default=0, ge=0)
    decision_summary: str = ""
    warning_count: int = Field(default=0, ge=0)
    failure_code: str | None = None


class GraphDecisionTraceEntry(BaseModel):
    """Compact decision summary captured during graph execution."""

    source_node: str
    decision_type: str
    chosen_branch: str = ""
    short_reason: str = ""
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    safe_mode_triggered: bool = False
    timestamp: datetime = Field(default_factory=_utc_now)


class GraphRoutingTraceEntry(BaseModel):
    """Compact branch/routing summary for a graph transition."""

    source_node: str
    decision_type: str = "routing"
    chosen_branch: str
    short_reason: str = ""
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    safe_mode_triggered: bool = False
    timestamp: datetime = Field(default_factory=_utc_now)


class GraphFailureTraceEntry(BaseModel):
    """Compact failure event for audit and future observability."""

    node_name: str
    failure_code: str
    short_reason: str
    timestamp: datetime = Field(default_factory=_utc_now)
    retry_count: int = Field(default=0, ge=0)
    warning_count: int = Field(default=0, ge=0)


class GraphTerminalTraceEntry(BaseModel):
    """Terminal outcome of the graph run."""

    final_status: str
    reason: str = ""
    safe_mode: bool = False
    completed_at: datetime = Field(default_factory=_utc_now)
    total_steps: int = Field(default=0, ge=0)
    total_latency_ms: float = Field(default=0.0, ge=0.0)
    failure_code: str | None = None
    warning_count: int = Field(default=0, ge=0)


class Phase12TraceSnapshot(BaseModel):
    """Compact trace snapshot ready for future persistence or observability."""

    metadata: GraphExecutionTraceMetadata
    node_entries: list[GraphNodeTraceEntry] = Field(default_factory=list)
    decision_entries: list[GraphDecisionTraceEntry] = Field(default_factory=list)
    routing_entries: list[GraphRoutingTraceEntry] = Field(default_factory=list)
    failure_entries: list[GraphFailureTraceEntry] = Field(default_factory=list)
    terminal_outcome: GraphTerminalTraceEntry | None = None
    summary: str = ""


def build_execution_trace_metadata(state: Phase12GraphState) -> GraphExecutionTraceMetadata:
    """Build graph execution metadata from the current Phase 12 state."""

    return GraphExecutionTraceMetadata(
        execution_id=state.execution_id,
        graph_name=state.graph_name,
        graph_version=state.graph_version,
        prompt_version=state.prompt_version,
        retrieval_policy_version=state.retrieval_policy_version,
        tool_policy_version=state.tool_policy_version,
        model_version=state.model_version,
        user_role=state.user_role.value,
        analysis_type=state.analysis_type.value,
        started_at=state.started_at,
        completed_at=state.completed_at,
        final_status=state.final_status.value if state.final_status else None,
        safe_mode=state.safe_mode,
        total_steps=state.total_steps,
        total_latency_ms=state.total_latency_ms,
    )


def build_compact_trace_summary(state: Phase12GraphState) -> str:
    """Return a short audit-safe human-readable trace summary."""

    max_items = phase12_settings.trace_summary_max_items
    parts: list[str] = [
        f"Graph started ({state.graph_name}, role={state.user_role.value})",
    ]

    visited = [entry.node_name for entry in state.node_traces]
    if visited:
        unique_nodes: list[str] = []
        for node in visited:
            if not unique_nodes or unique_nodes[-1] != node:
                unique_nodes.append(node)
        shown_nodes = " -> ".join(unique_nodes[:max_items])
        if len(unique_nodes) > max_items:
            shown_nodes += " -> ..."
        parts.append(f"Nodes visited: {shown_nodes}")

    if state.routing_traces:
        routes = ", ".join(
            f"{entry.source_node}:{entry.chosen_branch}"
            for entry in state.routing_traces[:max_items]
        )
        parts.append(f"Routes: {routes}")

    if state.decision_traces:
        decisions = ", ".join(
            f"{entry.decision_type}:{entry.chosen_branch or entry.short_reason}"
            for entry in state.decision_traces[:max_items]
        )
        parts.append(f"Decisions: {decisions}")

    retries = {
        node: count for node, count in state.retry_counts.items() if count > 0
    }
    if retries:
        retry_text = ", ".join(f"{node}x{count}" for node, count in list(retries.items())[:max_items])
        parts.append(f"Retries: {retry_text}")

    if state.safe_mode:
        parts.append("Safe mode activated")

    if state.failure_traces:
        failure_text = ", ".join(
            f"{entry.node_name}:{entry.failure_code}"
            for entry in state.failure_traces[:max_items]
        )
        parts.append(f"Failures: {failure_text}")

    if state.terminal_trace is not None:
        terminal = state.terminal_trace
        parts.append(
            f"Final outcome: {terminal.final_status} ({terminal.total_latency_ms:.1f} ms)"
        )
    elif state.final_status is not None:
        parts.append(f"Final outcome: {state.final_status.value}")

    return "; ".join(parts)


def trace_snapshot(state: Phase12GraphState) -> dict[str, Any]:
    """Return the structured trace snapshot as a JSON-safe dict."""

    snapshot = Phase12TraceSnapshot(
        metadata=build_execution_trace_metadata(state),
        node_entries=list(state.node_traces),
        decision_entries=list(state.decision_traces),
        routing_entries=list(state.routing_traces),
        failure_entries=list(state.failure_traces),
        terminal_outcome=state.terminal_trace,
        summary=build_compact_trace_summary(state),
    )
    return snapshot.model_dump(mode="json")


def record_event(
    state: Phase12GraphState,
    node_name: str,
    detail: dict[str, Any] | None = None,
) -> None:
    """Record a successful node event using the state's structured trace helpers."""

    payload = detail or {}
    decision_summary = str(payload.get("decision_summary") or payload.get("summary") or "")
    warning_count = int(payload.get("warning_count") or 0)
    retry_count = int(payload.get("retry_count") or state.retry_counts.get(node_name, 0))
    state.append_node_trace(
        node_name=node_name,
        status=TraceNodeStatus.COMPLETED,
        retry_count=retry_count,
        decision_summary=decision_summary,
        warning_count=warning_count,
    )
    state.append_node_history(
        node_name=node_name,
        outcome="completed",
        detail=payload,
        attempt=retry_count,
    )


def record_error(
    state: Phase12GraphState,
    node_name: str,
    error: str,
    *,
    failure_code: str = "phase12.error",
) -> None:
    """Record a failed node event using the state's structured trace helpers."""

    retry_count = state.retry_counts.get(node_name, 0)
    state.add_failure_reason(f"{failure_code}: {error}")
    state.append_failure_trace(
        node_name=node_name,
        failure_code=failure_code,
        reason=error,
        retry_count=retry_count,
    )
    state.append_node_trace(
        node_name=node_name,
        status=TraceNodeStatus.FAILED,
        retry_count=retry_count,
        failure_code=failure_code,
        decision_summary=error,
    )
    state.append_node_history(
        node_name=node_name,
        outcome="failed",
        detail={"error": error, "failure_code": failure_code},
        attempt=retry_count,
    )


def record_decision(
    state: Phase12GraphState,
    *,
    source_node: str,
    decision_type: str,
    chosen_branch: str = "",
    reason: str = "",
    confidence: float | None = None,
    safe_mode_triggered: bool = False,
) -> None:
    """Record a compact decision summary."""

    state.append_decision_trace(
        source_node=source_node,
        decision_type=decision_type,
        chosen_branch=chosen_branch,
        reason=reason,
        confidence=confidence,
        safe_mode_triggered=safe_mode_triggered,
    )
    state.append_decision_history(
        node_name=source_node,
        decision=decision_type,
        reason=reason,
        detail={
            "chosen_branch": chosen_branch,
            "confidence": confidence,
            "safe_mode_triggered": safe_mode_triggered,
        },
    )


def record_route(
    state: Phase12GraphState,
    *,
    source_node: str,
    chosen_branch: str,
    reason: str = "",
    confidence: float | None = None,
    safe_mode_triggered: bool = False,
) -> None:
    """Record a compact routing/branch summary."""

    state.append_routing_trace(
        source_node=source_node,
        chosen_branch=chosen_branch,
        reason=reason,
        confidence=confidence,
        safe_mode_triggered=safe_mode_triggered,
    )


def finalize_trace(
    state: Phase12GraphState,
    *,
    final_status: str,
    reason: str = "",
    failure_code: str | None = None,
) -> None:
    """Finalize graph-level trace metadata and terminal outcome."""

    state.finalize_trace(
        final_status=final_status,
        reason=reason,
        failure_code=failure_code,
    )
