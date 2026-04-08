"""Shared helpers for deterministic Phase 12 node implementations.

These helpers keep node modules small and consistent without moving policy
logic out of the nodes themselves.
"""

from __future__ import annotations

from typing import Any

from ..policy import Phase12FailureCode, Phase12PolicyDecision
from ..state import GraphExecutionStatus, Phase12GraphState
from ..tracing.graph_trace import finalize_trace, record_decision, record_error, record_event, record_route


def succeed_node(
    state: Phase12GraphState,
    *,
    node_name: str,
    decision_type: str,
    branch: str,
    reason: str,
    detail: dict[str, Any] | None = None,
    confidence: float | None = None,
    safe_mode_triggered: bool = False,
) -> Phase12GraphState:
    """Record a successful node completion with decision and route traces."""

    record_decision(
        state,
        source_node=node_name,
        decision_type=decision_type,
        chosen_branch=branch,
        reason=reason,
        confidence=confidence,
        safe_mode_triggered=safe_mode_triggered,
    )
    record_route(
        state,
        source_node=node_name,
        chosen_branch=branch,
        reason=reason,
        confidence=confidence,
        safe_mode_triggered=safe_mode_triggered,
    )
    record_event(
        state,
        node_name,
        {
            "decision_summary": reason,
            "warning_count": len(state.warnings),
            **(detail or {}),
        },
    )
    return state


def fail_node(
    state: Phase12GraphState,
    *,
    node_name: str,
    failure_code: Phase12FailureCode,
    reason: str,
    final_status: GraphExecutionStatus = GraphExecutionStatus.FAILED,
    detail: dict[str, Any] | None = None,
) -> Phase12GraphState:
    """Record a terminal node failure and finalize the trace safely."""

    state.status = final_status
    state.final_status = final_status
    state.add_failure_reason(reason)
    record_decision(
        state,
        source_node=node_name,
        decision_type=f"{node_name}_decision",
        chosen_branch=final_status.value,
        reason=reason,
        safe_mode_triggered=state.safe_mode,
    )
    if detail:
        record_route(
            state,
            source_node=node_name,
            chosen_branch=final_status.value,
            reason=reason,
        )
    record_error(
        state,
        node_name,
        reason,
        failure_code=failure_code.value,
    )
    finalize_trace(
        state,
        final_status=final_status.value,
        reason=reason,
        failure_code=failure_code.value,
    )
    return state


def apply_policy_decision(
    state: Phase12GraphState,
    *,
    node_name: str,
    decision: Phase12PolicyDecision,
    decision_type: str,
    success_branch: str,
    blocked_status: GraphExecutionStatus = GraphExecutionStatus.BLOCKED,
) -> Phase12GraphState:
    """Apply a policy decision and record consistent trace output."""

    for warning in decision.warnings:
        state.add_warning(warning)
    if decision.allowed:
        return succeed_node(
            state,
            node_name=node_name,
            decision_type=decision_type,
            branch=success_branch,
            reason=decision.reason or "policy_allowed",
            detail=decision.metadata,
        )
    return fail_node(
        state,
        node_name=node_name,
        failure_code=decision.failure_code or Phase12FailureCode.POLICY_BLOCKED,
        reason=decision.reason or "policy_denied",
        final_status=blocked_status,
        detail=decision.metadata,
    )
