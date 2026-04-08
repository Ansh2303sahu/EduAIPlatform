"""Loop-prevention helpers for Phase 12 LangGraph.

These helpers do not execute the graph. They inspect the typed graph state and
policy limits to help future node runners stop safely when progress stalls or a
bounded retry sequence starts looping.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .policy import (
    Phase12ExecutionLimits,
    Phase12FailureCode,
    build_execution_limits,
    get_current_node_retry_count,
    get_total_step_count,
    is_node_retry_limit_reached,
    is_repair_loop_limit_reached,
    is_retrieval_retry_limit_reached,
    is_repeated_transition_limit_reached,
    is_tool_call_limit_reached,
    is_total_step_limit_reached,
)
from .state import GraphExecutionStatus, Phase12GraphState


class LoopCheckResult(BaseModel):
    """Result of a loop-prevention check."""

    triggered: bool = False
    terminal: bool = False
    reason: str = ""
    failure_code: Phase12FailureCode | None = None
    fingerprint: str = ""
    repeat_count: int = 0
    transition: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class TerminalDecision(BaseModel):
    """Describe whether the graph should stop and what safe status to use."""

    should_stop: bool = False
    safe_stop: bool = False
    final_status: GraphExecutionStatus | None = None
    failure_code: Phase12FailureCode | None = None
    reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


def state_fingerprint(state: Phase12GraphState) -> str:
    """Return the stable lightweight fingerprint defined by the state model."""

    return state.loop_fingerprint()


def _fingerprint_history_from_state(state: Phase12GraphState) -> list[str]:
    fingerprints: list[str] = []
    for entry in state.audit_trace:
        value = entry.get("fingerprint")
        if isinstance(value, str) and value:
            fingerprints.append(value)
    for entry in state.node_history:
        value = entry.detail.get("fingerprint")
        if isinstance(value, str) and value:
            fingerprints.append(value)
    for entry in state.decision_history:
        value = entry.detail.get("fingerprint")
        if isinstance(value, str) and value:
            fingerprints.append(value)
    return fingerprints


def detect_repeated_fingerprint(
    state: Phase12GraphState,
    *,
    fingerprint_history: list[str] | None = None,
    limits: Phase12ExecutionLimits | None = None,
) -> LoopCheckResult:
    """Check whether the current state fingerprint is repeating too often."""

    active_limits = limits or build_execution_limits()
    history = fingerprint_history or _fingerprint_history_from_state(state)
    fingerprint = state_fingerprint(state)
    repeat_count = sum(1 for item in history if item == fingerprint)
    triggered = repeat_count >= active_limits.max_repeated_state_transitions
    return LoopCheckResult(
        triggered=triggered,
        terminal=triggered,
        reason="repeated_state_fingerprint" if triggered else "",
        failure_code=Phase12FailureCode.LOOP_PREVENTED if triggered else None,
        fingerprint=fingerprint,
        repeat_count=repeat_count,
        metadata={"history_count": len(history)},
    )


def _node_sequence(state: Phase12GraphState) -> list[str]:
    names: list[str] = []
    for entry in state.node_history:
        if not names or names[-1] != entry.node_name:
            names.append(entry.node_name)
    if state.current_node and (not names or names[-1] != state.current_node):
        names.append(state.current_node)
    return names


def detect_repeated_transition(
    state: Phase12GraphState,
    *,
    limits: Phase12ExecutionLimits | None = None,
) -> LoopCheckResult:
    """Check whether the graph keeps bouncing through the same transition."""

    active_limits = limits or build_execution_limits()
    names = _node_sequence(state)
    if len(names) < 2:
        return LoopCheckResult()

    transitions = [f"{a}->{b}" for a, b in zip(names, names[1:])]
    last_transition = transitions[-1]
    repeat_count = 1
    for transition in reversed(transitions[:-1]):
        if transition != last_transition:
            break
        repeat_count += 1

    triggered = is_repeated_transition_limit_reached(repeat_count, limits=active_limits)
    return LoopCheckResult(
        triggered=triggered,
        terminal=triggered,
        reason="repeated_node_transition" if triggered else "",
        failure_code=Phase12FailureCode.LOOP_PREVENTED if triggered else None,
        repeat_count=repeat_count,
        transition=last_transition,
        metadata={"transition_count": len(transitions)},
    )


def detect_no_progress(
    state: Phase12GraphState,
    *,
    fingerprint_history: list[str] | None = None,
    limits: Phase12ExecutionLimits | None = None,
) -> LoopCheckResult:
    """Check for no-progress conditions using fingerprint and retry signals."""

    active_limits = limits or build_execution_limits()
    fingerprint_result = detect_repeated_fingerprint(
        state,
        fingerprint_history=fingerprint_history,
        limits=active_limits,
    )
    transition_result = detect_repeated_transition(state, limits=active_limits)
    current_retry_count = get_current_node_retry_count(state)

    triggered = (
        fingerprint_result.triggered
        or transition_result.triggered
        or (
            current_retry_count > 0
            and current_retry_count >= active_limits.max_retries_per_node
            and fingerprint_result.repeat_count > 0
        )
    )
    return LoopCheckResult(
        triggered=triggered,
        terminal=triggered,
        reason="no_progress_detected" if triggered else "",
        failure_code=Phase12FailureCode.LOOP_PREVENTED if triggered else None,
        fingerprint=fingerprint_result.fingerprint,
        repeat_count=max(fingerprint_result.repeat_count, transition_result.repeat_count),
        transition=transition_result.transition,
        metadata={
            "current_retry_count": current_retry_count,
            "fingerprint_repeat_count": fingerprint_result.repeat_count,
            "transition_repeat_count": transition_result.repeat_count,
        },
    )


def _partial_output_present(state: Phase12GraphState) -> bool:
    return bool(
        state.final_report
        or state.repaired_report
        or state.draft_report
        or state.section_plan
    )


def terminal_decision(
    state: Phase12GraphState,
    *,
    limits: Phase12ExecutionLimits | None = None,
    fingerprint_history: list[str] | None = None,
) -> TerminalDecision:
    """Determine whether graph execution should stop safely."""

    active_limits = limits or build_execution_limits()
    partial = _partial_output_present(state)
    final_status = GraphExecutionStatus.PARTIAL if partial else GraphExecutionStatus.FAILED

    if is_total_step_limit_reached(state, limits=active_limits):
        return TerminalDecision(
            should_stop=True,
            safe_stop=True,
            final_status=final_status,
            failure_code=Phase12FailureCode.MAX_STEPS_EXCEEDED,
            reason="Maximum total graph steps exceeded.",
            metadata={"step_count": get_total_step_count(state)},
        )
    if is_node_retry_limit_reached(state, limits=active_limits):
        return TerminalDecision(
            should_stop=True,
            safe_stop=True,
            final_status=final_status,
            failure_code=Phase12FailureCode.LOOP_PREVENTED,
            reason="Maximum node retries reached.",
            metadata={"current_node": state.current_node},
        )
    if is_retrieval_retry_limit_reached(state, limits=active_limits):
        return TerminalDecision(
            should_stop=True,
            safe_stop=True,
            final_status=final_status,
            failure_code=Phase12FailureCode.RETRIEVAL_REPAIR_FAILED,
            reason="Maximum retrieval retries reached.",
            metadata={"retrieval_attempts": state.retrieval_attempts},
        )
    if is_repair_loop_limit_reached(state, limits=active_limits):
        return TerminalDecision(
            should_stop=True,
            safe_stop=True,
            final_status=final_status,
            failure_code=Phase12FailureCode.VALIDATION_FAILED,
            reason="Maximum repair loops reached.",
            metadata={"retry_counts": state.retry_counts},
        )
    if is_tool_call_limit_reached(state, limits=active_limits):
        return TerminalDecision(
            should_stop=True,
            safe_stop=True,
            final_status=final_status,
            failure_code=Phase12FailureCode.TOOL_BLOCKED,
            reason="Maximum tool calls reached.",
            metadata={"tool_call_count": state.tool_call_count},
        )

    no_progress = detect_no_progress(
        state,
        fingerprint_history=fingerprint_history,
        limits=active_limits,
    )
    if no_progress.triggered:
        return TerminalDecision(
            should_stop=True,
            safe_stop=True,
            final_status=final_status,
            failure_code=no_progress.failure_code,
            reason=no_progress.reason,
            metadata=no_progress.metadata,
        )

    return TerminalDecision()
