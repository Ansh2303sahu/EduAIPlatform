"""Final coherence guardrail node for Phase 12.

Performs lightweight structural checks on the graph's final state and any
already-produced report payload. It does not generate content, call external
services, or mutate storage.
"""

from __future__ import annotations

from ..loop_control import terminal_decision
from ..policy import Phase12FailureCode
from ..schemas import Phase12NodeDescriptor
from ..state import GraphExecutionStatus, GraphValidationStatus, Phase12GraphState
from ..tracing.graph_trace import finalize_trace
from ._helpers import fail_node, succeed_node

NODE_NAME = "final_guardrail"
NODE_SPEC = Phase12NodeDescriptor(
    name=NODE_NAME,
    description="Apply final lightweight consistency and output-shape guardrails.",
    wrapped_modules=["app.langgraph.loop_control", "app.langgraph.state"],
    reads=["pipeline_context.report", "validation_status", "warnings", "failure_reasons"],
    writes=["status", "final_status", "warnings", "failure_reasons"],
)


def _expected_report_keys(state: Phase12GraphState) -> list[str]:
    if state.role == "student":
        return ["summary", "issues", "strengths", "safety"]
    return ["rubric_breakdown", "feedback_explanation", "safety"]


def _coherent_terminal_status(state: Phase12GraphState) -> GraphExecutionStatus:
    report_present = bool(state.final_report)
    validation_status = state.validation_status
    has_failures = bool(state.failure_reasons)
    has_warnings = bool(state.warnings)

    if state.final_status == GraphExecutionStatus.BLOCKED and not report_present:
        return GraphExecutionStatus.BLOCKED
    if report_present:
        if has_failures or has_warnings or validation_status == GraphValidationStatus.REPAIRED:
            return GraphExecutionStatus.PARTIAL
        if validation_status == GraphValidationStatus.INVALID:
            return GraphExecutionStatus.PARTIAL
        return GraphExecutionStatus.COMPLETED
    if state.final_status == GraphExecutionStatus.BLOCKED:
        return GraphExecutionStatus.BLOCKED
    return GraphExecutionStatus.FAILED


async def final_guardrail_node(state: Phase12GraphState) -> Phase12GraphState:
    """Apply compact final checks to the graph output state.

    Inputs inspected:
    - final report payload and top-level keys
    - warning/failure collections
    - validation status and loop-control terminal hints

    Outputs updated:
    - final/coherent graph status
    - warnings for missing expected sections
    - structured trace and terminal outcome

    This node intentionally does not call ML, retrieval, generation, storage,
    or MCP.
    """

    state.set_current_node(NODE_NAME)
    guardrail = terminal_decision(state)
    if guardrail.should_stop and not state.final_report:
        return fail_node(
            state,
            node_name=NODE_NAME,
            failure_code=guardrail.failure_code or Phase12FailureCode.LOOP_PREVENTED,
            reason=guardrail.reason or "Terminal guardrail requested a safe stop before final output existed.",
            final_status=guardrail.final_status or GraphExecutionStatus.FAILED,
            detail=guardrail.metadata,
        )

    report = state.pipeline_context.report
    if not isinstance(report, dict):
        return fail_node(
            state,
            node_name=NODE_NAME,
            failure_code=Phase12FailureCode.VALIDATION_FAILED,
            reason="Final report payload is not a JSON-like object.",
        )

    if report:
        expected_keys = _expected_report_keys(state)
        missing_keys = [key for key in expected_keys if key not in report]
        if missing_keys:
            state.add_warning(
                "Final report is missing expected top-level keys: " + ", ".join(missing_keys) + "."
            )
        safety = report.get("safety")
        if safety is not None and not isinstance(safety, dict):
            return fail_node(
                state,
                node_name=NODE_NAME,
                failure_code=Phase12FailureCode.VALIDATION_FAILED,
                reason="Final report safety field is not a JSON-like object.",
            )

    if state.final_status == GraphExecutionStatus.BLOCKED and report:
        state.add_warning("Final report exists even though the graph was marked blocked; treating result as partial.")

    coherent_status = _coherent_terminal_status(state)
    if coherent_status == GraphExecutionStatus.COMPLETED and not report:
        return fail_node(
            state,
            node_name=NODE_NAME,
            failure_code=Phase12FailureCode.VALIDATION_FAILED,
            reason="Graph cannot complete without a final report payload.",
        )
    if coherent_status == GraphExecutionStatus.FAILED and not report:
        return fail_node(
            state,
            node_name=NODE_NAME,
            failure_code=Phase12FailureCode.VALIDATION_FAILED,
            reason="No final report payload is available for final guardrail evaluation.",
        )

    if coherent_status == GraphExecutionStatus.PARTIAL and not state.failure_reasons and not state.warnings:
        state.add_warning("Final output was downgraded to partial for conservative consistency handling.")

    if guardrail.should_stop and report:
        state.add_warning("Terminal guardrail triggered after partial output was produced; preserving the result conservatively.")
        if coherent_status == GraphExecutionStatus.COMPLETED:
            coherent_status = guardrail.final_status or GraphExecutionStatus.PARTIAL

    reason = {
        GraphExecutionStatus.COMPLETED: "Final report passed lightweight guardrail checks.",
        GraphExecutionStatus.PARTIAL: "Final report is usable but carries warnings, repairs, or partial-failure signals.",
        GraphExecutionStatus.BLOCKED: "Final state remains blocked by policy or prior guardrail decisions.",
        GraphExecutionStatus.FAILED: "Final output is not coherent enough to treat as complete.",
        GraphExecutionStatus.RUNNING: "Graph is still marked running unexpectedly.",
        GraphExecutionStatus.PENDING: "Graph remained pending unexpectedly.",
    }[coherent_status]

    state.status = coherent_status
    state.final_status = coherent_status
    succeed_node(
        state,
        node_name=NODE_NAME,
        decision_type="final_guardrail",
        branch=coherent_status.value,
        reason=reason,
        detail={
            "warning_count": len(state.warnings),
            "failure_count": len(state.failure_reasons),
            "final_report_keys": sorted(report.keys()),
            "validation_status": state.validation_status.value,
        },
        confidence=state.final_confidence if report else None,
        safe_mode_triggered=state.safe_mode,
    )
    finalize_trace(
        state,
        final_status=coherent_status.value,
        reason=reason,
        failure_code=(
            guardrail.failure_code.value
            if guardrail.should_stop and coherent_status != GraphExecutionStatus.COMPLETED and guardrail.failure_code
            else None
        ),
    )
    return state
