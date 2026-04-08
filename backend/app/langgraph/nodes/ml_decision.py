"""ML decision node for Phase 12.

Decides whether the graph should run ML later, reuse existing ML context, or
skip ML based on current typed state only.
"""

from __future__ import annotations

from ..policy import build_policy_bundle
from ..loop_control import terminal_decision
from ..schemas import Phase12NodeDescriptor
from ..state import IngestionStatus, Phase12GraphState
from ._helpers import succeed_node

NODE_NAME = "ml_decision"
NODE_SPEC = Phase12NodeDescriptor(
    name=NODE_NAME,
    description="Decide whether ML should run later, be skipped, or reuse existing context.",
    wrapped_modules=["app.langgraph.state", "app.langgraph.loop_control"],
    reads=["ml_required", "pipeline_context.ml_raw", "pipeline_context.ml_result", "ingestion_status"],
    writes=["ml_required", "warnings"],
)


async def ml_decision_node(state: Phase12GraphState) -> Phase12GraphState:
    """Make a bounded ML execution decision without calling ML services.

    Inputs inspected:
    - existing ML raw/result data
    - evidence sufficiency and final guardrail hints
    - current explicit ML requirement flag

    Outputs updated:
    - `ml_required`
    - warnings when evidence is too thin or ML is low confidence
    - decision and node traces

    This node intentionally does not perform ML inference.
    """

    state.set_current_node(NODE_NAME)
    policy_bundle = build_policy_bundle(state.user_role)
    low_conf_threshold = policy_bundle.role_policy.safe_mode_ml_confidence_threshold

    guardrail = terminal_decision(state)
    if guardrail.should_stop:
        state.ml_required = False
        state.add_warning("ML execution skipped because a terminal guardrail condition is already active.")
        return succeed_node(
            state,
            node_name=NODE_NAME,
            decision_type="ml_decision",
            branch="skip_terminal_guardrail",
            reason=guardrail.reason or "Terminal guardrail active.",
        )

    if state.ml_completed:
        state.ml_required = False
        if state.ml_confidence < low_conf_threshold:
            state.add_warning("Existing ML context is low confidence; safe-mode may be appropriate later.")
        return succeed_node(
            state,
            node_name=NODE_NAME,
            decision_type="ml_decision",
            branch="reuse_existing_ml_context",
            reason="A normalized ML context is already available.",
            detail={"low_confidence_threshold": low_conf_threshold},
            confidence=state.ml_confidence,
        )

    if state.pipeline_context.ml_raw:
        state.ml_required = False
        return succeed_node(
            state,
            node_name=NODE_NAME,
            decision_type="ml_decision",
            branch="reuse_existing_ml_raw",
            reason="Raw ML signals are already present and can be normalized later.",
        )

    if not state.ml_required:
        return succeed_node(
            state,
            node_name=NODE_NAME,
            decision_type="ml_decision",
            branch="skip_ml_not_required",
            reason="ML is not required for the current graph state.",
        )

    if state.ingestion_status == IngestionStatus.MISSING or state.evidence_quality_score < 0.08:
        state.ml_required = False
        state.add_warning("ML execution skipped because evidence is currently too thin.")
        return succeed_node(
            state,
            node_name=NODE_NAME,
            decision_type="ml_decision",
            branch="skip_ml_insufficient_evidence",
            reason="Evidence is too limited to justify an ML call.",
            confidence=state.evidence_quality_score,
        )

    reason = "ML should run later using the wrapped Phase 10 ML path."
    if state.evidence_quality_score < 0.35:
        reason = "Evidence is weak but still sufficient for a cautious ML attempt later."
    state.ml_required = True
    return succeed_node(
        state,
        node_name=NODE_NAME,
        decision_type="ml_decision",
        branch="run_ml_later",
        reason=reason,
        detail={"low_confidence_threshold": low_conf_threshold},
        confidence=state.evidence_quality_score,
    )
