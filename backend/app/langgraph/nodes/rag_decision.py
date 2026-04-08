"""RAG decision node for Phase 12.

Decides whether retrieval should run later, be skipped, or be strengthened
later using current policy, evidence, and state only.
"""

from __future__ import annotations

from ..policy import (
    Phase12FailureCode,
    build_policy_bundle,
    is_retrieval_retry_limit_reached,
    validate_retrieval_access,
)
from ..schemas import Phase12NodeDescriptor
from ..state import IngestionStatus, GraphExecutionStatus, Phase12GraphState
from ._helpers import fail_node, succeed_node

NODE_NAME = "rag_decision"
NODE_SPEC = Phase12NodeDescriptor(
    name=NODE_NAME,
    description="Decide whether role-safe retrieval should run, be skipped, or be strengthened later.",
    wrapped_modules=["app.langgraph.policy"],
    reads=["rag_required", "pipeline_context.rag", "ingestion_status", "retrieval_attempts"],
    writes=["rag_required", "warnings", "failure_reasons"],
)


async def rag_decision_node(state: Phase12GraphState) -> Phase12GraphState:
    """Make a bounded RAG execution decision without calling retrieval.

    Inputs inspected:
    - role-aware retrieval policy
    - current RAG context
    - evidence sufficiency and retrieval retry count

    Outputs updated:
    - `rag_required`
    - warnings and conservative retrieval-related failure reasons
    - decision and node traces

    This node intentionally does not execute retrieval.
    """

    state.set_current_node(NODE_NAME)
    policy_bundle = build_policy_bundle(state.user_role)

    retrieval_decision = validate_retrieval_access(
        role=state.user_role,
        requested_audience=state.role,
    )
    if not retrieval_decision.allowed:
        return fail_node(
            state,
            node_name=NODE_NAME,
            failure_code=retrieval_decision.failure_code or Phase12FailureCode.ROLE_BLOCKED,
            reason=retrieval_decision.reason or "Retrieval access is not allowed.",
            final_status=GraphExecutionStatus.BLOCKED,
            detail=retrieval_decision.metadata,
        )

    if state.rag_completed and state.rag_result.weak_retrieval:
        state.rag_required = True
        state.add_warning("Existing retrieval context is weak; a stronger retrieval pass may be needed later.")
        return succeed_node(
            state,
            node_name=NODE_NAME,
            decision_type="rag_decision",
            branch="strengthen_existing_rag_later",
            reason="An existing RAG context is present but marked weak.",
            detail={"retrieved_chunk_count": len(state.retrieved_chunks)},
            confidence=state.rag_result.confidence_score,
        )

    if state.rag_completed:
        state.rag_required = False
        return succeed_node(
            state,
            node_name=NODE_NAME,
            decision_type="rag_decision",
            branch="reuse_existing_rag",
            reason="A packaged RAG context is already available.",
            detail={"retrieved_chunk_count": len(state.retrieved_chunks)},
            confidence=state.rag_result.confidence_score,
        )

    if not state.rag_required:
        return succeed_node(
            state,
            node_name=NODE_NAME,
            decision_type="rag_decision",
            branch="skip_rag_not_required",
            reason="RAG is not required for the current graph state.",
        )

    if is_retrieval_retry_limit_reached(state, limits=policy_bundle.limits):
        state.rag_required = False
        state.add_failure_reason(Phase12FailureCode.RETRIEVAL_REPAIR_FAILED.value)
        return succeed_node(
            state,
            node_name=NODE_NAME,
            decision_type="rag_decision",
            branch="skip_rag_retry_limit",
            reason="Retrieval retry budget has been exhausted.",
            detail={"retrieval_attempts": state.retrieval_attempts},
        )

    if state.ingestion_status == IngestionStatus.MISSING:
        state.rag_required = False
        state.add_failure_reason(Phase12FailureCode.EVIDENCE_INSUFFICIENT.value)
        return succeed_node(
            state,
            node_name=NODE_NAME,
            decision_type="rag_decision",
            branch="skip_rag_insufficient_evidence",
            reason="Evidence is too limited to support retrieval.",
            confidence=state.evidence_quality_score,
        )

    if state.ingestion_status == IngestionStatus.PARTIAL or state.evidence_quality_score < 0.35:
        state.rag_required = True
        state.add_warning("Evidence is weak; retrieval should stay targeted and may need strengthening later.")
        return succeed_node(
            state,
            node_name=NODE_NAME,
            decision_type="rag_decision",
            branch="strengthen_rag_later",
            reason="Evidence is available but not strong enough for a confident first-pass retrieval.",
            detail={"retrieval_attempts": state.retrieval_attempts},
            confidence=state.evidence_quality_score,
        )

    state.rag_required = True
    return succeed_node(
        state,
        node_name=NODE_NAME,
        decision_type="rag_decision",
        branch="run_rag_later",
        reason="Role-safe retrieval should run later using the existing RAG path.",
        detail={"retrieval_attempts": state.retrieval_attempts},
        confidence=max(state.evidence_quality_score, 0.5),
    )
