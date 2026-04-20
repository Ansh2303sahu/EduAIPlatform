"""Safe-mode decision node for Phase 12.

Decides whether restricted execution should be enabled using policy, evidence,
confidence, and existing failure signals only.
"""

from __future__ import annotations

from app.langchain.enums import ExecutionMode, PipelineMode
from app.langchain.services.prompt_sanitizer import sanitize_input

from ..policy import Phase12FailureCode, build_policy_bundle, validate_safe_mode_downgrade
from ..schemas import Phase12NodeDescriptor
from ..state import Phase12GraphState
from ._helpers import succeed_node

NODE_NAME = "safe_mode_decision"
NODE_SPEC = Phase12NodeDescriptor(
    name=NODE_NAME,
    description="Decide whether the graph should downgrade into safe mode.",
    wrapped_modules=[
        "app.langgraph.policy",
        "app.langchain.services.prompt_sanitizer",
    ],
    reads=["warnings", "failure_reasons", "pipeline_context", "evidence_quality_score"],
    writes=["pipeline_context.mode", "pipeline_context.execution_mode", "warnings"],
)


def _candidate_failure_code(state: Phase12GraphState) -> Phase12FailureCode | None:
    threshold = build_policy_bundle(state.user_role).role_policy.safe_mode_ml_confidence_threshold
    reasons = " ".join(state.failure_reasons).lower()
    if state.validation_status.value in {"invalid", "repaired"} or "validation_failed" in reasons:
        return Phase12FailureCode.VALIDATION_FAILED
    if "generation_failed" in reasons:
        return Phase12FailureCode.GENERATION_FAILED
    if "retrieval_repair_failed" in reasons:
        return Phase12FailureCode.RETRIEVAL_REPAIR_FAILED
    if state.rag_completed and state.rag_result.weak_retrieval:
        return Phase12FailureCode.RETRIEVAL_WEAK
    if "evidence_insufficient" in reasons or state.ingestion_status.value == "missing":
        return Phase12FailureCode.EVIDENCE_INSUFFICIENT
    if state.ml_completed and state.ml_confidence < threshold:
        return Phase12FailureCode.ML_LOW_CONFIDENCE
    return None


async def safe_mode_decision_node(state: Phase12GraphState) -> Phase12GraphState:
    """Decide whether restricted execution should be enabled.

    Inputs inspected:
    - policy-controlled safe-mode downgrade rules
    - evidence weakness and insufficiency
    - failure reasons and validation state
    - prompt-injection indicators in available evidence

    Outputs updated:
    - PipelineContext mode and execution mode
    - warnings describing why safe mode was or was not activated
    - decision and node traces

    This node intentionally does not call generation or other external services.
    """

    state.set_current_node(NODE_NAME)
    policy_bundle = build_policy_bundle(state.user_role)

    combined = "\n".join(
        part
        for part in (
            state.extracted_text[:1600],
            state.ocr_text[:800],
            state.transcript_text[:800],
            state.evidence_summary,
        )
        if part
    )
    if combined:
        _, injected, reason = sanitize_input(combined, max_chars=3200)
        if injected:
            state.pipeline_context.injection_detected = True
            state.pipeline_context.injection_reason = reason
            state.add_warning("Potential prompt-injection patterns were detected in submission evidence.")

    if state.safe_mode:
        return succeed_node(
            state,
            node_name=NODE_NAME,
            decision_type="safe_mode_decision",
            branch="already_safe",
            reason="Safe mode is already active.",
            safe_mode_triggered=True,
        )

    failure_code = _candidate_failure_code(state)
    downgrade_decision = validate_safe_mode_downgrade(
        state=state,
        failure_code=failure_code,
    )

    if downgrade_decision.allowed:
        state.pipeline_context.mode = PipelineMode.RESTRICTED
        state.pipeline_context.execution_mode = ExecutionMode.SAFE
        state.pipeline_context.execution_meta.execution_mode = ExecutionMode.SAFE
        state.add_warning("Safe mode activated for conservative downstream handling.")
        reason = downgrade_decision.reason or "Safe mode activated by policy."
        return succeed_node(
            state,
            node_name=NODE_NAME,
            decision_type="safe_mode_decision",
            branch="activate_safe_mode",
            reason=reason,
            detail={
                **downgrade_decision.metadata,
                "low_confidence_threshold": policy_bundle.role_policy.safe_mode_ml_confidence_threshold,
            },
            confidence=state.ml_confidence if state.ml_completed else state.evidence_quality_score,
            safe_mode_triggered=True,
        )

    state.pipeline_context.mode = PipelineMode.NORMAL
    state.pipeline_context.execution_mode = ExecutionMode.NORMAL
    state.pipeline_context.execution_meta.execution_mode = ExecutionMode.NORMAL
    return succeed_node(
        state,
        node_name=NODE_NAME,
        decision_type="safe_mode_decision",
        branch="remain_normal",
        reason=downgrade_decision.reason or "Safe mode was not required.",
        detail={
            **downgrade_decision.metadata,
            "low_confidence_threshold": policy_bundle.role_policy.safe_mode_ml_confidence_threshold,
        },
        confidence=state.ml_confidence if state.ml_completed else state.evidence_quality_score,
        safe_mode_triggered=False,
    )
