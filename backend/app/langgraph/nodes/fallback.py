"""Fallback node for Phase 12.

Wraps the existing LangChain fallback payload builders so Phase 12 can keep the
same safe degraded-output behavior as Phase 10 when validation fails.
"""

from app.langchain.enums import DecisionSource
from app.langchain.services.fallback_service import (
    build_professor_fallback_payload,
    build_student_fallback_payload,
)

from ..schemas import Phase12NodeDescriptor
from ..state import Phase12GraphState
from ..tracing.graph_trace import record_event

NODE_NAME = "fallback"
NODE_SPEC = Phase12NodeDescriptor(
    name=NODE_NAME,
    description="Provide the existing Phase 10 safe fallback payload when generation or validation fails.",
    wrapped_modules=["app.langchain.services.fallback_service"],
    reads=["pipeline_context.report", "pipeline_context.validation_result"],
    writes=["pipeline_context.report", "pipeline_context.execution_meta.decision_source"],
)


async def run(state: Phase12GraphState) -> Phase12GraphState:
    """Ensure a schema-compatible report exists even if validation failed."""

    validation_result = state.pipeline_context.validation_result
    report = state.pipeline_context.report
    if report and validation_result and validation_result.valid:
        record_event(state, NODE_NAME, {"used": False})
        return state
    if state.role == "student":
        state.pipeline_context.report = build_student_fallback_payload(
            state.pipeline_context,
            reason="phase12_graph_fallback",
        )
    else:
        state.pipeline_context.report = build_professor_fallback_payload(
            state.pipeline_context,
            reason="phase12_graph_fallback",
        )
    state.pipeline_context.execution_meta.decision_source = DecisionSource.FALLBACK.value
    state.pipeline_context.execution_meta.fallback_used = True
    record_event(state, NODE_NAME, {"used": True})
    return state
