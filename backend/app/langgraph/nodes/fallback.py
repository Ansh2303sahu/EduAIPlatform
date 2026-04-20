"""Fallback node for Phase 12.

Wraps the existing LangChain fallback payload builders so Phase 12 can keep the
same safe degraded-output behavior as Phase 10 when validation fails.
"""

from app.langchain.enums import DecisionSource, ExecutionMode
from app.langchain.parsers.normalizer import normalize_professor_payload, normalize_student_payload
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
    reason = "validation_failure" if validation_result and validation_result.errors else "internal_error"
    if state.role == "student":
        native = build_student_fallback_payload(
            reason,
            detail="LangGraph fallback applied after generation validation failed.",
            citations=state.rag_result.citations,
        )
        if hasattr(native, "model_dump"):
            state.pipeline_context.report = normalize_student_payload(
                native.model_dump(mode="json")
            ).model_dump(exclude={"rag_meta"})
        else:
            state.pipeline_context.report = dict(native or {})
    else:
        native = build_professor_fallback_payload(
            reason,
            detail="LangGraph fallback applied after generation validation failed.",
            citations=state.rag_result.citations,
        )
        if hasattr(native, "model_dump"):
            state.pipeline_context.report = normalize_professor_payload(
                native.model_dump(mode="json")
            ).model_dump(exclude={"rag_meta"})
        else:
            state.pipeline_context.report = dict(native or {})
    state.pipeline_context.execution_meta.decision_source = DecisionSource.FALLBACK
    state.pipeline_context.execution_meta.fallback_used = True
    state.pipeline_context.execution_meta.execution_mode = ExecutionMode.FALLBACK
    record_event(state, NODE_NAME, {"used": True, "reason": reason})
    return state
