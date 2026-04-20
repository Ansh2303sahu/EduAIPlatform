"""ML context node for Phase 12.

Wraps the existing LangChain ML normalization functions so graph state reuses
the same role-aware ML shaping and explanatory context text already used in
Phase 10.
"""

from app.langchain.enums import DecisionSource
from app.langchain.services.ml_context_builder import (
    normalize_professor_ml_context,
    normalize_student_ml_context,
)

from ..schemas import Phase12NodeDescriptor
from ..state import Phase12GraphState
from ..tracing.graph_trace import record_event

NODE_NAME = "ml_context"
NODE_SPEC = Phase12NodeDescriptor(
    name=NODE_NAME,
    description="Normalize raw ML outputs into the shared PipelineContext ML structures.",
    wrapped_modules=["app.langchain.services.ml_context_builder"],
    reads=["pipeline_context.ml_raw", "pipeline_context.analysis_type"],
    writes=["pipeline_context.ml_result", "pipeline_context.ml_context_text"],
)


async def run(state: Phase12GraphState) -> Phase12GraphState:
    """Normalize ML responses into the shared Phase 10 pipeline context."""

    if state.role == "student":
        result = normalize_student_ml_context(state.pipeline_context.ml_raw)
    else:
        result = normalize_professor_ml_context(state.pipeline_context.ml_raw)
    text = result.context_text
    state.pipeline_context.ml_result = result
    state.pipeline_context.ml_context_text = text
    state.pipeline_context.execution_meta.decision_source = DecisionSource.HYBRID
    record_event(
        state,
        NODE_NAME,
        {
            "summary_present": bool(text),
            "ml_result_keys": sorted(result.model_dump(mode="json").keys()),
        },
    )
    return state
