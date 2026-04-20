"""Decision node for Phase 12.

This node prepares execution-mode decisions while reusing the existing prompt
sanitizer and Phase 10 execution metadata fields. It is intentionally heuristic
and internal at this scaffolding stage.
"""

from app.langchain.enums import DecisionSource, PipelineMode, execution_mode_from_pipeline_mode
from app.langchain.services.prompt_sanitizer import sanitize_input

from ..schemas import Phase12NodeDescriptor
from ..state import Phase12GraphState
from ..tracing.graph_trace import record_event

NODE_NAME = "decision"
NODE_SPEC = Phase12NodeDescriptor(
    name=NODE_NAME,
    description="Set internal execution mode using existing prompt sanitization and lightweight heuristics.",
    wrapped_modules=["app.langchain.services.prompt_sanitizer"],
    reads=["pipeline_context.ingestion", "pipeline_context.ml_result", "pipeline_context.rag"],
    writes=["pipeline_context.mode", "pipeline_context.execution_mode", "pipeline_context.execution_meta"],
)


def _should_restrict(state: Phase12GraphState) -> bool:
    ml_result = state.pipeline_context.ml_result
    rag_context = state.pipeline_context.rag
    if ml_result is None or rag_context is None:
        return False
    if state.role == "student":
        return ml_result.confidence_0_to_4 <= 1
    return str(ml_result.moderation_consistency or "").strip().lower() == "low"


async def run(state: Phase12GraphState) -> Phase12GraphState:
    """Apply safe execution heuristics without changing any public contracts."""

    combined = state.pipeline_context.ingestion.combined_text()
    _, flagged = sanitize_input(combined)
    restricted = flagged or _should_restrict(state)
    mode = PipelineMode.RESTRICTED if restricted else PipelineMode.NORMAL
    execution_mode = execution_mode_from_pipeline_mode(mode)
    state.pipeline_context.mode = mode
    state.pipeline_context.execution_mode = execution_mode
    state.pipeline_context.execution_meta.execution_mode = execution_mode
    state.pipeline_context.execution_meta.decision_source = (
        DecisionSource.ML if restricted else DecisionSource.HYBRID
    )
    record_event(
        state,
        NODE_NAME,
        {
            "restricted": restricted,
            "execution_mode": execution_mode.value,
        },
    )
    return state
