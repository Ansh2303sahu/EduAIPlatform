"""ML inference node for Phase 12.

Wraps the ML request helpers already exposed through
`app.services.report_generation_support` so Phase 12 does not create a second
ML call path.
"""

from ..adapters import report_support
from ..schemas import Phase12NodeDescriptor
from ..state import Phase12GraphState
from ..tracing.graph_trace import record_event

NODE_NAME = "ml_inference"
NODE_SPEC = Phase12NodeDescriptor(
    name=NODE_NAME,
    description="Call the existing student/professor ML inference services.",
    wrapped_modules=["app.services.report_generation_support"],
    reads=["pipeline_context.ingestion", "pipeline_context.analysis_type"],
    writes=["pipeline_context.ml_raw"],
)


async def run(state: Phase12GraphState) -> Phase12GraphState:
    """Populate raw ML output using the existing Phase 10 support layer."""

    if state.role == "student":
        state.pipeline_context.ml_raw = await report_support.call_student_ml_for_state(state)
    else:
        state.pipeline_context.ml_raw = await report_support.call_professor_ml_for_state(state)
    record_event(
        state,
        NODE_NAME,
        {
            "role": state.role,
            "analysis_type": state.pipeline_context.analysis_type.value,
        },
    )
    return state
