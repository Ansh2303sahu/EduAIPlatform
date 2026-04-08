"""Ingestion node for Phase 12.

Wraps `app.services.report_generation_support` so the graph uses the same file
loading, ingestion preparation, and submission-kind detection as Phase 10.
"""

from ..adapters import report_support
from ..schemas import Phase12NodeDescriptor
from ..state import Phase12GraphState
from ..tracing.graph_trace import record_event

NODE_NAME = "ingestion"
NODE_SPEC = Phase12NodeDescriptor(
    name=NODE_NAME,
    description="Load file row and ingestion bundle through report_generation_support.",
    wrapped_modules=["app.services.report_generation_support"],
    reads=["request.file_id", "request.submission_id"],
    writes=["file_row", "pipeline_context.ingestion", "pipeline_context.analysis_type"],
)


async def run(state: Phase12GraphState) -> Phase12GraphState:
    """Populate the graph state with the standard file row and ingestion bundle."""

    file_row = await report_support.load_file_for_state(state)
    ingestion_bundle = await report_support.build_ingestion_bundle_for_state(state)
    submission_kind = report_support.detect_submission_kind(ingestion_bundle)
    state.file_row = file_row
    state.pipeline_context.ingestion = ingestion_bundle
    state.pipeline_context.submission_id = (
        state.request.submission_id or ingestion_bundle.submission_id
    )
    state.apply_submission_kind(submission_kind)
    state.input_hash = report_support.sha256_json(
        {
            "file_id": state.request.file_id,
            "submission_id": state.pipeline_context.submission_id,
            "submission_kind": submission_kind,
            "ingestion": ingestion_bundle.model_dump(mode="json"),
        }
    )
    record_event(
        state,
        NODE_NAME,
        {
            "submission_kind": submission_kind,
            "submission_id": state.pipeline_context.submission_id,
        },
    )
    return state
