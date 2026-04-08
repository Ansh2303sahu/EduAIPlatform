"""Persistence node for Phase 12.

Wraps the current ai_reports storage path through report_generation_support and
RAG storage helpers. This keeps Phase 12 aligned with existing persistence and
report retrieval behavior.
"""

from ..adapters import storage
from ..schemas import Phase12NodeDescriptor
from ..state import Phase12GraphState
from ..tracing.graph_trace import record_event

NODE_NAME = "persistence"
NODE_SPEC = Phase12NodeDescriptor(
    name=NODE_NAME,
    description="Persist the generated report through the existing ai_reports storage path.",
    wrapped_modules=[
        "app.services.report_generation_support",
        "app.rag.store",
    ],
    reads=["pipeline_context.report", "pipeline_context.rag", "pipeline_context.execution_meta"],
    writes=["storage_payload"],
)


async def run(state: Phase12GraphState) -> Phase12GraphState:
    """Persist the standard ai_reports row using the existing backend storage path."""

    stored = await storage.persist_ai_report(state)
    state.storage_payload["stored_row"] = stored
    record_event(
        state,
        NODE_NAME,
        {"stored": True, "has_id": bool(stored.get("id"))},
    )
    return state
