"""Retrieval node for Phase 12.

Wraps the existing RAG packaging flow used by Phase 10. This keeps LangGraph
aligned with the same submission-aware retrieval and trace contracts instead of
adding a separate retrieval implementation.
"""

from ..adapters import rag
from ..schemas import Phase12NodeDescriptor
from ..state import Phase12GraphState
from ..tracing.graph_trace import record_event

NODE_NAME = "retrieval"
NODE_SPEC = Phase12NodeDescriptor(
    name=NODE_NAME,
    description="Build and package RAG context using the existing retrieval_packager and RAG modules.",
    wrapped_modules=[
        "app.langchain.services.retrieval_packager",
        "app.rag.retrieval.context_builder",
    ],
    reads=["pipeline_context.ingestion", "pipeline_context.ml_result", "pipeline_context.analysis_type"],
    writes=["pipeline_context.rag", "pipeline_context.execution_meta.retrieval_debug"],
)


async def run(state: Phase12GraphState) -> Phase12GraphState:
    """Populate packaged RAG context through the current Phase 10 retrieval stack."""

    payload, rag_context = rag.pack_rag_for_state(state)
    state.pipeline_context.rag = rag_context
    state.pipeline_context.execution_meta.retrieval_debug = payload
    record_event(
        state,
        NODE_NAME,
        {
            "citations": len(rag_context.citations),
            "retrieval_mode": rag_context.trace.get("mode"),
        },
    )
    return state
