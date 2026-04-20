"""Retrieval node for Phase 12.

Wraps the existing RAG packaging flow used by Phase 10. This keeps LangGraph
aligned with the same submission-aware retrieval and trace contracts instead of
adding a separate retrieval implementation.
"""

import time

from app.langchain.services.retrieval_packager import summarize_rag_trace

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

    t0 = time.perf_counter()
    state.retrieval_attempts += 1
    payload, rag_context = rag.pack_rag_for_state(state)
    state.pipeline_context.rag = rag_context
    trace_summary = summarize_rag_trace(rag_context.trace)
    state.pipeline_context.execution_meta.retrieval_debug = {
        **trace_summary,
        "query": str(payload.get("query") or trace_summary.get("query") or ""),
        "chunk_count": int(rag_context.chunk_count or len(rag_context.retrieved_chunks)),
        "weak_retrieval": bool(rag_context.weak_retrieval),
        "safe_review": bool(rag_context.safe_review),
        "confidence_score": float(rag_context.confidence_score or 0.0),
        "confidence_label": str(rag_context.confidence_label or "low"),
    }
    state.pipeline_context.timings_ms["rag"] = int((time.perf_counter() - t0) * 1000)
    record_event(
        state,
        NODE_NAME,
        {
            "citations": len(rag_context.citations),
            "retrieval_mode": rag_context.trace.get("mode"),
        },
    )
    return state
