"""Adapter for retrieval_packager and the existing RAG modules.

This wrapper keeps Phase 12 on top of the current submission-aware retrieval
stack. It does not replace or fork the Phase 9/10 retrieval implementation.
"""

from __future__ import annotations

from app.langchain.models import RagContext
from app.langchain.services.retrieval_packager import pack_professor_rag, pack_student_rag

from ..state import Phase12GraphState


def _default_query(state: Phase12GraphState) -> str:
    submission_kind = state.pipeline_context.submission_kind
    if state.role == "student":
        if submission_kind in {"project", "code"}:
            return "architecture implementation testing security"
        return "structure argument evidence referencing"
    if submission_kind in {"project", "code"}:
        return "rubric project architecture testing moderation"
    return "rubric policy moderation feedback"


def build_seed_payload_from_state(state: Phase12GraphState) -> dict[str, object]:
    """Build a compact retrieval payload reusing the current RAG contracts."""

    return {
        "submission_id": state.pipeline_context.submission_id,
        "ingestion": state.pipeline_context.ingestion.model_dump(mode="json")
        if state.pipeline_context.ingestion
        else {},
        "ml": state.pipeline_context.ml_result.model_dump(mode="json")
        if state.pipeline_context.ml_result
        else {},
        "analysis_type": state.pipeline_context.analysis_type.value,
        "submission_type": state.pipeline_context.submission_kind,
        "mode": state.pipeline_context.submission_kind,
        "query": _default_query(state),
    }


def pack_rag_for_state(state: Phase12GraphState) -> tuple[dict[str, object], RagContext]:
    """Package RAG context through the current Phase 10 retrieval wrapper."""

    payload = build_seed_payload_from_state(state)
    if state.role == "student":
        rag_context = pack_student_rag(payload)
    else:
        rag_context = pack_professor_rag(payload)
    return payload, rag_context


def summarize_rag_for_storage(rag_context: RagContext) -> dict[str, object]:
    """Return a JSON-safe summary of the packaged RAG context."""

    return rag_context.model_dump(mode="json")
