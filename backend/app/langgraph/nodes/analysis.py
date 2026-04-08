"""Analysis node for the Phase 15/16 generative pipeline."""

from __future__ import annotations

from app.langgraph.schemas import Phase12NodeDescriptor
from app.langgraph.state import Phase12GraphState
from ._helpers import succeed_node

NODE_NAME = "analysis"
NODE_SPEC = Phase12NodeDescriptor(
    name=NODE_NAME,
    description="Summarize evidence, ML, and RAG state into analysis hints for later planning/generation.",
    wrapped_modules=["app.langgraph.state"],
    reads=["pipeline_context.ingestion", "pipeline_context.ml_result", "pipeline_context.rag"],
    writes=["storage_payload", "decision_history"],
)


async def analysis_node(state: Phase12GraphState) -> Phase12GraphState:
    """Build a compact internal analysis summary without any external calls."""

    state.set_current_node(NODE_NAME)
    ml_summary = (
        {
            "predicted_label": state.ml_result.predicted_label,
            "predicted_band": state.ml_result.predicted_band,
            "confidence": state.ml_confidence,
            "disagreement_markers": list(state.ml_result.disagreement_markers),
        }
        if state.ml_completed
        else {}
    )
    rag_summary = {
        "chunk_count": len(state.retrieved_chunks),
        "weak_retrieval": state.rag_result.weak_retrieval,
        "confidence_score": state.rag_result.confidence_score,
    }
    evidence_summary = {
        "text_chars": len(state.extracted_text.strip()),
        "ocr_chars": len(state.ocr_text.strip()),
        "transcript_chars": len(state.transcript_text.strip()),
        "evidence_quality_score": state.evidence_quality_score,
    }
    state.storage_payload["analysis_bundle"] = {
        "role": state.role,
        "analysis_type": state.analysis_type.value,
        "submission_kind": state.pipeline_context.submission_kind,
        "ml": ml_summary,
        "rag": rag_summary,
        "evidence": evidence_summary,
    }
    return succeed_node(
        state,
        node_name=NODE_NAME,
        decision_type="analysis",
        branch="analysis_ready",
        reason="Evidence, ML, and RAG signals were summarized for planning.",
        detail=state.storage_payload["analysis_bundle"],
        confidence=max(state.evidence_quality_score, state.ml_confidence),
    )
