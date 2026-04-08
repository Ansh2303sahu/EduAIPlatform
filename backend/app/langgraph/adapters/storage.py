"""Adapter for ai_reports persistence.

This wrapper keeps Phase 12 writes on top of the same `ai_reports` persistence
path used by Phase 7 and Phase 10. It wraps report_generation_support for row
insertion and `app.rag.store` for RAG storage field shaping.
"""

from __future__ import annotations

from app.rag.store import build_storage_fields_from_rag_meta
from app.services import report_generation_support as support

from ..state import Phase12GraphState
from ..tracing.model_versions import build_phase12_model_versions


def build_ai_report_row(state: Phase12GraphState) -> dict[str, object]:
    """Build a standard ai_reports row without changing existing storage shape."""

    report_json = state.pipeline_context.report or {}
    rag_meta = (
        state.pipeline_context.rag.model_dump(mode="json")
        if state.pipeline_context.rag
        else {}
    )
    row: dict[str, object] = {
        "file_id": state.pipeline_context.file_id,
        "submission_id": state.pipeline_context.submission_id,
        "role": state.role,
        "report_json": report_json,
        "report_hash": support.sha256_json(report_json),
        "prompt_hash": state.prompt_hash,
        "input_hash": state.input_hash,
        "model_versions": build_phase12_model_versions(state),
        "needs_review": bool(report_json.get("safety", {}).get("needs_review", False)),
    }
    row.update(build_storage_fields_from_rag_meta(rag_meta))
    return row


async def persist_ai_report(state: Phase12GraphState) -> dict[str, object]:
    """Persist the Phase 12 row through the existing report support insert helper."""

    row = build_ai_report_row(state)
    return await support.post_row("ai_reports", row)
