from __future__ import annotations

from typing import Any

from app.rag.payloads import storage_fields_from_rag_meta


def build_storage_fields_from_rag_meta(rag_meta: dict[str, Any] | None) -> dict[str, Any]:
    return storage_fields_from_rag_meta(rag_meta)


def build_storage_fields_from_result(rag) -> dict[str, Any]:
    rag_meta = {
        "trace": rag.trace.model_dump(),
        "citations": [c.model_dump() for c in rag.citations],
        "confidence_score": rag.confidence_score,
        "confidence_label": rag.confidence_label,
        "safe_review": rag.safe_review,
        "retrieved_chunks": [c.model_dump() for c in rag.chunks],
    }
    return build_storage_fields_from_rag_meta(rag_meta)

def store_rag_result(supabase, report_id, rag):
    supabase.table("ai_reports").update(
        build_storage_fields_from_result(rag)
    ).eq("id", report_id).execute()
