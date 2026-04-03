from __future__ import annotations

from copy import deepcopy
from typing import Any


def inject_rag_fields(body: dict[str, Any], rag_payload: dict[str, Any]) -> dict[str, Any]:
    enriched = deepcopy(body)

    enriched["rag"] = rag_payload
    enriched["grounding_context"] = rag_payload.get("context", "")
    enriched["grounding_citations"] = rag_payload.get("citations", [])
    enriched["grounding_retrieved_chunks"] = rag_payload.get("retrieved_chunks", [])
    enriched["retrieval_confidence_score"] = rag_payload.get("confidence_score", 0.0)
    enriched["retrieval_confidence_label"] = rag_payload.get("confidence_label", "low")
    enriched["retrieval_safe_review"] = rag_payload.get("safe_review", False)
    enriched["retrieval_trace"] = rag_payload.get("trace", {})
    enriched["grounding_instruction"] = rag_payload.get("instruction", "")

    return enriched


def rag_meta_from_grounding_fields(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(body.get("rag")) or bool(body.get("grounding_context")),
        "confidence_score": body.get("retrieval_confidence_score", 0.0),
        "confidence_label": body.get("retrieval_confidence_label", "low"),
        "safe_review": bool(body.get("retrieval_safe_review", False)),
        "citations": body.get("grounding_citations", []),
        "retrieved_chunks": body.get("grounding_retrieved_chunks", []),
        "trace": body.get("retrieval_trace", {}),
    }


def storage_fields_from_rag_meta(rag_meta: dict[str, Any] | None) -> dict[str, Any]:
    rag = rag_meta if isinstance(rag_meta, dict) else {}
    return {
        "rag_trace": rag.get("trace", {}),
        "citations": rag.get("citations", []),
        "retrieval_confidence": rag.get("confidence_score", 0.0),
        "retrieval_confidence_label": rag.get("confidence_label", "low"),
        "safe_review": bool(rag.get("safe_review", False)),
        "retrieved_chunks": rag.get("retrieved_chunks", []),
    }
