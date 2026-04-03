"""
Retrieval packaging for Phase 10.

Takes Phase 9 retrieval payloads and converts them into a prompt-friendly
package with role isolation, deduplication, character-budget trimming, and
weak-retrieval signaling.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from hashlib import sha1
from typing import Any, Mapping, Sequence

from app.core.config import settings
from app.langchain.config import phase10_settings
from app.langchain.models import RagContext
from app.langchain.services.prompt_sanitizer import (
    sanitize_plain_text,
    sanitize_retrieved_text,
)
from app.rag.citations import citations_from_chunks
from app.rag.payloads import inject_rag_fields
from app.rag.retrieval.context_builder import (
    build_professor_rag_payload,
    build_student_rag_payload,
)
from app.rag.schemas import RetrievedChunk

logger = logging.getLogger("phase10.retrieval_packager")

_DEFAULT_PROMPT_CHAR_BUDGET = max(1500, min(phase10_settings.max_input_chars, 12_000))
_WEAK_RETRIEVAL_MIN_CONTEXT_CHARS = 220
_MIN_TRUNCATED_CHUNK_CHARS = 80


def package_retrieval_chunks(
    *,
    role: str,
    retrieved_chunks: Sequence[Mapping[str, Any]] | None,
    citations: Sequence[Mapping[str, Any]] | None = None,
    confidence_score: float = 0.0,
    confidence_label: str = "low",
    safe_review: bool = False,
    trace: Mapping[str, Any] | None = None,
    instruction: str = "",
    context_text: str = "",
    max_chars: int = _DEFAULT_PROMPT_CHAR_BUDGET,
) -> dict[str, Any]:
    """
    Package Phase 9 chunks into a prompt-ready retrieval bundle.

    This function is pure and testable: it accepts already-retrieved chunks and
    returns a normalized package without performing retrieval itself.
    """
    normalized_role = _normalize_role(role)
    chunks = _normalize_chunks(retrieved_chunks or [], role=normalized_role)
    deduped_chunks = _dedupe_chunks(chunks)
    selected_chunks = _select_chunks_for_budget(deduped_chunks, max_chars=max_chars)
    rendered_context = _render_context(selected_chunks)

    if not rendered_context:
        rendered_context = sanitize_retrieved_text(context_text, max_chars=max_chars)

    packaged_citations = _package_citations(
        citations=citations or [],
        selected_chunks=selected_chunks,
        role=normalized_role,
    )

    chunk_count = len(selected_chunks)
    weak_retrieval = _is_weak_retrieval(
        chunk_count=chunk_count,
        context_text=rendered_context,
        citations=packaged_citations,
        confidence_label=confidence_label,
        safe_review=safe_review,
    )

    packaged_trace = deepcopy(dict(trace or {}))
    packaged_trace.update(
        {
            "packaging_role": normalized_role,
            "packaging_char_budget": max_chars,
            "packaging_chunk_count": chunk_count,
            "packaging_context_chars": len(rendered_context),
            "packaging_weak_retrieval": weak_retrieval,
        }
    )

    packaged_instruction = sanitize_plain_text(
        instruction,
        max_chars=max(400, min(max_chars // 4, 2_000)),
        filter_injection_phrases=True,
    )

    return {
        "enabled": bool(rendered_context),
        "audience": normalized_role,
        "context": rendered_context,
        "context_text": rendered_context,
        "citations": packaged_citations,
        "retrieved_chunks": [chunk.model_dump() for chunk in selected_chunks],
        "chunk_count": chunk_count,
        "weak_retrieval": weak_retrieval,
        "confidence_score": _as_float(confidence_score, default=0.0),
        "confidence_label": _normalize_confidence_label(confidence_label),
        "safe_review": bool(safe_review or weak_retrieval),
        "trace": packaged_trace,
        "instruction": packaged_instruction,
    }


def package_retrieval_payload(
    rag_payload: Mapping[str, Any] | None,
    *,
    role: str,
    max_chars: int = _DEFAULT_PROMPT_CHAR_BUDGET,
) -> dict[str, Any]:
    """Package an existing Phase 9 payload without rerunning retrieval."""
    payload = dict(rag_payload or {})
    return package_retrieval_chunks(
        role=role,
        retrieved_chunks=payload.get("retrieved_chunks") or [],
        citations=payload.get("citations") or [],
        confidence_score=_as_float(payload.get("confidence_score"), default=0.0),
        confidence_label=_as_str(payload.get("confidence_label"), default="low"),
        safe_review=bool(payload.get("safe_review", False)),
        trace=_as_dict(payload.get("trace")),
        instruction=_as_str(payload.get("instruction")),
        context_text=_as_str(payload.get("context")),
        max_chars=max_chars,
    )


def pack_student_rag(llm_payload: dict[str, Any]) -> tuple[dict[str, Any], RagContext]:
    """
    Build, package, and inject student retrieval context.

    Returns ``(enriched_payload, rag_context)``.
    """
    if not settings.rag_enabled:
        logger.debug("pack_student_rag: RAG disabled, skipping retrieval")
        return llm_payload, RagContext(enabled=False)

    rag_payload = build_student_rag_payload(llm_payload)
    packaged_payload = package_retrieval_payload(rag_payload, role="student")
    enriched = inject_rag_fields(llm_payload, packaged_payload)
    return enriched, _rag_payload_to_context(packaged_payload)


def pack_professor_rag(llm_payload: dict[str, Any]) -> tuple[dict[str, Any], RagContext]:
    """
    Build, package, and inject professor retrieval context.

    Returns ``(enriched_payload, rag_context)``.
    """
    if not settings.rag_enabled:
        logger.debug("pack_professor_rag: RAG disabled, skipping retrieval")
        return llm_payload, RagContext(enabled=False)

    rag_payload = build_professor_rag_payload(llm_payload)
    packaged_payload = package_retrieval_payload(rag_payload, role="professor")
    enriched = inject_rag_fields(llm_payload, packaged_payload)
    return enriched, _rag_payload_to_context(packaged_payload)


def _rag_payload_to_context(rag_payload: Mapping[str, Any]) -> RagContext:
    """Map a packaged retrieval payload dict to a typed ``RagContext``."""
    context_text = _as_str(rag_payload.get("context_text") or rag_payload.get("context"))
    retrieved_chunks = rag_payload.get("retrieved_chunks") or []
    return RagContext(
        enabled=bool(rag_payload.get("enabled", True)),
        context=context_text,
        context_text=context_text,
        instruction=_as_str(rag_payload.get("instruction")),
        citations=list(rag_payload.get("citations") or []),
        retrieved_chunks=list(retrieved_chunks),
        chunk_count=_clamp_int(
            rag_payload.get("chunk_count"),
            default=len(retrieved_chunks) if isinstance(retrieved_chunks, list) else 0,
            minimum=0,
        ),
        weak_retrieval=bool(rag_payload.get("weak_retrieval", False)),
        confidence_score=_as_float(rag_payload.get("confidence_score"), default=0.0),
        confidence_label=_normalize_confidence_label(rag_payload.get("confidence_label")),
        safe_review=bool(rag_payload.get("safe_review", False)),
        trace=dict(rag_payload.get("trace") or {}),
    )


def _normalize_role(role: str) -> str:
    return "professor" if str(role).strip().lower() == "professor" else "student"


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_str(value: Any, *, default: str = "") -> str:
    text = str(value).strip() if value is not None else ""
    return text or default


def _as_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp_int(value: Any, *, default: int, minimum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        try:
            parsed = int(float(value))
        except (TypeError, ValueError):
            parsed = default
    return max(minimum, parsed)


def _normalize_confidence_label(value: Any) -> str:
    label = _as_str(value, default="low").lower()
    if label in {"high", "medium", "low"}:
        return label
    if label == "med":
        return "medium"
    return "low"


def _normalize_chunks(
    raw_chunks: Sequence[Mapping[str, Any]],
    *,
    role: str,
) -> list[RetrievedChunk]:
    normalized: list[RetrievedChunk] = []

    for item in raw_chunks:
        data = _as_dict(item)
        if not data:
            continue

        audience = _normalize_role(_as_str(data.get("audience"), default=role))
        if audience != role:
            continue

        content = sanitize_retrieved_text(_as_str(data.get("content")), max_chars=_DEFAULT_PROMPT_CHAR_BUDGET)
        if not content:
            continue

        chunk_payload = {
            "chunk_id": _as_str(data.get("chunk_id"), default=_fallback_chunk_id(data)),
            "document_id": _as_str(data.get("document_id"), default="unknown-document"),
            "document_title": sanitize_plain_text(_as_str(data.get("document_title"), default="Untitled"), max_chars=200),
            "section": sanitize_plain_text(_as_str(data.get("section"), default="unknown"), max_chars=120),
            "category": sanitize_plain_text(_as_str(data.get("category"), default="general"), max_chars=80),
            "audience": audience,
            "content": content,
            "score": _as_float(data.get("score"), default=0.0),
            "source_priority": _clamp_int(data.get("source_priority"), default=0, minimum=0),
            "is_official": bool(data.get("is_official", False)),
            "metadata": deepcopy(_as_dict(data.get("metadata"))),
        }

        try:
            normalized.append(RetrievedChunk.model_validate(chunk_payload))
        except Exception:
            continue

    return normalized


def _fallback_chunk_id(chunk: Mapping[str, Any]) -> str:
    fingerprint = "|".join(
        [
            _as_str(chunk.get("document_id")),
            _as_str(chunk.get("document_title")),
            _as_str(chunk.get("section")),
            _as_str(chunk.get("content"))[:120],
        ]
    )
    return sha1(fingerprint.encode("utf-8")).hexdigest()[:16]


def _dedupe_chunks(chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    best_by_key: dict[tuple[str, str], RetrievedChunk] = {}
    first_seen: dict[tuple[str, str], int] = {}

    for idx, chunk in enumerate(chunks):
        key = (chunk.document_id, chunk.chunk_id)
        if key not in first_seen:
            first_seen[key] = idx
        current = best_by_key.get(key)
        if current is None or chunk.score > current.score:
            best_by_key[key] = chunk

    return [
        best_by_key[key]
        for key in sorted(best_by_key, key=lambda item: first_seen[item])
    ]


def _render_context_block(index: int, chunk: RetrievedChunk) -> str:
    official = "official" if chunk.is_official else "supporting"
    return "\n".join(
        [
            f"[Source {index}]",
            f"Title: {chunk.document_title}",
            f"Section: {chunk.section}",
            f"Category: {chunk.category}",
            f"Type: {official}",
            f"Score: {chunk.score:.2f}",
            f"Content: {chunk.content}",
        ]
    )


def _truncate_chunk_for_budget(index: int, chunk: RetrievedChunk, remaining_chars: int) -> RetrievedChunk | None:
    overhead_chunk = chunk.model_copy(update={"content": ""})
    overhead = len(_render_context_block(index, overhead_chunk))
    available_content_chars = remaining_chars - overhead
    if available_content_chars < _MIN_TRUNCATED_CHUNK_CHARS:
        return None

    content = chunk.content[:available_content_chars].rstrip()
    if len(content) < len(chunk.content) and len(content) > 3:
        content = content[:-3].rstrip() + "..."
    if len(content) < _MIN_TRUNCATED_CHUNK_CHARS:
        return None
    return chunk.model_copy(update={"content": content})


def _select_chunks_for_budget(chunks: Sequence[RetrievedChunk], *, max_chars: int) -> list[RetrievedChunk]:
    selected: list[RetrievedChunk] = []
    used_chars = 0

    for chunk in chunks:
        index = len(selected) + 1
        block = _render_context_block(index, chunk)
        separator_len = 2 if selected else 0
        if used_chars + separator_len + len(block) <= max_chars:
            selected.append(chunk)
            used_chars += separator_len + len(block)
            continue

        remaining_chars = max_chars - used_chars - separator_len
        truncated = _truncate_chunk_for_budget(index, chunk, remaining_chars)
        if truncated is not None:
            selected.append(truncated)
        break

    return selected


def _render_context(chunks: Sequence[RetrievedChunk]) -> str:
    return "\n\n".join(
        _render_context_block(index, chunk)
        for index, chunk in enumerate(chunks, start=1)
    )


def _package_citations(
    *,
    citations: Sequence[Mapping[str, Any]],
    selected_chunks: Sequence[RetrievedChunk],
    role: str,
) -> list[dict[str, Any]]:
    selected_keys = {(chunk.document_id, chunk.chunk_id) for chunk in selected_chunks}
    packaged: list[dict[str, Any]] = []

    for item in citations:
        data = deepcopy(_as_dict(item))
        if not data:
            continue

        key = (_as_str(data.get("document_id")), _as_str(data.get("chunk_id")))
        if selected_keys and key not in selected_keys:
            continue

        data["title"] = sanitize_plain_text(_as_str(data.get("title"), default="Untitled"), max_chars=200)
        data["section"] = sanitize_plain_text(_as_str(data.get("section"), default="unknown"), max_chars=120)
        if "category" in data:
            data["category"] = sanitize_plain_text(_as_str(data.get("category")), max_chars=80)
        packaged.append(data)

    if packaged:
        return packaged

    return [citation.model_dump() for citation in citations_from_chunks(selected_chunks)]


def _is_weak_retrieval(
    *,
    chunk_count: int,
    context_text: str,
    citations: Sequence[Mapping[str, Any]],
    confidence_label: str,
    safe_review: bool,
) -> bool:
    if safe_review:
        return True
    if _normalize_confidence_label(confidence_label) == "low":
        return True
    if chunk_count == 0 or len(citations) == 0:
        return True
    if len(context_text.strip()) < _WEAK_RETRIEVAL_MIN_CONTEXT_CHARS:
        return True
    return False
