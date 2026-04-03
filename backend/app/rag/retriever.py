from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, cast

from app.core.config import settings
from app.rag.cache.cache_keys import build_retrieval_cache_key
from app.rag.cache.retrieval_cache import InMemoryRetrievalCache
from app.rag.citations import citations_from_chunks
from app.rag.config import get_rag_runtime_config
from app.rag.confidence import confidence_from_chunks, safe_review_required
from app.rag.guards import validate_retrieval_request
from app.rag.integrity import detect_integrity_relevance
from app.rag.query_builder import build_queries
from app.rag.reranker import rerank_chunks
from app.rag.retrieval.adaptive_depth import choose_top_k
from app.rag.retrieval.context_budget import trim_chunks_to_budget
from app.rag.retrieval.ml_guidance import summarize_ml_signals
from app.rag.security.versioning import is_active_version
from app.rag.trace import build_trace
from app.rag.schemas import (
    RetrievedChunk,
    RetrievalQuery,
    RetrievalResult,
)
from app.rag.topic_detector import detect_topic
from app.rag.vector_store import get_vector_store


_RETRIEVAL_CACHE = InMemoryRetrievalCache(ttl_seconds=settings.rag_cache_ttl_seconds)
_MMR_LAMBDA_MULT = 0.35
_MMR_FETCH_K_MIN = 12
_MMR_FETCH_K_MAX = 24


_TOPIC_CATEGORY_MAP: dict[str, dict[str, list[str]]] = {
    "student": {
        "citation": ["referencing", "academic_integrity"],
        "structure": ["writing"],
        "clarity": ["writing", "critical_thinking"],
        "integrity": ["academic_integrity", "referencing"],
    },
    "professor": {
        "rubric": ["rubrics", "academic_quality"],
        "policy": ["marking_policy", "rubrics"],
        "moderation": ["moderation", "marking_policy"],
        "feedback": ["feedback_templates", "academic_quality"],
    },
}


def _clamp_unit_interval(score: float) -> float:
    return max(0.0, min(1.0, float(score)))


def _select_relevance_score_fn(store: Any) -> Callable[[float], float] | None:
    selector = getattr(store, "_select_relevance_score_fn", None)
    if not callable(selector):
        return None

    try:
        score_fn = selector()
    except Exception:
        return None

    if not callable(score_fn):
        return None
    return cast(Callable[[float], float], score_fn)


def normalize_score(
    score: float,
    relevance_score_fn: Callable[[float], float] | None = None,
) -> float:
    raw_score = float(score)

    if relevance_score_fn is not None:
        return _clamp_unit_interval(relevance_score_fn(raw_score))

    if 0.0 <= raw_score <= 1.0:
        return 1.0 - raw_score
    return 1.0 / (1.0 + max(raw_score, 0.0))


def _metadata_filter_dict(req: RetrievalQuery) -> dict[str, Any] | None:
    conditions: list[dict[str, Any]] = []

    if req.filters.status is not None:
        conditions.append({"status": req.filters.status})
    if req.filters.category is not None:
        conditions.append({"category": req.filters.category})
    if req.filters.version is not None:
        conditions.append({"version": req.filters.version})
    if req.filters.is_official is not None:
        conditions.append({"is_official": req.filters.is_official})

    conditions.append({"audience": req.audience})

    if len(conditions) == 1:
        return conditions[0]
    if not conditions:
        return None
    return {"$and": conditions}


def _metadata_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _apply_topic_preferences(req: RetrievalQuery, detected_topic: str | None) -> RetrievalQuery:
    if req.preferred_categories or not detected_topic:
        return req

    audience_map = _TOPIC_CATEGORY_MAP.get(req.audience, {})
    preferred_categories = audience_map.get(detected_topic, [])
    if not preferred_categories:
        return req

    return req.model_copy(update={"preferred_categories": preferred_categories})


def _apply_context_limits(chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    limit = min(top_k, settings.rag_context_max_chunks)
    limited = chunks[:limit]
    if settings.rag_enable_contextual_compression:
        limited = trim_chunks_to_budget(
            limited,
            max_tokens=settings.rag_context_max_tokens,
        )
    return limited[:limit]


def _cache_clone(result: RetrievalResult) -> RetrievalResult:
    return result.model_copy(deep=True)


def _chunk_merge_key(document_id: str, chunk_id: str) -> tuple[str, str]:
    return (document_id, chunk_id)


def _doc_merge_key(metadata: dict[str, Any] | None) -> tuple[str, str]:
    md = metadata or {}
    return (
        str(md.get("document_id") or ""),
        str(md.get("chunk_id") or ""),
    )


def similarity_search_clean(
    store: Any,
    query: str,
    *,
    k: int = 4,
    metadata_filter: dict[str, Any] | None = None,
) -> list[tuple[Any, float]]:
    search_kwargs: dict[str, Any] = {"k": k}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    relevance_score_fn = _select_relevance_score_fn(store)
    results = store.similarity_search_with_score(query, **search_kwargs)
    return [
        (doc, normalize_score(raw_score, relevance_score_fn))
        for doc, raw_score in results
    ]


def mmr_search_clean(
    store: Any,
    query: str,
    *,
    k: int = 4,
    fetch_k: int = 12,
    metadata_filter: dict[str, Any] | None = None,
) -> list[tuple[Any, float]]:
    fetch_k = max(k, min(fetch_k, _MMR_FETCH_K_MAX))
    scored_results = similarity_search_clean(
        store,
        query,
        k=fetch_k,
        metadata_filter=metadata_filter,
    )

    mmr_search = getattr(store, "max_marginal_relevance_search", None)
    if not callable(mmr_search):
        return scored_results[:k]

    try:
        mmr_docs = mmr_search(
            query,
            k=k,
            fetch_k=max(fetch_k, _MMR_FETCH_K_MIN),
            lambda_mult=_MMR_LAMBDA_MULT,
            filter=metadata_filter,
        )
    except Exception:
        return scored_results[:k]

    if not mmr_docs:
        return scored_results[:k]

    scored_by_key: dict[tuple[str, str], tuple[Any, float]] = {}
    for doc, score in scored_results:
        key = _doc_merge_key(getattr(doc, "metadata", None))
        current = scored_by_key.get(key)
        if current is None or score > current[1]:
            scored_by_key[key] = (doc, score)

    selected: list[tuple[Any, float]] = []
    seen: set[tuple[str, str]] = set()
    for doc in mmr_docs:
        key = _doc_merge_key(getattr(doc, "metadata", None))
        if key in seen:
            continue
        seen.add(key)
        selected.append(scored_by_key.get(key, (doc, 0.0)))

    return selected[:k]


def _select_diverse_chunks(chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    if len(chunks) <= top_k:
        return chunks

    selected: list[RetrievedChunk] = []
    per_doc_counts: dict[str, int] = defaultdict(int)
    max_per_doc_passes = [1, 2, 99]

    for limit in max_per_doc_passes:
        for chunk in chunks:
            if chunk in selected:
                continue
            if per_doc_counts[chunk.document_id] >= limit:
                continue
            selected.append(chunk)
            per_doc_counts[chunk.document_id] += 1
            if len(selected) >= top_k:
                return selected

    return selected[:top_k]


def retrieve(req: RetrievalQuery) -> RetrievalResult:
    runtime = get_rag_runtime_config()
    req = validate_retrieval_request(req.model_copy(deep=True))
    requested_top_k = req.top_k

    if settings.rag_enable_adaptive_depth:
        req.top_k = choose_top_k(req)

    detected_topic = detect_topic(req.query, req.audience) if settings.rag_enable_topic_detection else None
    req = _apply_topic_preferences(req, detected_topic)
    ml_signal_summary = summarize_ml_signals(req.ml_signals) if settings.rag_enable_ml_guided_retrieval else ""
    cache_key = build_retrieval_cache_key(req)

    if settings.rag_cache_enabled:
        cached = _RETRIEVAL_CACHE.get(cache_key)
        if isinstance(cached, RetrievalResult):
            return _cache_clone(
                cached.model_copy(
                    update={
                        "trace": cached.trace.model_copy(
                            update={
                                "cache_hit": True,
                                "requested_top_k": requested_top_k,
                                "effective_top_k": req.top_k,
                                "detected_topic": detected_topic,
                                "ml_signal_summary": ml_signal_summary,
                                "token_budget": settings.rag_context_max_tokens if settings.rag_enable_contextual_compression else None,
                            }
                        )
                    }
                )
            )

    store = get_vector_store(req.audience)
    queries = build_queries(req)
    metadata_filter = _metadata_filter_dict(req)

    merged: dict[tuple[str, str], RetrievedChunk] = {}
    retrieved_ids: list[str] = []
    retrieved_scores: list[float] = []

    initial_k = min(
        max(req.top_k * (4 if settings.rag_enable_multi_query else 3), runtime.top_k_initial),
        max(runtime.top_k_initial * 2, 20),
    )

    collection_name = (
        runtime.student_collection
        if req.audience == "student"
        else runtime.professor_collection
    )

    for q in queries:
        # Pull raw Chroma distances and normalize them locally so framework
        # relevance warnings do not fire before we can clamp the values.
        results = mmr_search_clean(
            store,
            q,
            k=initial_k,
            fetch_k=max(initial_k * 2, _MMR_FETCH_K_MIN),
            metadata_filter=metadata_filter,
        )

        for doc, normalized_score in results:
            md = dict(doc.metadata or {})
            if not is_active_version(md.get("status")):
                continue

            chunk_id = str(md.get("chunk_id") or "")
            if not chunk_id:
                continue

            source_priority = int(md.get("source_priority") or 0)

            document_id = str(md.get("document_id") or "")

            item = RetrievedChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                document_title=str(md.get("document_title") or "Untitled"),
                section=str(md.get("section") or "unknown"),
                category=str(md.get("category") or "general"),
                audience=req.audience,
                content=doc.page_content,
                score=normalized_score,
                source_priority=source_priority,
                is_official=_metadata_bool(md.get("is_official", False)),
                metadata=md,
            )

            merge_key = _chunk_merge_key(document_id, chunk_id)
            current = merged.get(merge_key)
            if current is None or item.score > current.score:
                merged[merge_key] = item

            retrieved_ids.append(chunk_id)
            retrieved_scores.append(normalized_score)

    reranked_chunks = rerank_chunks(req, list(merged.values()))
    final_chunks = _select_diverse_chunks(reranked_chunks, req.top_k)
    final_chunks = _apply_context_limits(final_chunks, req.top_k)

    citations = citations_from_chunks(final_chunks)

    confidence_score, confidence_label = confidence_from_chunks(final_chunks)
    safe_review = safe_review_required(confidence_label)
    integrity_focus = detect_integrity_relevance(req.query, final_chunks)

    trace = build_trace(
        req=req,
        queries=queries,
        retrieved_chunk_ids=retrieved_ids,
        final_chunks=final_chunks,
        scores=retrieved_scores,
        collection_name=collection_name,
        cache_hit=False,
        detected_topic=detected_topic,
        ml_signal_summary=ml_signal_summary,
        integrity_focus=integrity_focus,
        requested_top_k=requested_top_k,
        effective_top_k=req.top_k,
        token_budget=settings.rag_context_max_tokens if settings.rag_enable_contextual_compression else None,
    )

    result = RetrievalResult(
        audience=req.audience,
        query=req.query,
        chunks=final_chunks,
        citations=citations,
        confidence_score=confidence_score,
        confidence_label=confidence_label,
        trace=trace,
        safe_review=safe_review,
    )

    if settings.rag_cache_enabled:
        _RETRIEVAL_CACHE.set(cache_key, _cache_clone(result))

    return result
