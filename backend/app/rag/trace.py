from __future__ import annotations

from app.rag.schemas import RetrievalQuery, RetrievalTrace, RetrievedChunk


def build_trace(
    *,
    req: RetrievalQuery,
    queries: list[str],
    retrieved_chunk_ids: list[str],
    final_chunks: list[RetrievedChunk],
    scores: list[float],
    collection_name: str | None = None,
    cache_hit: bool = False,
    detected_topic: str | None = None,
    ml_signal_summary: str | None = None,
    integrity_focus: bool = False,
    requested_top_k: int | None = None,
    effective_top_k: int | None = None,
    token_budget: int | None = None,
) -> RetrievalTrace:
    return RetrievalTrace(
        audience=req.audience,
        query=req.query,
        collection_name=collection_name,
        rewritten_queries=queries,
        retrieved_chunk_ids=retrieved_chunk_ids,
        final_chunk_ids=[c.chunk_id for c in final_chunks],
        scores=scores,
        cache_hit=cache_hit,
        detected_topic=detected_topic,
        ml_signal_summary=ml_signal_summary,
        integrity_focus=integrity_focus,
        requested_top_k=requested_top_k,
        effective_top_k=effective_top_k,
        token_budget=token_budget,
    )
