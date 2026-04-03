from __future__ import annotations

from app.rag.schemas import RetrievalQuery


def choose_top_k(req: RetrievalQuery) -> int:
    query_len = len(req.query.split())
    if query_len <= 8:
        return min(max(req.top_k, 3), 6)
    if query_len <= 25:
        return min(max(req.top_k, 4), 8)
    return min(max(req.top_k, 5), 10)
