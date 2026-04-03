from __future__ import annotations

from app.rag.schemas import RetrievalQuery
from app.rag.utils.text_cleaner import normalize_whitespace


MAX_QUERY_LENGTH = 12000


def validate_retrieval_request(req: RetrievalQuery) -> RetrievalQuery:
    req.query = normalize_whitespace(req.query)
    if not req.query:
        raise ValueError('query must not be empty')
    if len(req.query) > MAX_QUERY_LENGTH:
        req.query = req.query[:MAX_QUERY_LENGTH]
    if req.top_k < 1:
        req.top_k = 1
    if req.top_k > 20:
        req.top_k = 20
    return req
