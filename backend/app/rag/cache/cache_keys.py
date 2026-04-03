from __future__ import annotations

import json

from app.rag.schemas import RetrievalQuery
from app.rag.utils.hash_utils import sha256_text


def build_retrieval_cache_key(req: RetrievalQuery) -> str:
    payload = {
        'audience': req.audience,
        'query': req.query,
        'top_k': req.top_k,
        'filters': req.filters.model_dump(),
        'ml_signals': req.ml_signals,
        'submission_type': req.submission_type,
        'analysis_type': req.analysis_type,
        'preferred_categories': req.preferred_categories,
        'official_bias': req.official_bias,
    }
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return sha256_text(data)
