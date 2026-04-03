from __future__ import annotations

from app.core.config import settings
from app.rag.schemas import ConfidenceLabel, RetrievedChunk


def confidence_from_chunks(chunks: list[RetrievedChunk]) -> tuple[float, ConfidenceLabel]:
    if not chunks:
        return 0.0, 'low'

    weights = [1.0, 0.8, 0.6, 0.4, 0.3]
    weighted_total = 0.0
    total_weight = 0.0
    for idx, chunk in enumerate(chunks):
        weight = weights[idx] if idx < len(weights) else 0.25
        weighted_total += chunk.score * weight
        total_weight += weight

    avg_score = weighted_total / max(total_weight, 1e-6)
    official_bonus = 0.10 if any(c.is_official for c in chunks) else 0.0
    diversity_bonus = min(len({c.document_id for c in chunks}) * 0.035, 0.14)
    category_bonus = 0.04 if len({c.category for c in chunks}) >= 2 else 0.0
    score = max(0.0, min(1.0, avg_score + official_bonus + diversity_bonus + category_bonus))

    if score >= 0.75:
        return score, 'high'
    if score >= settings.rag_min_confidence:
        return score, 'medium'
    return score, 'low'


def safe_review_required(confidence_label: ConfidenceLabel) -> bool:
    return confidence_label == 'low'
