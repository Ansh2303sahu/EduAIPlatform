from __future__ import annotations

from typing import Iterable

from app.rag.schemas import RetrievalResult


def summarize_retrieval_results(results: Iterable[RetrievalResult]) -> dict[str, float]:
    results = list(results)
    if not results:
        return {
            'count': 0,
            'avg_confidence': 0.0,
            'high_confidence_rate': 0.0,
            'safe_review_rate': 0.0,
            'avg_chunks': 0.0,
        }

    count = len(results)
    avg_confidence = sum(r.confidence_score for r in results) / count
    high_confidence_rate = sum(1 for r in results if r.confidence_label == 'high') / count
    safe_review_rate = sum(1 for r in results if r.safe_review) / count
    avg_chunks = sum(len(r.chunks) for r in results) / count
    return {
        'count': float(count),
        'avg_confidence': avg_confidence,
        'high_confidence_rate': high_confidence_rate,
        'safe_review_rate': safe_review_rate,
        'avg_chunks': avg_chunks,
    }
