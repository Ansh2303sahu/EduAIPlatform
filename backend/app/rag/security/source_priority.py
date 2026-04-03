from __future__ import annotations


def boost_score_with_source_priority(score: float, source_priority: int) -> float:
    return min(1.0, float(score) + min(int(source_priority) / 1000.0, 0.15))
