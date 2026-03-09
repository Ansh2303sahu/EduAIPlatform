from __future__ import annotations

# Student feedback categories (you can expand later)
FEEDBACK_CATEGORIES = {
    "grammar": 0,
    "structure": 1,
    "clarity": 2,
    "evidence": 3,
    "argument": 4,
    "other": 5,
}

def map_feedback_category(name: str) -> int:
    k = (name or "").strip().lower()
    return FEEDBACK_CATEGORIES.get(k, FEEDBACK_CATEGORIES["other"])
