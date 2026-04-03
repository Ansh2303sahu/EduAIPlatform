from __future__ import annotations


def estimate_tokens(text: str) -> int:
    text = text or ''
    return max(1, len(text.split()) * 4 // 3) if text.strip() else 0
