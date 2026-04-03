from __future__ import annotations


def normalize_whitespace(text: str) -> str:
    return ' '.join((text or '').replace('\x00', ' ').split())
