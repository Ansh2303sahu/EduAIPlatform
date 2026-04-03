from __future__ import annotations

from typing import Any


def summarize_ml_signals(ml_signals: dict[str, Any]) -> str:
    if not ml_signals:
        return ''
    parts: list[str] = []
    for key, value in ml_signals.items():
        parts.append(f'{key}={value}')
    return '; '.join(parts)
