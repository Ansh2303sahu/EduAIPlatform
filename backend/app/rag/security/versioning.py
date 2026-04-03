from __future__ import annotations


def is_active_version(status: str | None) -> bool:
    return (status or 'active').strip().lower() == 'active'
