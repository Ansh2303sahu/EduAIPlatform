from __future__ import annotations

def sanitize_text(s: str | None) -> str:
    """
    Postgres TEXT cannot contain NUL (\x00).
    Also normalize weird control chars safely.
    """
    if not s:
        return ""
    return s.replace("\x00", "")
