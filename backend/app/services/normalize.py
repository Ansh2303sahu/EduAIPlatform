from __future__ import annotations

from typing import Any


def clamp_text(text: str, *, max_chars: int) -> tuple[str, bool]:
    if text is None:
        return "", False
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def sanitize_pg_text(s: str | None) -> str:
    # postgres TEXT cannot include NUL
    if not s:
        return ""
    return s.replace("\x00", "")


def clamp_list(items: list[Any], *, max_len: int) -> list[Any]:
    if not items:
        return []
    return items[:max_len]


def normalize_table(
    *,
    columns: list[Any],
    rows: list[list[Any]],
    max_cols: int,
    max_rows: int,
    max_cell_chars: int,
) -> tuple[list[str], list[list[str]]]:
    cols = [str(c)[:max_cell_chars] for c in (columns or [])[:max_cols]]

    out_rows: list[list[str]] = []
    for r in (rows or [])[:max_rows]:
        rr = []
        for c in (r or [])[:max_cols]:
            rr.append(str("" if c is None else c)[:max_cell_chars])
        out_rows.append(rr)

    return cols, out_rows
