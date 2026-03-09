from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class TextExample:
    text: str
    label: int
    source: str
    meta: Dict[str, Any]

@dataclass
class PairExample:
    text_a: str
    text_b: str
    label: int
    source: str
    meta: Dict[str, Any]

def require_cols(row: Dict[str, Any], cols: list[str], source: str) -> None:
    missing = [c for c in cols if c not in row or row[c] is None]
    if missing:
        raise ValueError(f"[{source}] missing columns: {missing}")
