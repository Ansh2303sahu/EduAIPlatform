from __future__ import annotations

from app.rag.schemas import RetrievedChunk
from app.rag.utils.token_counter import estimate_tokens


def trim_chunks_to_budget(chunks: list[RetrievedChunk], max_tokens: int = 1400) -> list[RetrievedChunk]:
    total = 0
    kept: list[RetrievedChunk] = []
    for chunk in chunks:
        size = estimate_tokens(chunk.content)
        if kept and total + size > max_tokens:
            break
        kept.append(chunk)
        total += size
    return kept
