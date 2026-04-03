from __future__ import annotations

from typing import Iterable

from app.rag.schemas import CitationOut, RetrievedChunk


def citations_from_chunks(chunks: Iterable[RetrievedChunk]) -> list[CitationOut]:
    return [
        CitationOut(
            title=chunk.document_title,
            section=chunk.section,
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            category=chunk.category,
            score=chunk.score,
        )
        for chunk in chunks
    ]


def format_citation_labels(citations: Iterable[CitationOut]) -> list[str]:
    return [f"{c.title} | section {c.section}" for c in citations]
