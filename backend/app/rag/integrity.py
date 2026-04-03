from __future__ import annotations

from typing import Iterable

from app.rag.schemas import RetrievedChunk


ACADEMIC_INTEGRITY_TERMS = {
    'plagiarism',
    'citation',
    'referencing',
    'original work',
    'academic misconduct',
}


def detect_integrity_relevance(query: str, chunks: Iterable[RetrievedChunk]) -> bool:
    haystack = ' '.join([query.lower(), *[c.content.lower() for c in chunks]])
    return any(term in haystack for term in ACADEMIC_INTEGRITY_TERMS)
