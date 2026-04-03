from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from app.rag.schemas import RetrievedChunk

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_./+-]*")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "your",
    "their",
    "about",
    "have",
    "will",
    "when",
    "where",
    "were",
    "which",
    "should",
    "could",
    "would",
    "being",
    "been",
    "than",
    "then",
    "them",
    "they",
    "there",
    "because",
    "while",
    "does",
    "used",
    "using",
}


def tokenize_text(text: str) -> list[str]:
    return [
        token
        for token in _TOKEN_RE.findall((text or "").lower())
        if len(token) > 2 and token not in _STOPWORDS
    ]


def tokenize_set(text: str) -> set[str]:
    return set(tokenize_text(text))


def keyword_overlap_score(query: str, content: str) -> float:
    q_terms = tokenize_text(query)
    c_terms = Counter(tokenize_text(content))
    if not q_terms:
        return 0.0
    hits = sum(1 for term in q_terms if c_terms.get(term, 0) > 0)
    return hits / len(q_terms)


def hybrid_overlap_score(
    query: str,
    content: str,
    metadata_text: str = "",
    *,
    content_weight: float = 0.65,
    metadata_weight: float = 0.35,
) -> float:
    content_score = keyword_overlap_score(query, content)
    metadata_score = keyword_overlap_score(query, metadata_text)
    return min(1.0, (content_score * content_weight) + (metadata_score * metadata_weight))


def rerank_with_keyword_overlap(query: str, chunks: Iterable[RetrievedChunk]) -> list[RetrievedChunk]:
    rescored: list[tuple[float, RetrievedChunk]] = []
    for chunk in chunks:
        lexical = keyword_overlap_score(query, chunk.content)
        rescored.append((min(1.0, chunk.score + lexical * 0.1), chunk))

    ordered = sorted(rescored, key=lambda item: item[0], reverse=True)
    output: list[RetrievedChunk] = []
    for score, chunk in ordered:
        output.append(chunk.model_copy(update={'score': score}))
    return output
