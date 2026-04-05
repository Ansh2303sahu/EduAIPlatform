from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


AudienceType = Literal["student", "professor"]
ConfidenceLabel = Literal["high", "medium", "low"]


class RetrievalFilters(BaseModel):
    category: str | None = None
    version: str | None = None
    status: str | None = "active"
    is_official: bool | None = None


class RetrievalQuery(BaseModel):
    audience: AudienceType
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)
    ml_signals: dict[str, Any] = Field(default_factory=dict)
    submission_type: str | None = None
    analysis_type: str | None = None
    preferred_categories: list[str] = Field(default_factory=list)
    official_bias: float = Field(default=0.0, ge=0.0, le=1.0)
    # Submission-aware fields for targeted query building.
    # text_excerpt: cleaned first ~600 chars of the submission body.
    # keywords: topic-specific terms extracted from the submission.
    # Both are optional and backward-compatible (no existing code is broken).
    title_hint: str | None = None
    text_excerpt: str | None = None
    keywords: list[str] = Field(default_factory=list)
    mode: str | None = None
    degraded_input: bool = False


class CitationOut(BaseModel):
    title: str
    section: str
    document_id: str
    chunk_id: str
    category: str | None = None
    score: float | None = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    section: str
    category: str
    audience: AudienceType
    content: str
    score: float
    source_priority: int = 0
    is_official: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalTrace(BaseModel):
    audience: AudienceType
    query: str
    collection_name: str | None = None
    rewritten_queries: list[str] = Field(default_factory=list)
    expanded_queries: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    final_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_titles: list[str] = Field(default_factory=list)
    selected_titles: list[str] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    cache_hit: bool = False
    detected_topic: str | None = None
    ml_signal_summary: str | None = None
    integrity_focus: bool = False
    requested_top_k: int | None = None
    effective_top_k: int | None = None
    token_budget: int | None = None
    # Retrieval diagnostics (added for targeted query audit)
    applied_filters: dict[str, Any] = Field(default_factory=dict)
    final_categories: list[str] = Field(default_factory=list)
    reranking_changed_order: bool = False
    keywords_used: list[str] = Field(default_factory=list)
    text_excerpt: str | None = None
    title_hint: str | None = None
    mode: str | None = None
    degraded_input: bool = False
    initial_candidate_count: int = 0
    confidence_score: float | None = None
    confidence_label: str | None = None
    eligibility_summary: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    ok: bool = True
    audience: AudienceType
    query: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    citations: list[CitationOut] = Field(default_factory=list)
    confidence_score: float = 0.0
    confidence_label: ConfidenceLabel = "low"
    trace: RetrievalTrace
    safe_review: bool = False
