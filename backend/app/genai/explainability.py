"""Explainability helpers for Phase 15/16."""

from __future__ import annotations

from app.genai.config import genai_settings
from app.genai.schemas import (
    ConfidenceBand,
    EvidenceReference,
    ExplanationTab,
    FeatureContribution,
    RagTraceEntry,
    ReasoningStep,
    SourcesTab,
)
from app.langgraph.state import Phase12GraphState


def confidence_band(score: float) -> ConfidenceBand:
    """Map a unit confidence score to a compact confidence band."""

    if score >= genai_settings.confidence_high_threshold:
        return "high"
    if score >= genai_settings.confidence_medium_threshold:
        return "medium"
    return "low"


def rag_evidence_references(state: Phase12GraphState) -> list[EvidenceReference]:
    """Build evidence references only from retrieved chunks/citations."""

    refs: list[EvidenceReference] = []
    seen: set[tuple[str, str]] = set()
    for chunk in state.retrieved_chunks[: genai_settings.max_evidence_references]:
        if not isinstance(chunk, dict):
            continue
        source = str(
            chunk.get("document_title")
            or chunk.get("source")
            or chunk.get("title")
            or "Knowledge base source"
        )
        chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or "")
        key = (source, chunk_id)
        if key in seen:
            continue
        seen.add(key)
        relevance_raw = (
            chunk.get("rerank_score")
            or chunk.get("score")
            or chunk.get("similarity")
            or 0.0
        )
        try:
            relevance = max(0.0, min(1.0, float(relevance_raw)))
        except (TypeError, ValueError):
            relevance = 0.0
        excerpt = str(chunk.get("content") or chunk.get("snippet") or "")[:260]
        refs.append(
            EvidenceReference(
                source=source,
                chunk_id=chunk_id,
                relevance_score=relevance,
                excerpt=excerpt,
                category=str(chunk.get("category") or ""),
            )
        )
    return refs


def approximate_feature_importance(state: Phase12GraphState) -> list[FeatureContribution]:
    """Approximate feature contributions from available ML and evidence signals."""

    candidates: list[FeatureContribution] = []
    if state.ml_completed:
        label = state.ml_result.predicted_label or state.ml_result.predicted_band or "ml_signal"
        candidates.append(
            FeatureContribution(
                feature=f"ml:{label}",
                contribution=min(1.0, max(0.15, state.ml_confidence)),
                direction="positive" if state.ml_confidence >= 0.5 else "mixed",
                explanation="Phase 6 ML predictions influenced the explanation and confidence calibration.",
            )
        )
        if state.ml_result.disagreement_markers:
            candidates.append(
                FeatureContribution(
                    feature="ml:disagreement_markers",
                    contribution=min(1.0, 0.12 * len(state.ml_result.disagreement_markers)),
                    direction="negative",
                    explanation="Disagreement markers reduced certainty and triggered conservative wording.",
                )
            )
    if state.evidence_quality_score > 0.0:
        candidates.append(
            FeatureContribution(
                feature="evidence_quality",
                contribution=min(1.0, state.evidence_quality_score),
                direction="positive" if state.evidence_quality_score >= 0.4 else "mixed",
                explanation="Evidence richness directly shaped how specific the report could be.",
            )
        )
    if state.rag_completed:
        candidates.append(
            FeatureContribution(
                feature="rag_grounding",
                contribution=min(1.0, max(state.rag_result.confidence_score, 0.15)),
                direction="positive" if not state.rag_result.weak_retrieval else "mixed",
                explanation="Retrieved guidance influenced grounded evidence references and confidence.",
            )
        )
    if state.safe_mode:
        candidates.append(
            FeatureContribution(
                feature="safe_mode",
                contribution=0.35,
                direction="negative",
                explanation="Safe mode reduced assertiveness because the evidence or system confidence was limited.",
            )
        )
    return candidates[: genai_settings.explanation_max_features]


def reasoning_steps(state: Phase12GraphState) -> list[ReasoningStep]:
    """Build a compact step-by-step reasoning summary."""

    steps = [
        ReasoningStep(
            step="Ingestion",
            detail=state.evidence_summary or "Submission evidence was loaded and summarized.",
            confidence=state.evidence_quality_score,
        ),
        ReasoningStep(
            step="ML calibration",
            detail=(
                "Phase 6 ML signals were used as calibration hints."
                if state.ml_completed
                else "No strong ML calibration signal was available."
            ),
            confidence=state.ml_confidence if state.ml_completed else None,
        ),
        ReasoningStep(
            step="RAG grounding",
            detail=(
                f"Retrieved {len(state.retrieved_chunks)} knowledge-base chunks for grounding."
                if state.rag_completed
                else "No additional RAG grounding was available."
            ),
            confidence=state.rag_result.confidence_score if state.rag_completed else None,
        ),
        ReasoningStep(
            step="Generation",
            detail="A structured draft was generated, critiqued, and refined before validation.",
            confidence=state.final_confidence if state.final_report else None,
        ),
    ]
    return steps[: genai_settings.max_reasoning_steps]


def build_explanation_tab(
    state: Phase12GraphState,
    *,
    hybrid_summary: str,
    counterfactual: str,
) -> ExplanationTab:
    """Create the explanation tab from state and derived summaries."""

    band = confidence_band(state.final_confidence)
    return ExplanationTab(
        reasoning_steps=reasoning_steps(state),
        feature_importance=approximate_feature_importance(state),
        confidence_band=band,
        confidence_summary=(
            f"Final confidence is {band} ({state.final_confidence:.2f}) based on evidence quality, "
            "ML calibration, grounding strength, and critique validation."
        ),
        hybrid_summary=hybrid_summary,
        counterfactual_explanation=counterfactual,
    )


def build_sources_tab(state: Phase12GraphState) -> SourcesTab:
    """Create the sources tab from RAG citations and trace data."""

    refs = rag_evidence_references(state)
    trace_entries = [
        RagTraceEntry(
            source=ref.source,
            chunk_id=ref.chunk_id,
            relevance_score=ref.relevance_score,
            category=ref.category,
        )
        for ref in refs
    ]
    source_summary = (
        f"{len(refs)} evidence references were surfaced from the current RAG context."
        if refs
        else "No grounded knowledge-base evidence was available; the output relied more heavily on submission evidence."
    )
    return SourcesTab(
        evidence_references=refs,
        rag_trace=trace_entries,
        source_summary=source_summary,
    )
