"""Evidence sufficiency classification node for Phase 12.

Assesses whether the currently available evidence is broadly sufficient, weak,
or insufficient using only the wrapped PipelineContext inputs and existing
state fields. It does not run retrieval or generation.
"""

from __future__ import annotations

from ..policy import Phase12FailureCode
from ..schemas import Phase12NodeDescriptor
from ..state import IngestionStatus, Phase12GraphState
from ._helpers import succeed_node

NODE_NAME = "evidence_sufficiency"
NODE_SPEC = Phase12NodeDescriptor(
    name=NODE_NAME,
    description="Classify current evidence as sufficient, weak, or insufficient.",
    wrapped_modules=["app.langgraph.state"],
    reads=["pipeline_context.ingestion", "evidence_summary", "evidence_quality_score"],
    writes=["ingestion_status", "evidence_summary", "evidence_quality_score", "warnings"],
)


def _table_count(tables: dict[str, object]) -> int:
    if not tables:
        return 0
    total = 0
    for value in tables.values():
        if isinstance(value, list):
            total += len(value)
        else:
            total += 1
    return total


async def evidence_sufficiency_node(state: Phase12GraphState) -> Phase12GraphState:
    """Classify evidence sufficiency using existing ingested modalities only.

    Inputs inspected:
    - extracted text, OCR text, transcript text
    - tables metadata
    - existing evidence summary and quality score

    Outputs updated:
    - evidence summary and evidence quality score
    - ingestion status
    - warnings and conservative failure reasons

    This node intentionally does not perform retrieval or call external systems.
    """

    state.set_current_node(NODE_NAME)

    text_chars = len(state.extracted_text.strip())
    ocr_chars = len(state.ocr_text.strip())
    transcript_chars = len(state.transcript_text.strip())
    table_count = _table_count(state.tables_json)
    total_chars = text_chars + ocr_chars + transcript_chars
    modality_count = sum(
        1
        for present in (
            text_chars > 0,
            ocr_chars > 0,
            transcript_chars > 0,
            table_count > 0,
        )
        if present
    )

    char_score = min(1.0, total_chars / 2200.0)
    modality_bonus = min(0.2, max(0, modality_count - 1) * 0.05)
    table_bonus = 0.1 if table_count > 0 else 0.0
    summary_bonus = 0.05 if state.evidence_summary.strip() else 0.0
    computed_score = round(
        max(
            state.evidence_quality_score,
            min(1.0, char_score + modality_bonus + table_bonus + summary_bonus),
        ),
        3,
    )
    state.evidence_quality_score = computed_score

    if total_chars >= 1200 or computed_score >= 0.60 or (total_chars >= 700 and modality_count >= 2):
        classification = "sufficient"
        state.ingestion_status = IngestionStatus.READY
    elif total_chars >= 250 or table_count > 0 or computed_score >= 0.20:
        classification = "weak"
        state.ingestion_status = IngestionStatus.PARTIAL
        state.add_warning("Evidence is limited; downstream nodes should stay conservative.")
    else:
        classification = "insufficient"
        state.ingestion_status = (
            IngestionStatus.MISSING if total_chars == 0 and table_count == 0 else IngestionStatus.PARTIAL
        )
        state.add_warning("Evidence is insufficient for confident downstream decisions.")
        state.add_failure_reason(Phase12FailureCode.EVIDENCE_INSUFFICIENT.value)

    state.evidence_summary = (
        "Evidence profile: "
        f"text={text_chars} chars, ocr={ocr_chars} chars, transcript={transcript_chars} chars, "
        f"tables={table_count}, classification={classification}."
    )

    return succeed_node(
        state,
        node_name=NODE_NAME,
        decision_type="evidence_sufficiency",
        branch=classification,
        reason=f"Evidence classified as {classification}.",
        detail={
            "evidence_quality_score": computed_score,
            "total_chars": total_chars,
            "modality_count": modality_count,
            "table_count": table_count,
        },
        confidence=computed_score,
    )
