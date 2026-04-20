"""Adapter for `app.services.report_generation_support`.

This is the single Phase 12 bridge for ingestion loading, ML inference calls,
hashing, and shared file utilities. It intentionally wraps the current Phase 10
support layer instead of duplicating that logic.
"""

from __future__ import annotations

from typing import Any

from app.langchain.enums import AnalysisType
from app.langchain.models import IngestionBundle
from app.services import report_generation_support as support

from ..state import Phase12GraphState


def build_current_user_from_state(state: Phase12GraphState):
    """Build the standard backend user object for shared support helpers."""

    return state.build_current_user()


async def load_file_for_state(state: Phase12GraphState) -> dict[str, Any]:
    """Load the file row using the existing report support helper."""

    current_user = build_current_user_from_state(state)
    return await support.load_file(state.request.file_id, current_user)


async def build_ingestion_bundle_for_state(state: Phase12GraphState) -> IngestionBundle:
    """Build the standard ingestion bundle through report_generation_support."""

    existing = state.pipeline_context.ingestion
    if (
        existing.text_content
        or existing.ocr_text
        or existing.audio_transcript
        or existing.tables_json is not None
    ):
        return existing

    current_user = build_current_user_from_state(state)
    payload = await support.build_ingestion_bundle(state.request.file_id, current_user)
    bundle = IngestionBundle.model_validate(payload)
    state.pipeline_context.ingestion = bundle
    return bundle


def ingestion_bundle_from_dict(data: dict[str, Any]) -> IngestionBundle:
    """Rehydrate a typed ingestion bundle from stored data."""

    return IngestionBundle.model_validate(data)


def detect_submission_kind(bundle: IngestionBundle) -> str:
    """Reuse the existing submission-kind detection heuristic."""

    return support.detect_submission_kind(bundle.model_dump(mode="json"))


def classify_submission_form(bundle: IngestionBundle) -> str:
    """Return the fine-grained submission form used for prompt and RAG routing."""

    return support.classify_submission_form(bundle.model_dump(mode="json"))


def analysis_type_for_role(*, role: str, submission_kind: str) -> AnalysisType:
    """Map role and submission kind to the existing Phase 10 analysis types."""

    normalized = (submission_kind or "academic").strip().lower()
    if role == "student":
        return (
            AnalysisType.STUDENT_PROJECT
            if normalized in {"project", "code"}
            else AnalysisType.STUDENT_ACADEMIC
        )
    return (
        AnalysisType.PROFESSOR_PROJECT
        if normalized in {"project", "code"}
        else AnalysisType.PROFESSOR_ACADEMIC
    )


async def call_student_ml_for_state(state: Phase12GraphState) -> dict[str, Any]:
    """Call the existing student ML inference flow."""

    if state.pipeline_context.ml_raw:
        return dict(state.pipeline_context.ml_raw)
    current_user = build_current_user_from_state(state)
    ingestion = state.pipeline_context.ingestion.model_dump(mode="json")
    return await support.call_ai_student_multimodal(current_user, ingestion)


async def call_professor_ml_for_state(state: Phase12GraphState) -> dict[str, Any]:
    """Call the existing professor ML inference flow."""

    if state.pipeline_context.ml_raw:
        return dict(state.pipeline_context.ml_raw)
    current_user = build_current_user_from_state(state)
    ingestion = state.pipeline_context.ingestion.model_dump(mode="json")
    return await support.call_ai_professor_multimodal(current_user, ingestion)


def sha256_json(value: Any) -> str:
    """Reuse the existing report-support hashing helper."""

    return support.sha256_json(value)
