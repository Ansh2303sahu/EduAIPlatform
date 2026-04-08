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
    return await support.get_file_or_404(state.request.file_id, current_user)


async def build_ingestion_bundle_for_state(state: Phase12GraphState) -> IngestionBundle:
    """Build the standard ingestion bundle through report_generation_support."""

    file_row = state.file_row or await load_file_for_state(state)
    state.file_row = file_row
    return support.build_ingestion_bundle(file_row, state.request.file_id)


def ingestion_bundle_from_dict(data: dict[str, Any]) -> IngestionBundle:
    """Rehydrate a typed ingestion bundle from stored data."""

    return IngestionBundle.model_validate(data)


def detect_submission_kind(bundle: IngestionBundle) -> str:
    """Reuse the existing submission-kind detection heuristic."""

    return support.detect_submission_kind(bundle.combined_text())


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

    file_row = state.file_row or await load_file_for_state(state)
    current_user = build_current_user_from_state(state)
    return await support.call_student_ml(file_row, current_user)


async def call_professor_ml_for_state(state: Phase12GraphState) -> dict[str, Any]:
    """Call the existing professor ML inference flow."""

    file_row = state.file_row or await load_file_for_state(state)
    current_user = build_current_user_from_state(state)
    return await support.call_professor_ml(file_row, current_user)


def sha256_json(value: Any) -> str:
    """Reuse the existing report-support hashing helper."""

    return support.sha256_json(value)
