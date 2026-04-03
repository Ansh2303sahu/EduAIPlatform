"""
Tests for fallback payload builders and execution logging helpers.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.langchain.enums import DecisionSource, ExecutionMode
from app.langchain.services.execution_logger import Phase10ExecutionLogger
from app.langchain.services.fallback_service import (
    build_professor_fallback_payload,
    build_student_fallback_payload,
)


def test_build_student_fallback_payload_sets_safe_mode_and_fallback_flags() -> None:
    payload = build_student_fallback_payload("weak_retrieval")

    assert payload.safe_mode is True
    assert payload.fallback_used is True
    assert payload.confidence <= 0.2
    assert payload.issues
    assert "retrieved evidence" in payload.issues[0].description.lower()


def test_build_professor_fallback_payload_sets_review_flags() -> None:
    payload = build_professor_fallback_payload("validation_failure")

    assert payload.safe_mode is True
    assert payload.fallback_used is True
    assert payload.needs_review is True
    assert payload.discrepancy_flag is True
    assert payload.rubric_breakdown[0].band == "Needs review"


def test_execution_logger_build_payload_tracks_failures_without_raw_output() -> None:
    execution_logger = Phase10ExecutionLogger(
        "req-1",
        "student",
        file_id="file-1",
        user_id="user-1",
        execution_mode=ExecutionMode.SAFE,
        decision_source=DecisionSource.HYBRID,
        store_raw_output=False,
        enable_logging=False,
    )

    execution_logger.set_model_used("mistral")
    execution_logger.mark_retry()
    execution_logger.mark_parse_failure("missing closing brace")
    execution_logger.mark_validation_failure("summary field missing")
    execution_logger.set_flags(weak_retrieval=True, low_confidence=True)
    execution_logger.mark_fallback_triggered("malformed_output")

    payload = execution_logger.build_payload(raw_output="secret raw output")

    assert payload["model_used"] == "mistral"
    assert payload["retry_count"] == 1
    assert payload["parse_failures"] == 1
    assert payload["validation_failures"] == 1
    assert payload["weak_retrieval"] is True
    assert payload["low_confidence"] is True
    assert payload["fallback_triggered"] is True
    assert payload["fallback_reason"] == "malformed_output"
    assert "raw_output_excerpt" not in payload


def test_execution_logger_raw_output_is_opt_in() -> None:
    execution_logger = Phase10ExecutionLogger(
        "req-2",
        "professor",
        store_raw_output=True,
        enable_logging=False,
    )

    payload = execution_logger.build_payload(raw_output="x" * 2000)

    assert "raw_output_excerpt" in payload
    assert len(payload["raw_output_excerpt"]) <= 1200


@pytest.mark.asyncio
async def test_execution_logger_persist_uses_audit_log_when_user_present() -> None:
    execution_logger = Phase10ExecutionLogger(
        "req-3",
        "student",
        file_id="file-3",
        user_id="user-3",
        enable_logging=True,
        store_raw_output=False,
    )
    execution_logger.set_model_used("claude-sonnet")
    execution_logger.mark_fallback_triggered("low_confidence")

    with patch("app.langchain.services.execution_logger.audit_log", new_callable=AsyncMock) as audit_mock:
        payload = await execution_logger.persist()

    audit_mock.assert_awaited_once()
    assert payload["request_id"] == "req-3"
    assert payload["fallback_reason"] == "low_confidence"
