from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


PLACEHOLDER_REPORT = {
    "summary": "Automated review generated with limited confidence.",
    "issues": [],
    "strengths": [],
    "architecture_review": {
        "overview": "Not assessed.",
        "backend": "Not assessed.",
        "frontend": "Not assessed.",
        "database": "Not assessed.",
        "security": "Not assessed.",
    },
    "implementation_review": {
        "features_built": [],
        "technical_quality": "Not assessed.",
        "integration_quality": "Not assessed.",
    },
    "evaluation_review": {
        "testing_present": "Not assessed.",
        "limitations": "Not assessed.",
        "academic_quality": "Not assessed.",
    },
    "improvement_plan": [],
    "checklist": [],
    "confidence": {"mode": "normal", "overall": 0.75},
    "model_agreement": {
        "ml_confidence": 0.0,
        "llm_confidence": 0.0,
        "final_confidence": 0.0,
    },
    "safety": {"needs_review": False, "reason": ""},
}

GOOD_REPORT = {
    "summary": "A useful student review with grounded detail.",
    "issues": [{"title": "Testing coverage", "evidence": "Testing evidence is limited.", "severity": "med"}],
    "strengths": [{"title": "Architecture clarity", "evidence": "The report explains the stack clearly."}],
    "architecture_review": {
        "overview": "The system is structured coherently.",
        "backend": "FastAPI handles the service layer.",
        "frontend": "React handles the UI layer.",
        "database": "PostgreSQL is described adequately.",
        "security": "Basic security considerations are discussed.",
    },
    "implementation_review": {
        "features_built": ["Dashboard", "Authentication"],
        "technical_quality": "Implementation is mostly coherent.",
        "integration_quality": "Components integrate reasonably well.",
    },
    "evaluation_review": {
        "testing_present": "Some testing evidence is provided.",
        "limitations": "Testing depth is still limited.",
        "academic_quality": "The write-up is technically clear.",
    },
    "improvement_plan": [
        {"action": "Add deeper tests", "why": "Coverage is light.", "how": "Introduce integration tests.", "priority": 1}
    ],
    "checklist": [{"item": "Add integration tests", "done": False}],
    "confidence": {"mode": "normal", "overall": 0.78},
    "model_agreement": {
        "ml_confidence": 0.75,
        "llm_confidence": 0.78,
        "final_confidence": 0.77,
    },
    "safety": {"needs_review": False, "reason": ""},
}


@pytest.mark.asyncio
async def test_phase7_student_generate_flags_placeholder_report(
    async_client,
    student_a_token,
    monkeypatch,
):
    import app.api.phase7 as phase7

    monkeypatch.setattr(phase7, "_rate_limit", lambda _user_id: None)
    monkeypatch.setattr(
        phase7,
        "_load_file",
        AsyncMock(return_value={"id": "file-1", "submission_id": "sub-1"}),
    )
    monkeypatch.setattr(
        phase7,
        "_build_ingestion_bundle",
        AsyncMock(
            return_value={
                "text_content": "A student submission about a small web app.",
                "ocr_text": "",
                "audio_transcript": "",
                "tables_json": None,
            }
        ),
    )
    monkeypatch.setattr(
        phase7,
        "_call_ai_student_multimodal",
        AsyncMock(
            return_value={
                "feedback_category": "project_review",
                "quality_band": "high",
                "confidence_0_to_4": 4,
            }
        ),
    )
    monkeypatch.setattr(phase7.settings, "rag_enabled", False, raising=False)
    monkeypatch.setattr(
        phase7,
        "_call_llm",
        AsyncMock(
            return_value=(
                PLACEHOLDER_REPORT,
                "mistral:latest",
                {"primary_model": "mistral:latest", "fallback_model": "gemma3:latest"},
            )
        ),
    )

    async def fake_post_row(_table: str, payload: dict):
        return {
            "id": "report-1",
            "created_at": "2026-04-03T12:00:00Z",
            **payload,
        }

    monkeypatch.setattr(phase7, "_post_row", fake_post_row)

    response = await async_client.post(
        "/api/phase7/student/generate",
        headers={"Authorization": f"Bearer {student_a_token}"},
        json={"file_id": "file-1", "force": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["report"]["safety"]["needs_review"] is True
    assert body["report"]["safety"]["reason"] == "low_content_quality"
    assert body["report"]["confidence"]["overall"] == pytest.approx(0.35)
    assert body["stored"]["needs_review"] is True
    assert body["stored"]["model_versions"]["quality_gate"]["degraded_placeholder"] is True
    assert body["stored"]["model_versions"]["quality_gate"]["reason"] == "low_content_quality"
    assert body["stored"]["model_versions"]["agreement"]["final_confidence"] == pytest.approx(0.7)
    assert body["stored"]["model_versions"]["llm_primary"] == "mistral:latest"
    assert body["stored"]["model_versions"]["llm_fallback"] == "gemma3:latest"


@pytest.mark.asyncio
async def test_phase7_latest_student_prefers_older_good_row_over_newer_degraded(
    async_client,
    student_a_token,
    monkeypatch,
):
    import app.api.phase7 as phase7

    monkeypatch.setattr(
        phase7,
        "_load_file",
        AsyncMock(return_value={"id": "file-1", "submission_id": "sub-1"}),
    )
    monkeypatch.setattr(
        phase7,
        "_get_rows",
        AsyncMock(
            return_value=[
                {
                    "id": "new-degraded",
                    "file_id": "file-1",
                    "role": "student",
                    "created_at": "2026-04-03T12:05:00Z",
                    "needs_review": True,
                    "report_json": {
                        **PLACEHOLDER_REPORT,
                        "safety": {"needs_review": True, "reason": "low_content_quality"},
                    },
                    "model_versions": {
                        "quality_gate": {"degraded_placeholder": True, "reason": "low_content_quality"}
                    },
                },
                {
                    "id": "older-good",
                    "file_id": "file-1",
                    "role": "student",
                    "created_at": "2026-04-03T12:00:00Z",
                    "needs_review": False,
                    "report_json": GOOD_REPORT,
                    "model_versions": {},
                },
            ]
        ),
    )

    response = await async_client.get(
        "/api/phase7/latest/student/file-1",
        headers={"Authorization": f"Bearer {student_a_token}"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["found"] is True
    assert body["item"]["id"] == "older-good"
    assert body["item"]["report_json"]["summary"] == GOOD_REPORT["summary"]
    assert body["selection_metadata"]["preferred_non_degraded"] is True
    assert body["selection_metadata"]["total_reports_considered"] == 2


@pytest.mark.asyncio
async def test_phase7_latest_student_returns_newest_when_only_degraded_rows_exist(
    async_client,
    student_a_token,
    monkeypatch,
):
    import app.api.phase7 as phase7

    monkeypatch.setattr(
        phase7,
        "_load_file",
        AsyncMock(return_value={"id": "file-1", "submission_id": "sub-1"}),
    )
    monkeypatch.setattr(
        phase7,
        "_get_rows",
        AsyncMock(
            return_value=[
                {
                    "id": "new-degraded",
                    "file_id": "file-1",
                    "role": "student",
                    "created_at": "2026-04-03T12:05:00Z",
                    "needs_review": True,
                    "report_json": {
                        **PLACEHOLDER_REPORT,
                        "safety": {"needs_review": True, "reason": "low_content_quality"},
                    },
                    "model_versions": {
                        "quality_gate": {"degraded_placeholder": True, "reason": "low_content_quality"}
                    },
                },
                {
                    "id": "older-degraded",
                    "file_id": "file-1",
                    "role": "student",
                    "created_at": "2026-04-03T12:00:00Z",
                    "needs_review": True,
                    "report_json": {
                        **PLACEHOLDER_REPORT,
                        "safety": {"needs_review": True, "reason": "low_content_quality"},
                    },
                    "model_versions": {
                        "quality_gate": {"degraded_placeholder": True, "reason": "low_content_quality"}
                    },
                },
            ]
        ),
    )

    response = await async_client.get(
        "/api/phase7/latest/student/file-1",
        headers={"Authorization": f"Bearer {student_a_token}"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["found"] is True
    assert body["item"]["id"] == "new-degraded"
    assert body["item"]["report_json"]["safety"]["reason"] == "low_content_quality"
    assert body["selection_metadata"]["preferred_non_degraded"] is False
    assert body["selection_metadata"]["total_reports_considered"] == 2


@pytest.mark.asyncio
async def test_phase7_latest_student_single_good_row_unchanged(
    async_client,
    student_a_token,
    monkeypatch,
):
    import app.api.phase7 as phase7

    monkeypatch.setattr(
        phase7,
        "_load_file",
        AsyncMock(return_value={"id": "file-1", "submission_id": "sub-1"}),
    )
    monkeypatch.setattr(
        phase7,
        "_get_rows",
        AsyncMock(
            return_value=[
                {
                    "id": "only-good",
                    "file_id": "file-1",
                    "role": "student",
                    "created_at": "2026-04-03T12:00:00Z",
                    "needs_review": False,
                    "report_json": GOOD_REPORT,
                    "model_versions": {},
                }
            ]
        ),
    )

    response = await async_client.get(
        "/api/phase7/latest/student/file-1",
        headers={"Authorization": f"Bearer {student_a_token}"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["found"] is True
    assert body["item"]["id"] == "only-good"
    assert body["item"]["report_json"]["summary"] == GOOD_REPORT["summary"]
    assert body["selection_metadata"]["preferred_non_degraded"] is False
    assert body["selection_metadata"]["total_reports_considered"] == 1
