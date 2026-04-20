from __future__ import annotations

from types import SimpleNamespace
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

GOOD_PROFESSOR_REPORT = {
    "summary": "The script is coherent overall, but the evidence depth in the discussion section needs closer moderation.",
    "evaluator_overview": "The opening and structure are clear, though later sections are less analytical.",
    "rubric_alignment": ["Argument", "Evidence use", "Structure"],
    "rubric_breakdown": [
        {
            "criterion": "Argument",
            "band": "Merit",
            "justification": "The thesis is clear and mostly sustained.",
        },
        {
            "criterion": "Evidence use",
            "band": "Pass",
            "justification": "Examples are present, but comparison and evaluation are limited in the later sections.",
        },
    ],
    "strengths": [{"title": "Clear thesis", "detail": "The position is explicit from the beginning."}],
    "concerns": [
        {
            "title": "Late-section analysis is thin",
            "detail": "The discussion names evidence but rarely weighs it.",
            "severity": "med",
        }
    ],
    "section_observations": [
        {
            "section_name": "Discussion",
            "observation": "Relevant examples are selected.",
            "concern": "Comparison is limited.",
            "next_step": "Check whether the mark should be moderated for analysis depth.",
        }
    ],
    "moderation_notes": [
        {
            "risk": "Evidence depth",
            "note": "Review whether the final band should be limited by the weaker discussion section.",
        }
    ],
    "action_recommendations": ["Moderate the discussion section against the evidence-use descriptor."],
    "confidence_explanation": "Confidence is moderate because the opening is strong but the later evidence chain is thinner.",
    "safety": {"needs_review": False, "reason": ""},
}


@pytest.mark.asyncio
async def test_phase7_student_generate_flags_placeholder_report(
    async_client,
    student_a_token,
    monkeypatch,
):
    import app.api.phase7 as phase7
    from app.langchain.models import RagContext

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

    fake_state = SimpleNamespace(
        pipeline_context=SimpleNamespace(
            request_id="graph-req-1",
            report={
                **PLACEHOLDER_REPORT,
                "safety": {"needs_review": True, "reason": "low_content_quality"},
                "confidence": {"mode": "normal", "overall": 0.35},
            },
            rag=RagContext(enabled=False),
            ml_raw={
                "feedback_category": "project_review",
                "quality_band": "high",
                "confidence_0_to_4": 4,
            },
        ),
        storage_payload={
            "stored_row": {
                "id": "report-1",
                "created_at": "2026-04-03T12:00:00Z",
                "needs_review": True,
                "model_versions": {
                    "pipeline": "phase12_langgraph",
                    "quality_gate": {
                        "degraded_placeholder": True,
                        "reason": "low_content_quality",
                    },
                    "agreement": {"final_confidence": 0.35},
                    "llm_primary": "gemma3:4b",
                    "llm_fallback": "phi3:mini",
                },
            }
        },
        final_status=SimpleNamespace(value="partial"),
        status=SimpleNamespace(value="partial"),
    )
    monkeypatch.setattr(
        phase7._graph_service,
        "run_generation",
        AsyncMock(return_value=fake_state),
    )

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
    assert body["stored"]["model_versions"]["pipeline"] == "phase12_langgraph"
    assert body["stored"]["model_versions"]["agreement"]["final_confidence"] == pytest.approx(0.35)
    assert body["stored"]["model_versions"]["llm_primary"] == "gemma3:4b"
    assert body["stored"]["model_versions"]["llm_fallback"] == "phi3:mini"


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


@pytest.mark.asyncio
async def test_phase7_latest_student_normalizes_phase15_student_shape(
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
                    "id": "phase15-student",
                    "file_id": "file-1",
                    "role": "student",
                    "created_at": "2026-04-03T12:10:00Z",
                    "needs_review": False,
                    "report_json": {
                        "report_type": "student",
                        "summary": "The submission is grounded and mostly coherent.",
                        "strengths": ["The argument is easy to follow."],
                        "weaknesses": ["Evaluation depth is still thin in the testing discussion."],
                        "suggestions": ["Expand the testing evaluation with clearer edge-case evidence."],
                        "improvement_plan": {
                            "actions": [
                                {
                                    "title": "Expand the testing evaluation",
                                    "rationale": "The report names testing, but it does not yet explain coverage depth clearly.",
                                    "steps": [
                                        "Add one paragraph on edge cases.",
                                        "Explain what the current tests do not cover.",
                                    ],
                                    "priority": "high",
                                }
                            ],
                            "timeline": "This week",
                        },
                        "learning_path": {
                            "recommended_practice": [
                                "Add one concise edge-case testing paragraph.",
                            ],
                            "milestones": [
                                {
                                    "title": "Re-check the evidence chain",
                                    "objective": "Ensure every major claim is backed by a specific example.",
                                    "activities": ["Link each claim to an explicit test or observation."],
                                }
                            ],
                        },
                        "confidence": {"score": 0.82, "band": "high"},
                        "safety": {"needs_review": False, "reason": ""},
                    },
                    "model_versions": {"pipeline": "phase15_phase16_genai"},
                    "rag_meta": {
                        "enabled": True,
                        "confidence_score": 0.79,
                        "confidence_label": "high",
                        "safe_review": False,
                        "citations": [
                            {
                                "title": "Testing Evidence",
                                "section": "evaluation",
                                "document_id": "doc-1",
                                "chunk_id": "chunk-1",
                            }
                        ],
                        "retrieved_chunks": [
                            {
                                "chunk_id": "chunk-1",
                                "document_id": "doc-1",
                                "document_title": "Testing Evidence",
                                "section": "evaluation",
                                "category": "testing",
                                "audience": "student",
                                "content": "Edge-case coverage should be discussed explicitly.",
                                "score": 0.88,
                            }
                        ],
                        "trace": {
                            "query": "student testing evaluation evidence",
                            "collection_name": "student_knowledge_base",
                        },
                    },
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
    report = body["item"]["report_json"]
    assert report["issues"][0]["title"] == "Evaluation depth is still thin in the testing discussion."
    assert report["improvement_plan"][0]["action"] == "Expand the testing evaluation"
    assert report["improvement_plan"][0]["priority"] == 1
    assert report["checklist"][0]["item"] == "Add one concise edge-case testing paragraph."
    assert report["model_agreement"]["final_confidence"] == pytest.approx(0.82)
    assert body["item"]["retrieval_confidence"] == pytest.approx(0.79)
    assert body["item"]["citations"][0]["title"] == "Testing Evidence"


@pytest.mark.asyncio
async def test_phase7_latest_professor_prefers_richer_report_over_newer_generic_row(
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
                    "id": "new-generic",
                    "file_id": "file-1",
                    "role": "professor",
                    "created_at": "2026-04-03T12:10:00Z",
                    "needs_review": False,
                    "report_json": {
                        "summary": "Detailed feedback explanation unavailable.",
                        "feedback_explanation": "Detailed feedback explanation unavailable.",
                        "rubric_breakdown": [
                            {
                                "criterion": "Overall academic quality",
                                "band": "Needs review",
                                "justification": "Detailed feedback explanation unavailable.",
                            }
                        ],
                        "moderation_notes": [],
                        "safety": {"needs_review": False, "reason": ""},
                    },
                    "model_versions": {
                        "agreement": {"final_confidence": 0.84},
                    },
                },
                {
                    "id": "older-rich",
                    "file_id": "file-1",
                    "role": "professor",
                    "created_at": "2026-04-03T12:00:00Z",
                    "needs_review": False,
                    "report_json": GOOD_PROFESSOR_REPORT,
                    "model_versions": {
                        "agreement": {"final_confidence": 0.8},
                    },
                },
            ]
        ),
    )

    response = await async_client.get(
        "/api/phase7/latest/professor/file-1",
        headers={"Authorization": f"Bearer {student_a_token}"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["found"] is True
    assert body["item"]["id"] == "older-rich"
    assert body["item"]["report_json"]["summary"] == GOOD_PROFESSOR_REPORT["summary"]
    assert body["selection_metadata"]["preferred_non_degraded"] is True
    assert body["selection_metadata"]["preferred_richer_report"] is True
    assert body["selection_metadata"]["total_reports_considered"] == 2
