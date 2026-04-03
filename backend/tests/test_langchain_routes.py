from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


SAMPLE_STUDENT_REPORT = {
    "summary": "Strong project direction with some testing gaps.",
    "issues": [{"title": "Testing depth", "evidence": "Limited direct test evidence.", "severity": "med"}],
    "strengths": [{"title": "Architecture clarity", "evidence": "Clear full-stack structure."}],
    "architecture_review": {
        "overview": "Clear architecture.",
        "backend": "FastAPI service layer.",
        "frontend": "React client.",
        "database": "Not assessed.",
        "security": "Not assessed.",
    },
    "implementation_review": {
        "features_built": ["Login", "Dashboard"],
        "technical_quality": "Good overall.",
        "integration_quality": "Adequate.",
    },
    "evaluation_review": {
        "testing_present": "Limited evidence.",
        "limitations": "Testing evidence is thin.",
        "academic_quality": "Good technical communication.",
    },
    "improvement_plan": [
        {"action": "Add test evidence", "why": "Stronger support.", "how": "Include results.", "priority": 1}
    ],
    "checklist": [{"item": "Document tests", "done": False}],
    "confidence": {"mode": "normal", "overall": 0.8},
    "model_agreement": {"ml_confidence": 0.75, "llm_confidence": 0.8, "final_confidence": 0.78},
    "safety": {"needs_review": False, "reason": ""},
}

SAMPLE_PROFESSOR_REPORT = {
    "rubric_breakdown": [
        {"criterion": "Overall academic quality", "band": "Merit", "justification": "Well-supported work."}
    ],
    "feedback_explanation": "The submission is coherent and reasonably well supported.",
    "moderation_notes": [{"risk": "Borderline banding", "note": "Check evidence depth."}],
    "safety": {"needs_review": True, "reason": "Manual moderation recommended."},
}


def test_langchain_routers_use_shared_support_module():
    import app.langchain.routers.professor as professor_router
    import app.langchain.routers.student as student_router

    assert student_router._build_ingestion_bundle.__module__ == "app.services.report_generation_support"
    assert student_router._call_ai_student_multimodal.__module__ == "app.services.report_generation_support"
    assert professor_router._build_ingestion_bundle.__module__ == "app.services.report_generation_support"
    assert professor_router._call_ai_professor_multimodal.__module__ == "app.services.report_generation_support"


@pytest.mark.asyncio
async def test_langchain_student_generate_public_response(monkeypatch):
    import app.langchain.routers.student as student_router

    monkeypatch.setattr(student_router, "_rate_limit", lambda _user_id: None)
    monkeypatch.setattr(
        student_router,
        "_load_file",
        AsyncMock(return_value={"id": "file-1", "submission_id": "sub-1", "mime_type": "application/pdf", "created_at": "2026-04-01"}),
    )
    monkeypatch.setattr(
        student_router,
        "_build_ingestion_bundle",
        AsyncMock(
            return_value={
                "text_content": "A project report about a React and FastAPI platform.",
                "ocr_text": "",
                "audio_transcript": "",
                "tables_json": None,
            }
        ),
    )
    monkeypatch.setattr(student_router, "_detect_submission_kind", lambda _ingestion: "project")
    monkeypatch.setattr(student_router, "_get_rows", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        student_router,
        "_call_ai_student_multimodal",
        AsyncMock(return_value={"feedback_category": "project_review", "quality_band": "high", "confidence_0_to_4": 3}),
    )
    monkeypatch.setattr(
        student_router,
        "_post_row",
        AsyncMock(return_value={"id": "rep-1", "file_id": "file-1", "submission_id": "sub-1", "role": "student", "needs_review": False, "created_at": "2026-04-01T12:00:00Z"}),
    )
    monkeypatch.setattr(
        student_router._pipeline,
        "run",
        AsyncMock(
            return_value={
                "report": SAMPLE_STUDENT_REPORT,
                "rag_meta": {
                    "enabled": True,
                    "confidence_score": 0.82,
                    "confidence_label": "high",
                    "safe_review": False,
                    "weak_retrieval": False,
                    "chunk_count": 2,
                    "citations": [{"title": "Guide", "section": "Architecture"}],
                    "retrieved_chunks": [{"chunk_id": "secret"}],
                    "trace": {"internal": True},
                },
                "ml": {"feedback_category": "project_review", "quality_band": "high", "confidence_0_to_4": 3},
                "model_used": "mistral",
                "fallback_used": False,
                "decision_source": "hybrid",
                "execution_mode": "normal",
                "storage_fields": {},
                "storage_payload": {"model_versions": {"llm_model_used": "mistral"}},
                "execution_metadata": {"internal": "do-not-expose"},
                "validation_status": {"valid": True},
            }
        ),
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/api/langchain/student/generate",
            headers={"Authorization": "Bearer token-student-a"},
            json={"file_id": "file-1", "force": True},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["meta"]["role"] == "student"
    assert payload["meta"]["decision_source"] == "hybrid"
    assert "execution_metadata" not in payload
    assert "storage_payload" not in payload
    assert "retrieved_chunks" not in payload["rag_meta"]
    assert "trace" not in payload["rag_meta"]


@pytest.mark.asyncio
async def test_phase10_student_alias_returns_cached_public_response(monkeypatch):
    import app.langchain.routers.student as student_router

    monkeypatch.setattr(student_router, "_rate_limit", lambda _user_id: None)
    monkeypatch.setattr(
        student_router,
        "_load_file",
        AsyncMock(return_value={"id": "file-1", "submission_id": "sub-1"}),
    )
    monkeypatch.setattr(
        student_router,
        "_build_ingestion_bundle",
        AsyncMock(return_value={"text_content": "cached text", "ocr_text": "", "audio_transcript": "", "tables_json": None}),
    )
    monkeypatch.setattr(student_router, "_detect_submission_kind", lambda _ingestion: "academic")
    monkeypatch.setattr(
        student_router,
        "_get_rows",
        AsyncMock(
            return_value=[
                {
                    "id": "rep-cached",
                    "file_id": "file-1",
                    "submission_id": "sub-1",
                    "role": "student",
                    "needs_review": False,
                    "created_at": "2026-04-01T12:00:00Z",
                    "report_json": SAMPLE_STUDENT_REPORT,
                    "citations": [{"title": "Guide", "section": "Intro"}],
                    "retrieved_chunks": [{"chunk_id": "c1"}],
                    "retrieval_confidence": 0.7,
                    "retrieval_confidence_label": "high",
                    "safe_review": False,
                    "model_versions": {"llm_model_used": "mistral"},
                }
            ]
        ),
    )
    pipeline_run = AsyncMock()
    monkeypatch.setattr(student_router._pipeline, "run", pipeline_run)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/api/phase10/student/generate",
            headers={"Authorization": "Bearer token-student-a"},
            json={"file_id": "file-1", "force": False},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["cached"] is True
    assert payload["stored"]["id"] == "rep-cached"
    assert payload["report"]["summary"] == SAMPLE_STUDENT_REPORT["summary"]
    pipeline_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_langchain_professor_route_enforces_role_separation():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/api/langchain/professor/generate",
            headers={"Authorization": "Bearer token-student-a"},
            json={"file_id": "file-p1", "force": True},
        )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_langchain_professor_generate_public_response(monkeypatch):
    import app.langchain.routers.professor as professor_router

    monkeypatch.setattr(professor_router, "_rate_limit", lambda _user_id: None)
    monkeypatch.setattr(
        professor_router,
        "_load_file",
        AsyncMock(return_value={"id": "file-p1", "submission_id": "sub-p1", "mime_type": "application/pdf", "created_at": "2026-04-01"}),
    )
    monkeypatch.setattr(
        professor_router,
        "_build_ingestion_bundle",
        AsyncMock(
            return_value={
                "text_content": "An academic essay about cloud governance.",
                "ocr_text": "",
                "audio_transcript": "",
                "tables_json": None,
            }
        ),
    )
    monkeypatch.setattr(professor_router, "_detect_submission_kind", lambda _ingestion: "academic")
    monkeypatch.setattr(professor_router, "_get_rows", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        professor_router,
        "_call_ai_professor_multimodal",
        AsyncMock(return_value={"rubric_band": "merit", "argument_depth": "high", "moderation_consistency": "high"}),
    )
    monkeypatch.setattr(
        professor_router,
        "_post_row",
        AsyncMock(return_value={"id": "rep-p1", "file_id": "file-p1", "submission_id": "sub-p1", "role": "professor", "needs_review": True, "created_at": "2026-04-01T12:00:00Z"}),
    )
    monkeypatch.setattr(
        professor_router._pipeline,
        "run",
        AsyncMock(
            return_value={
                "report": SAMPLE_PROFESSOR_REPORT,
                "rag_meta": {
                    "enabled": True,
                    "confidence_score": 0.51,
                    "confidence_label": "medium",
                    "safe_review": False,
                    "weak_retrieval": False,
                    "chunk_count": 1,
                    "citations": [{"title": "Policy", "section": "Bands"}],
                    "retrieved_chunks": [{"chunk_id": "hidden"}],
                    "trace": {"hidden": True},
                },
                "ml": {"rubric_band": "merit", "argument_depth": "high", "moderation_consistency": "high"},
                "model_used": "mistral",
                "fallback_used": False,
                "decision_source": "hybrid",
                "execution_mode": "normal",
                "discrepancy_flag": True,
                "storage_fields": {},
                "storage_payload": {"model_versions": {"llm_model_used": "mistral"}},
                "execution_metadata": {"internal": "do-not-expose"},
                "validation_status": {"valid": True},
            }
        ),
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/api/langchain/professor/generate",
            headers={"Authorization": "Bearer token-admin"},
            json={"file_id": "file-p1", "force": True},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["meta"]["role"] == "professor"
    assert payload["meta"]["discrepancy_flag"] is True
    assert "execution_metadata" not in payload
    assert "storage_payload" not in payload
    assert "retrieved_chunks" not in payload["rag_meta"]
    assert "trace" not in payload["rag_meta"]
