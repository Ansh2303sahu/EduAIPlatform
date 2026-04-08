from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.api import ai_generate
from app.genai.schemas import (
    AIReportResponse,
    AuditTab,
    CompareResponse,
    ExplainResponse,
    ExplanationTab,
    FairnessTab,
    PredictionTab,
    ProfessorModerationReport,
    ReportConfidence,
    SafetySummary,
    SourcesTab,
    StudentReport,
)
from app.genai.explainability import confidence_band
from app.langgraph.graphs.professor_generative_graph import get_professor_generative_compiled_graph
from app.langgraph.graphs.student_generative_graph import get_student_generative_compiled_graph
from app.langgraph.schemas import Phase12ExecutionRequest
from app.langgraph.state import Phase12GraphState
from app.langgraph.nodes.validation import validation_node


def _student_response() -> AIReportResponse:
    report = StudentReport(
        summary="Student summary",
        strengths=["Clear structure"],
        weaknesses=["Evidence could be stronger"],
        suggestions=["Add more support for the main claim"],
        confidence_score=0.72,
        confidence=ReportConfidence(score=0.72, band="medium", rationale="Grounded but not perfect."),
        safety=SafetySummary(needs_review=False, reason=""),
    )
    return AIReportResponse(
        request_id="req-student",
        prediction=PredictionTab(report_type="student", report=report),
        explanation=ExplanationTab(confidence_band="medium"),
        sources=SourcesTab(),
        warnings=[],
        fairness=FairnessTab(),
        audit=AuditTab(
            request_id="req-student",
            execution_id="exec-student",
            role="student",
            graph_version="15.16.0",
            prompt_version="phase15_16.v1",
            output_version="phase15_16.output.v1",
            model_version="mistral:latest",
            validator_model_version="phi3:latest",
            timestamp=datetime.now(timezone.utc),
            final_status="completed",
        ),
    )


def _professor_response() -> AIReportResponse:
    report = ProfessorModerationReport(
        summary="Professor summary",
        feedback_explanation="Grounded moderation explanation.",
        strengths=["Reasoning is mostly coherent"],
        weaknesses=["Rubric alignment should be tighter"],
        suggestions=["Tie final judgement to stronger evidence"],
        confidence_score=0.78,
        confidence=ReportConfidence(score=0.78, band="high", rationale="Grounded moderation."),
        safety=SafetySummary(needs_review=False, reason=""),
    )
    return AIReportResponse(
        request_id="req-prof",
        prediction=PredictionTab(report_type="professor", report=report),
        explanation=ExplanationTab(confidence_band="high"),
        sources=SourcesTab(),
        warnings=[],
        fairness=FairnessTab(),
        audit=AuditTab(
            request_id="req-prof",
            execution_id="exec-prof",
            role="professor",
            graph_version="15.16.0",
            prompt_version="phase15_16.v1",
            output_version="phase15_16.output.v1",
            model_version="mistral:latest",
            validator_model_version="phi3:latest",
            timestamp=datetime.now(timezone.utc),
            final_status="completed",
        ),
    )


@pytest.mark.asyncio
async def test_student_generate_route_shape(async_client, monkeypatch):
    async def _fake_generate_student_report(body, user):
        return _student_response()

    monkeypatch.setattr(ai_generate._service, "generate_student_report", _fake_generate_student_report)
    async with async_client as ac:
        response = await ac.post(
            "/api/ai/generate/student-report",
            json={"file_id": "file-123"},
            headers={"Authorization": "Bearer token-student-a"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"]["report_type"] == "student"
    assert "explanation" in data
    assert "sources" in data
    assert "fairness" in data


@pytest.mark.asyncio
async def test_professor_generate_route_shape(async_client, monkeypatch):
    async def _fake_generate_professor_report(body, user):
        return _professor_response()

    monkeypatch.setattr(ai_generate._service, "generate_professor_report", _fake_generate_professor_report)
    async with async_client as ac:
        response = await ac.post(
            "/api/ai/generate/professor-report",
            json={"file_id": "file-456"},
            headers={"Authorization": "Bearer token-admin"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"]["report_type"] == "professor"
    assert data["prediction"]["report"]["feedback_explanation"] == "Grounded moderation explanation."


@pytest.mark.asyncio
async def test_explain_compare_audit_routes(async_client, monkeypatch):
    async def _fake_explain(*, file_id, role, user):
        return ExplainResponse(file_id=file_id, role=role, explanation=ExplanationTab(confidence_band="medium"))

    async def _fake_compare(*, file_id, role, user):
        return CompareResponse(
            file_id=file_id,
            role=role,
            fairness=FairnessTab(),
            comparison_summary="Agreement is high.",
            confidence_score=0.73,
            confidence_band="medium",
            evidence_reference_count=4,
            warning_count=1,
        )

    async def _fake_audit(*, file_id, role, user, include_pdf=False):
        return ai_generate.AuditResponse(
            file_id=file_id,
            role=role,
            audit=AuditTab(
                request_id="req",
                execution_id="exec",
                role=role,
                graph_version="15.16.0",
                prompt_version="phase15_16.v1",
                output_version="phase15_16.output.v1",
                model_version="mistral:latest",
                validator_model_version="phi3:latest",
                timestamp=datetime.now(timezone.utc),
                final_status="completed",
            ),
            warnings=[],
        )

    monkeypatch.setattr(ai_generate._service, "explain", _fake_explain)
    monkeypatch.setattr(ai_generate._service, "compare", _fake_compare)
    monkeypatch.setattr(ai_generate._service, "audit", _fake_audit)

    async with async_client as ac:
        explain = await ac.get("/api/ai/explain/file-123", headers={"Authorization": "Bearer token-student-a"})
        compare = await ac.get("/api/ai/compare/file-123", headers={"Authorization": "Bearer token-student-a"})
        audit = await ac.get("/api/ai/audit/file-123", headers={"Authorization": "Bearer token-student-a"})

    assert explain.status_code == 200
    assert compare.status_code == 200
    assert audit.status_code == 200
    assert compare.json()["comparison_summary"] == "Agreement is high."
    assert compare.json()["warning_count"] == 1


@pytest.mark.asyncio
async def test_pdf_route(async_client, monkeypatch):
    async def _fake_pdf(*, file_id, role, user):
        return b"%PDF-1.4 fake", f"eduaiplatform-{role}-report-{file_id}.pdf"

    monkeypatch.setattr(ai_generate._service, "pdf", _fake_pdf)

    async with async_client as ac:
        response = await ac.get("/api/ai/pdf/file-123", headers={"Authorization": "Bearer token-student-a"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert "attachment; filename=" in response.headers["content-disposition"]
    assert response.content.startswith(b"%PDF")


@pytest.mark.asyncio
async def test_validation_node_repairs_invalid_student_report():
    state = Phase12GraphState.create(
        Phase12ExecutionRequest(
            file_id="file-x",
            user_id="user-x",
            role="student",
            correlation_id="corr-x",
        )
    )
    state.draft_report = {"broken": True}
    state.evidence_quality_score = 0.5
    validated = await validation_node(state)
    report = StudentReport.model_validate(validated.pipeline_context.report)
    assert report.report_type == "student"
    assert validated.pipeline_context.validation_result.valid is True
    assert validated.pipeline_context.validation_result.repaired is True


def test_generative_graphs_compile():
    assert get_student_generative_compiled_graph() is not None
    assert get_professor_generative_compiled_graph() is not None


def test_confidence_band_thresholds_are_respected():
    assert confidence_band(0.80) == "high"
    assert confidence_band(0.60) == "medium"
    assert confidence_band(0.20) == "low"
