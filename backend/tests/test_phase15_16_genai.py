from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.api import ai_generate
from app.genai import service as genai_service
from app.genai.schemas import (
    AICheckGenAI,
    AICheckLLM,
    AICheckLangChain,
    AICheckLangGraph,
    AICheckMCP,
    AICheckML,
    AICheckN8N,
    AICheckRAG,
    AICheckResponse,
    AICheckSummary,
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
from app.langchain.models import ExecutionMetadata, RagContext
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
            prompt_version="phase15_16.v2",
            output_version="phase15_16.output.v1",
            model_version="gemma3:4b",
            validator_model_version="phi3:mini",
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
            prompt_version="phase15_16.v2",
            output_version="phase15_16.output.v1",
            model_version="gemma3:4b",
            validator_model_version="phi3:mini",
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
                prompt_version="phase15_16.v2",
                output_version="phase15_16.output.v1",
                model_version="gemma3:4b",
                validator_model_version="phi3:mini",
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
async def test_ai_check_route_shape(async_client, monkeypatch):
    async def _fake_check(*, file_id, role, user):
        return AICheckResponse(
            file_id=file_id,
            role=role,
            summary=AICheckSummary(
                selected_pipeline="phase15_phase16_genai",
                status="completed",
                confidence_score=0.81,
                confidence_band="high",
                report_summary="Stored report is available.",
            ),
            langchain=AICheckLangChain(
                available=True,
                pipeline="phase10_langchain",
                chain_name="phase10_student_generation",
                chain_version="1.0",
                prompt_version="v1",
                model_used="gemma3:4b",
                execution_mode="normal",
                retrieved_chunk_count=6,
                confidence_score=0.8,
                summary="Stored LangChain output is available for this file.",
            ),
            langgraph=AICheckLangGraph(
                available=True,
                pipeline="phase15_phase16_genai",
                graph_name="phase12_student_graph",
                graph_version="15.16.0",
                final_status="completed",
                total_steps=11,
                total_latency_ms=2400.0,
                trace_summary="Nodes visited: input_validation -> generation -> validation",
            ),
            genai=AICheckGenAI(
                available=True,
                pipeline="phase15_phase16_genai",
                model_version="gemma3:4b",
                validator_model_version="phi3:mini",
                final_status="completed",
                confidence_score=0.81,
                confidence_band="high",
                report_summary="Stored report is available.",
            ),
            rag=AICheckRAG(
                enabled=True,
                confidence_score=0.74,
                confidence_label="medium",
                citations_count=4,
                retrieved_chunk_count=6,
                summary="4 citations across 6 retrieved chunks.",
            ),
            ml=AICheckML(
                available=True,
                confidence_score=0.66,
                model_names=["student.feedback_classifier_multimodal.v1"],
                source="phase7",
                summary="ML calibration is available.",
            ),
            llm=AICheckLLM(
                available=True,
                model_used="gemma3:4b",
                primary_model="gemma3:4b",
                fallback_model="phi3:mini",
                route="gemma3:4b -> phi3:mini",
                source="phase15_phase16_genai",
            ),
            mcp=AICheckMCP(
                enabled=True,
                orchestration_enabled=True,
                llm_enabled=True,
                visible_tools=["student.summariser.v1"],
                summary="MCP is enabled.",
            ),
            n8n=AICheckN8N(
                configured=True,
                generation_bridge_active=True,
                summary="n8n bridge is configured.",
            ),
        )

    monkeypatch.setattr(ai_generate._service, "check", _fake_check)

    async with async_client as ac:
        response = await ac.get("/api/ai/check/file-123", headers={"Authorization": "Bearer token-student-a"})

    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["selected_pipeline"] == "phase15_phase16_genai"
    assert data["langchain"]["available"] is True
    assert data["langgraph"]["available"] is True
    assert data["genai"]["model_version"] == "gemma3:4b"
    assert data["rag"]["citations_count"] == 4


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


def test_execution_metadata_serializes_string_backed_enums():
    meta = ExecutionMetadata(
        request_id="req-1",
        execution_mode="normal",
        decision_source="hybrid",
    )

    model_versions = meta.to_model_versions_dict()

    assert model_versions["execution_mode"] == "normal"
    assert model_versions["decision_source"] == "hybrid"


@pytest.mark.asyncio
async def test_genai_persist_flattens_rag_fields_for_ai_reports(monkeypatch):
    captured: dict[str, object] = {}

    async def _fake_post_row(table: str, payload: dict[str, object]) -> dict[str, object]:
        captured["table"] = table
        captured["payload"] = payload
        return {
            "id": "row-1",
            "file_id": str(payload.get("file_id") or ""),
            "submission_id": payload.get("submission_id"),
            "role": str(payload.get("role") or ""),
            "created_at": "2026-04-15T10:00:00Z",
            "needs_review": bool(payload.get("needs_review")),
        }

    monkeypatch.setattr(genai_service.support, "post_row", _fake_post_row)

    state = Phase12GraphState.create(
        Phase12ExecutionRequest(
            file_id="file-rag",
            user_id="user-rag",
            role="student",
            correlation_id="corr-rag",
        )
    )
    state.pipeline_context.rag = RagContext(
        enabled=True,
        confidence_score=0.74,
        confidence_label="medium",
        safe_review=True,
        citations=[{"document_id": "doc-1", "chunk_id": "chunk-1"}],
        retrieved_chunks=[{"chunk_id": "chunk-1", "score": 0.91}],
        trace={"query": "student essay structure evidence", "collection_name": "student_knowledge_base"},
    )

    stored = await genai_service.GenAIService()._persist(state, _student_response())

    payload = captured["payload"]
    assert captured["table"] == "ai_reports"
    assert isinstance(payload, dict)
    assert "rag_meta" not in payload
    assert payload["rag_trace"] == {
        "query": "student essay structure evidence",
        "collection_name": "student_knowledge_base",
    }
    assert payload["citations"] == [{"document_id": "doc-1", "chunk_id": "chunk-1"}]
    assert payload["retrieved_chunks"] == [{"chunk_id": "chunk-1", "score": 0.91}]
    assert payload["retrieval_confidence"] == pytest.approx(0.74)
    assert payload["retrieval_confidence_label"] == "medium"
    assert payload["safe_review"] is True
    assert stored is not None
    assert stored.id == "row-1"


def test_ai_check_rag_reads_flattened_genai_storage_fields():
    rag = ai_generate._service._build_rag_check(
        genai_row={
            "citations": [
                {
                    "title": "Essay Structure",
                    "section": "writing",
                    "document_id": "doc-1",
                    "chunk_id": "chunk-1",
                }
            ],
            "retrieved_chunks": [
                {
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "document_title": "Essay Structure",
                    "section": "writing",
                    "category": "writing",
                    "audience": "student",
                    "content": "Clear structure helps the reader follow the argument.",
                    "score": 0.91,
                }
            ],
            "rag_trace": {
                "query": "student essay structure evidence",
                "collection_name": "student_knowledge_base",
            },
            "retrieval_confidence": 0.76,
            "retrieval_confidence_label": "high",
            "safe_review": True,
        },
        baseline_row=None,
    )

    assert rag.enabled is True
    assert rag.citations_count == 1
    assert rag.retrieved_chunk_count == 1
    assert rag.query == "student essay structure evidence"
    assert rag.collection_name == "student_knowledge_base"
    assert rag.confidence_score == pytest.approx(0.76)
    assert rag.confidence_label == "high"
    assert rag.safe_review is True


def test_ai_check_rag_falls_back_to_baseline_when_genai_stub_is_empty():
    rag = ai_generate._service._build_rag_check(
        genai_row={
            "rag_meta": {
                "enabled": False,
                "confidence_score": 0.0,
                "confidence_label": "low",
                "safe_review": False,
                "citations": [],
                "retrieved_chunks": [],
                "trace": {},
            },
            "rag_trace": {},
        },
        baseline_row={
            "citations": [
                {
                    "title": "Essay Structure",
                    "section": "writing",
                    "document_id": "doc-1",
                    "chunk_id": "chunk-1",
                }
            ],
            "retrieved_chunks": [
                {
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "document_title": "Essay Structure",
                    "section": "writing",
                    "category": "writing",
                    "audience": "student",
                    "content": "Clear structure helps the reader follow the argument.",
                    "score": 0.91,
                }
            ],
            "rag_trace": {
                "query": "student essay structure evidence",
                "collection_name": "student_knowledge_base",
            },
            "retrieval_confidence": 0.76,
            "retrieval_confidence_label": "high",
            "safe_review": False,
        },
    )

    assert rag.enabled is True
    assert rag.citations_count == 1
    assert rag.retrieved_chunk_count == 1
    assert rag.query == "student essay structure evidence"
    assert rag.confidence_score == pytest.approx(0.76)


def test_ai_check_uses_phase12_baseline_metadata_when_no_phase15_16_row():
    service = genai_service.GenAIService()
    rows = [
        {
            "id": "phase12-row",
            "file_id": "file-123",
            "submission_id": None,
            "role": "student",
            "created_at": "2026-04-15T10:00:00Z",
            "needs_review": False,
            "report_json": {
                "summary": "Stored baseline report is available.",
                "issues": [],
                "strengths": [],
                "confidence": {"overall": 0.79},
                "model_agreement": {"final_confidence": 0.79, "ml_confidence": 0.64},
                "safety": {"needs_review": False, "reason": ""},
            },
            "model_versions": {
                "pipeline": "phase12_langgraph",
                "graph_trace": {
                    "node_entries": [
                        {"node_name": "generation"},
                        {"node_name": "mcp_tools"},
                    ]
                },
                "langchain": {
                    "available": True,
                    "pipeline": "phase12_langgraph",
                    "chain_name": "phase10_student_generation",
                    "chain_version": "1.0.0",
                    "prompt_version": "phase10.student.v1",
                    "schema_version": "phase10.student.output.v1",
                    "provider": "ollama",
                    "model_used": "gemma3:4b",
                    "primary_model": "gemma3:4b",
                    "fallback_model": "phi3:mini",
                    "execution_mode": "normal",
                    "decision_source": "hybrid",
                    "retrieval_mode": "student",
                    "retrieved_chunk_count": 3,
                    "confidence_score": 0.79,
                    "summary": "LangChain baseline metadata is stored.",
                },
                "langgraph": {
                    "available": True,
                    "pipeline": "phase12_langgraph",
                    "graph_name": "phase12_student_graph",
                    "graph_version": "1.0.0",
                    "prompt_version": "phase10.student.v1",
                    "output_version": "phase12.student.output.v1",
                    "final_status": "completed",
                    "safe_mode": False,
                    "total_steps": 8,
                    "total_latency_ms": 1280.0,
                    "node_count": 8,
                    "decision_count": 4,
                    "failure_count": 0,
                    "trace_summary": "input_validation -> generation -> persistence",
                    "warnings": [],
                },
                "genai": {
                    "available": True,
                    "pipeline": "phase12_langgraph",
                    "model_version": "gemma3:4b",
                    "validator_model_version": "phi3:mini",
                    "final_status": "completed",
                    "confidence_score": 0.79,
                    "confidence_band": "high",
                    "report_summary": "Stored baseline report is available.",
                    "warning_count": 0,
                },
                "ml": {
                    "available": True,
                    "confidence_score": 0.64,
                    "model_names": ["student.feedback_classifier_multimodal.v1"],
                    "source": "phase12_langgraph",
                    "summary": "ML calibration is available.",
                },
                "llm": {
                    "available": True,
                    "model_used": "gemma3:4b",
                    "primary_model": "gemma3:4b",
                    "fallback_model": "phi3:mini",
                    "route": "gemma3:4b -> phi3:mini",
                    "source": "phase12_langgraph",
                },
                "mcp": {
                    "enabled": False,
                    "orchestration_enabled": False,
                    "llm_enabled": False,
                    "graph_used": True,
                    "tool_call_count": 0,
                    "visible_tools": [],
                    "summary": "MCP is not enabled in backend configuration.",
                },
            },
            "citations": [{"document_id": "doc-1", "chunk_id": "chunk-1"}],
            "retrieved_chunks": [{"chunk_id": "chunk-1", "score": 0.88}],
            "rag_trace": {"query": "student essay structure evidence", "collection_name": "student_knowledge_base"},
            "retrieval_confidence": 0.77,
            "retrieval_confidence_label": "high",
            "safe_review": False,
        }
    ]

    genai_row, graph_row, langchain_row, baseline_row = service._select_check_rows("student", rows)
    phase = service._phase_payload_or_none(genai_row)
    check = AICheckResponse(
        file_id="file-123",
        role="student",
        summary=service._build_check_summary("student", genai_row, baseline_row, phase),
        langchain=service._build_langchain_check("student", genai_row, langchain_row, baseline_row),
        langgraph=service._build_langgraph_check(graph_row, phase),
        genai=service._build_genai_check("student", genai_row, baseline_row, phase),
        rag=service._build_rag_check(genai_row, baseline_row),
        ml=service._build_ml_check(genai_row, baseline_row, phase),
        llm=service._build_llm_check(genai_row, baseline_row),
        mcp=service._build_mcp_check("student", graph_row),
        n8n=service._build_n8n_check("student", graph_row),
    )

    assert check.summary.selected_pipeline == "phase12_langgraph"
    assert check.summary.status == "completed"
    assert check.langgraph.available is True
    assert check.langgraph.graph_name == "phase12_student_graph"
    assert check.genai.available is True
    assert check.genai.pipeline == "phase12_langgraph"
    assert check.langchain.available is True
    assert check.mcp.graph_used is True
    assert check.rag.enabled is True


def test_ai_check_langgraph_falls_back_to_trace_count_when_stored_steps_are_zero():
    service = genai_service.GenAIService()

    row = {
        "model_versions": {
            "pipeline": "phase12_langgraph",
            "graph_trace": {
                "node_entries": [
                    {"node_name": "input_validation"},
                    {"node_name": "retrieval"},
                    {"node_name": "generation"},
                ]
            },
            "langgraph": {
                "available": True,
                "pipeline": "phase12_langgraph",
                "graph_name": "phase12_student_graph",
                "graph_version": "1.0.0",
                "prompt_version": "phase10.student.v1",
                "output_version": "phase12.student.output.v1",
                "final_status": "completed",
                "safe_mode": False,
                "total_steps": 0,
                "total_latency_ms": 720.0,
                "node_count": 0,
                "decision_count": 1,
                "failure_count": 0,
                "trace_summary": "input_validation -> retrieval -> generation",
                "warnings": [],
            },
        }
    }

    check = service._build_langgraph_check(row, phase=None)

    assert check.total_steps == 3
    assert check.node_count == 3
