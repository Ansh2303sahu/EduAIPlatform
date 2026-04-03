"""
Tests for Phase 10 Pydantic schemas.

Verifies that request/response schemas accept valid data and reject invalid
data, and that they are structurally compatible with Phase 7 report shapes.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.langchain.schemas import (
    LangChainProfessorResponse,
    LangChainStudentResponse,
    Phase10GenerateIn,
    Phase10ProfessorReport,
    Phase10StudentReport,
)


# ---------------------------------------------------------------------------
# Phase10GenerateIn
# ---------------------------------------------------------------------------

def test_generate_in_valid():
    req = Phase10GenerateIn(file_id="abc-123")
    assert req.file_id == "abc-123"
    assert req.force is False


def test_generate_in_force():
    req = Phase10GenerateIn(file_id="abc-123", force=True)
    assert req.force is True


def test_generate_in_empty_file_id():
    with pytest.raises(ValidationError):
        Phase10GenerateIn(file_id="")


# ---------------------------------------------------------------------------
# Phase10StudentReport
# ---------------------------------------------------------------------------

def test_student_report_minimal():
    report = Phase10StudentReport(summary="Good work.")
    assert report.summary == "Good work."
    assert report.issues == []
    assert report.confidence.mode == "normal"


def test_student_report_full():
    data = {
        "summary": "Well structured project.",
        "issues": [{"title": "Missing tests", "evidence": "No test file found.", "severity": "high"}],
        "strengths": [{"title": "Clean code", "evidence": "Good naming conventions."}],
        "architecture_review": {
            "overview": "Solid.",
            "backend": "FastAPI.",
            "frontend": "React.",
            "database": "PostgreSQL.",
            "security": "Basic auth.",
        },
        "implementation_review": {
            "features_built": ["Login", "Dashboard"],
            "technical_quality": "Good.",
            "integration_quality": "Adequate.",
        },
        "evaluation_review": {
            "testing_present": "No.",
            "limitations": "No tests.",
            "academic_quality": "Good.",
        },
        "improvement_plan": [{"action": "Add tests", "why": "Coverage.", "how": "pytest", "priority": 1}],
        "checklist": [{"item": "Tests added", "done": False}],
        "confidence": {"mode": "normal", "overall": 0.8},
        "model_agreement": {"ml_confidence": 0.7, "llm_confidence": 0.8, "final_confidence": 0.75},
        "safety": {"needs_review": False, "reason": ""},
    }
    report = Phase10StudentReport.model_validate(data)
    assert report.issues[0].severity == "high"
    assert report.confidence.overall == 0.8


def test_student_report_invalid_severity():
    with pytest.raises(ValidationError):
        Phase10StudentReport.model_validate({
            "summary": "x",
            "issues": [{"title": "t", "evidence": "e", "severity": "critical"}],
        })


# ---------------------------------------------------------------------------
# Phase10ProfessorReport
# ---------------------------------------------------------------------------

def test_professor_report_minimal():
    report = Phase10ProfessorReport(
        rubric_breakdown=[],
        feedback_explanation="",
        moderation_notes=[],
    )
    assert report.safety.needs_review is False


def test_professor_report_full():
    data = {
        "rubric_breakdown": [
            {"criterion": "Structure", "band": "Merit", "justification": "Well organised."}
        ],
        "feedback_explanation": "The submission meets the Merit band across all criteria.",
        "moderation_notes": [{"risk": "Plagiarism risk", "note": "Check references."}],
        "safety": {"needs_review": True, "reason": "Suspected plagiarism."},
    }
    report = Phase10ProfessorReport.model_validate(data)
    assert report.rubric_breakdown[0].band == "Merit"
    assert report.safety.needs_review is True


def test_langchain_student_response_public_shape():
    response = LangChainStudentResponse.model_validate(
        {
            "cached": False,
            "request_id": "req-1",
            "ml": {"feedback_category": "project_review"},
            "report": {"summary": "Good work."},
            "rag_meta": {
                "enabled": True,
                "confidence_score": 0.8,
                "confidence_label": "high",
                "safe_review": False,
                "chunk_count": 2,
                "weak_retrieval": False,
                "citations": [{"title": "Guide"}],
            },
            "meta": {
                "role": "student",
                "needs_review": False,
                "model_used": "mistral",
                "fallback_used": False,
                "decision_source": "hybrid",
                "execution_mode": "normal",
            },
            "stored": {
                "id": "row-1",
                "file_id": "file-1",
                "submission_id": "sub-1",
                "role": "student",
                "needs_review": False,
            },
        }
    )

    assert response.meta.role == "student"
    assert response.rag_meta.chunk_count == 2


def test_langchain_professor_response_allows_discrepancy_flag():
    response = LangChainProfessorResponse.model_validate(
        {
            "cached": True,
            "request_id": "req-2",
            "ml": {},
            "report": {
                "rubric_breakdown": [
                    {"criterion": "Overall", "band": "Merit", "justification": "Supported."}
                ],
                "feedback_explanation": "Supported judgment.",
                "moderation_notes": [],
            },
            "rag_meta": {},
            "meta": {
                "role": "professor",
                "needs_review": True,
                "model_used": "mistral",
                "fallback_used": False,
                "decision_source": "hybrid",
                "execution_mode": "normal",
                "discrepancy_flag": True,
            },
            "stored": None,
        }
    )

    assert response.meta.role == "professor"
    assert response.meta.discrepancy_flag is True
