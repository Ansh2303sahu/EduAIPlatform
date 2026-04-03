"""
Tests for parsers/normalizer.py.

Verifies that raw LLM output dicts are coerced correctly for both student
and professor reports, including edge cases (missing keys, wrong types,
rag_meta stripping, severity aliases, safe_mode overrides).
"""

from __future__ import annotations

import pytest

from app.langchain.parsers.normalizer import (
    normalize_professor_output,
    normalize_student_output,
)


# ---------------------------------------------------------------------------
# Student normalizer
# ---------------------------------------------------------------------------

class TestNormalizeStudentOutput:
    def test_minimal_input(self):
        result = normalize_student_output({"summary": "OK"}, safe_mode=False)
        assert result["summary"] == "OK"
        assert result["issues"] == []
        assert result["confidence"]["mode"] == "normal"

    def test_safe_mode_forces_restricted(self):
        result = normalize_student_output({"summary": "x"}, safe_mode=True)
        assert result["confidence"]["mode"] == "restricted"
        assert result["safety"]["needs_review"] is True

    def test_severity_aliases(self):
        raw = {
            "summary": "s",
            "issues": [
                {"title": "t1", "evidence": "e1", "severity": "critical"},
                {"title": "t2", "evidence": "e2", "severity": "medium"},
                {"title": "t3", "evidence": "e3", "severity": "minor"},
            ],
        }
        result = normalize_student_output(raw, safe_mode=False)
        severities = [i["severity"] for i in result["issues"]]
        assert severities == ["high", "med", "low"]

    def test_rag_meta_stripped(self):
        raw = {"summary": "s", "rag_meta": {"enabled": True}}
        result = normalize_student_output(raw, safe_mode=False)
        assert "rag_meta" not in result

    def test_truncation(self):
        long_summary = "x" * 2000
        result = normalize_student_output({"summary": long_summary}, safe_mode=False)
        assert len(result["summary"]) <= 1200

    def test_priority_coercion(self):
        raw = {
            "summary": "s",
            "improvement_plan": [{"action": "a", "why": "w", "how": "h", "priority": 15}],
        }
        result = normalize_student_output(raw, safe_mode=False)
        assert result["improvement_plan"][0]["priority"] == 10  # clamped to max

    def test_empty_input(self):
        result = normalize_student_output({}, safe_mode=False)
        assert "summary" in result
        assert result["issues"] == []

    def test_non_dict_input(self):
        result = normalize_student_output(None, safe_mode=False)
        assert isinstance(result, dict)

    def test_confidence_fallback_when_missing(self):
        result = normalize_student_output({"summary": "s"}, safe_mode=False)
        assert result["confidence"]["overall"] == pytest.approx(0.75, abs=0.01)

    def test_confidence_fallback_safe_mode(self):
        result = normalize_student_output({"summary": "s"}, safe_mode=True)
        assert result["confidence"]["overall"] == pytest.approx(0.35, abs=0.01)


# ---------------------------------------------------------------------------
# Professor normalizer
# ---------------------------------------------------------------------------

class TestNormalizeProfessorOutput:
    def test_minimal_input(self):
        result = normalize_professor_output({
            "feedback_explanation": "Good work.",
            "rubric_breakdown": [
                {"criterion": "Structure", "band": "Merit", "justification": "Well organised."}
            ],
        })
        assert result["feedback_explanation"] == "Good work."
        assert len(result["rubric_breakdown"]) == 1

    def test_empty_rubric_gets_fallback(self):
        result = normalize_professor_output({
            "feedback_explanation": "Needs improvement.",
            "rubric_breakdown": [],
        })
        assert len(result["rubric_breakdown"]) == 1
        assert result["rubric_breakdown"][0]["criterion"] == "Overall academic quality"

    def test_rag_meta_stripped(self):
        raw = {
            "feedback_explanation": "x",
            "rubric_breakdown": [],
            "rag_meta": {"enabled": True},
        }
        result = normalize_professor_output(raw)
        assert "rag_meta" not in result

    def test_truncation_justification(self):
        raw = {
            "feedback_explanation": "x",
            "rubric_breakdown": [{"criterion": "C", "band": "B", "justification": "j" * 2000}],
        }
        result = normalize_professor_output(raw)
        assert len(result["rubric_breakdown"][0]["justification"]) <= 1200

    def test_safety_defaults(self):
        result = normalize_professor_output({})
        assert result["safety"]["needs_review"] is False
        assert isinstance(result["safety"]["reason"], str)
