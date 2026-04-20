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

    def test_rich_student_fields_are_preserved(self):
        raw = {
            "summary": "The report is mostly coherent but needs stronger evidence in the discussion.",
            "overall_judgment": "A credible draft with one major evidence gap.",
            "strengths": [
                {"title": "Clear framing", "detail": "The introduction sets up the task clearly."}
            ],
            "weaknesses": [
                {
                    "title": "Analysis is thin",
                    "detail": "The discussion section names examples but does not evaluate them.",
                    "severity": "medium",
                }
            ],
            "section_feedback": [
                {
                    "section_name": "Discussion",
                    "what_works": "Examples are relevant.",
                    "what_needs_improvement": "The analysis stays descriptive.",
                    "recommended_fix": "Add a short comparison of the strongest and weakest examples.",
                }
            ],
            "priority_issue": {
                "title": "Deepen the discussion section",
                "why_it_matters": "It currently limits the overall mark.",
                "how_to_fix_it": "Compare the evidence rather than listing it.",
            },
            "confidence_explanation": "Confidence is moderate because the evidence is clear in the introduction but thinner in the discussion.",
            "evidence_coverage": "The middle section has the weakest grounding.",
        }

        result = normalize_student_output(raw, safe_mode=False)

        assert result["overall_judgment"] == "A credible draft with one major evidence gap."
        assert result["strengths"][0]["detail"] == "The introduction sets up the task clearly."
        assert result["weaknesses"][0]["severity"] == "med"
        assert result["section_feedback"][0]["section_name"] == "Discussion"
        assert result["priority_issue"]["title"] == "Deepen the discussion section"
        assert "moderate" in result["confidence_explanation"].lower()
        assert "weakest grounding" in result["evidence_coverage"].lower()


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

    def test_professor_rich_fields_and_legacy_breakdown_can_coexist(self):
        result = normalize_professor_output(
            {
                "summary": "Evidence supports a mid-band judgement with one moderation caution.",
                "evaluator_overview": "Argument quality is adequate, but evidence coverage is uneven.",
                "rubric_breakdown": [
                    {
                        "criterion": "Argument",
                        "band": "Merit",
                        "justification": "The core position is clear and mostly supported.",
                    }
                ],
                "rubric_alignment": ["Argument quality", "Use of evidence"],
                "strengths": [{"title": "Clear thesis", "detail": "The position is explicit from the opening."}],
                "concerns": [
                    {
                        "title": "Evidence depth",
                        "detail": "Later sections assert points without enough comparison.",
                    }
                ],
                "section_observations": [
                    {
                        "section_name": "Conclusion",
                        "observation": "Returns to the thesis.",
                        "concern": "It does not weigh counterarguments.",
                        "next_step": "Check whether the body already addresses them.",
                    }
                ],
                "marking_considerations": ["Re-check evidence weighting in the final two sections."],
                "action_recommendations": ["Review whether the evidence depth matches the proposed band."],
                "confidence_explanation": "Confidence is moderate because the opening is strong but the later analysis is uneven.",
            }
        )

        assert result["summary"].startswith("Evidence supports")
        assert result["evaluator_overview"].startswith("Argument quality")
        assert result["rubric_alignment"] == ["Argument quality", "Use of evidence"]
        assert result["strengths"][0]["detail"] == "The position is explicit from the opening."
        assert result["concerns"][0]["detail"] == "Later sections assert points without enough comparison."
        assert result["section_observations"][0]["section_name"] == "Conclusion"
        assert result["marking_considerations"][0].startswith("Re-check evidence weighting")
        assert result["action_recommendations"][0].startswith("Review whether")
