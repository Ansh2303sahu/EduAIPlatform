from __future__ import annotations

from app.services.report_richness import (
    build_report_preview,
    extract_best_summary,
    report_low_content_quality,
    report_richness_score,
)


def test_extract_best_summary_prefers_richer_student_fields() -> None:
    report = {
        "summary": "Automated review generated with limited confidence.",
        "overall_judgment": "A solid submission with one major evidence gap in the discussion section.",
        "priority_issue": {"title": "Deepen the discussion analysis"},
    }

    assert (
        extract_best_summary("student", report)
        == "A solid submission with one major evidence gap in the discussion section."
    )


def test_report_preview_counts_rich_fields() -> None:
    report = {
        "summary": "The essay is coherent and mostly well supported.",
        "strengths": [{"title": "Clear thesis"}, {"title": "Relevant examples"}],
        "weaknesses": [{"title": "Thin comparison", "severity": "med"}],
        "priority_issue": {
            "title": "Strengthen comparative analysis",
            "why_it_matters": "It affects the mark-bearing analysis criterion.",
            "how_to_fix_it": "Compare the strongest and weakest source directly.",
        },
        "checklist": [
            {"item": "Add one comparative paragraph"},
            {"item": "Tighten the conclusion"},
        ],
    }

    preview = build_report_preview("student", report)

    assert preview["summary"].startswith("The essay is coherent")
    assert preview["priority_issue"]["title"] == "Strengthen comparative analysis"
    assert preview["strengths_count"] == 2
    assert preview["weaknesses_count"] == 1
    assert preview["checklist_preview"] == [
        "Add one comparative paragraph",
        "Tighten the conclusion",
    ]


def test_low_content_quality_detects_generic_placeholder_report() -> None:
    report = {
        "summary": "Automated review generated with limited confidence.",
        "feedback_explanation": "This report combines submission evidence and mixed signals.",
        "strengths": [],
        "issues": [],
        "improvement_plan": [],
        "checklist": [],
    }

    assert report_low_content_quality("student", report) is True
    assert report_richness_score("student", report) < 6


def test_low_content_quality_accepts_rich_professor_report() -> None:
    report = {
        "summary": "The submission meets the mid-band standard, though the later analysis weakens the consistency of the mark.",
        "evaluator_overview": "Argument quality is clear at the start, but the final third is less analytical and needs closer moderation.",
        "rubric_alignment": ["Argument", "Evidence use", "Structure"],
        "strengths": [
            {"title": "Clear position", "detail": "The opening establishes the claim and scope precisely."}
        ],
        "concerns": [
            {
                "title": "Evidence thins in the discussion",
                "detail": "Later claims are asserted rather than compared.",
                "severity": "med",
            }
        ],
        "section_observations": [
            {
                "section_name": "Discussion",
                "observation": "Examples are relevant.",
                "concern": "They are not weighed against one another.",
                "next_step": "Check whether the mark should be capped for analysis depth.",
            }
        ],
        "marking_considerations": ["Review whether the analytical depth matches the proposed band."],
        "action_recommendations": ["Moderate the discussion section against the rubric wording."],
        "confidence_explanation": "Confidence is moderate because the opening is strong but the evidence coverage drops later.",
    }

    assert report_low_content_quality("professor", report) is False
    assert report_richness_score("professor", report) >= 8
