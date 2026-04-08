"""Deterministic report and critique builders for Phase 15/16."""

from __future__ import annotations

from typing import Any

from app.genai.explainability import confidence_band, rag_evidence_references
from app.genai.schemas import (
    CritiquePoint,
    CritiqueReport,
    ImprovementAction,
    ImprovementPlan,
    LearningMilestone,
    LearningPath,
    ProfessorModerationReport,
    ReportConfidence,
    SafetySummary,
    StudentReport,
)
from app.langgraph.state import Phase12GraphState


def _base_strengths(state: Phase12GraphState) -> list[str]:
    strengths: list[str] = []
    if len(state.extracted_text.strip()) > 800:
        strengths.append("The submission provides enough written material for a targeted review.")
    if state.ml_completed:
        strengths.append("Phase 6 ML calibration signals were available to support the explanation.")
    if state.rag_completed and state.retrieved_chunks:
        strengths.append("Knowledge-base guidance was retrieved to ground the feedback.")
    if not strengths:
        strengths.append("Some submission evidence was available for analysis.")
    return strengths[:4]


def _base_weaknesses(state: Phase12GraphState) -> list[str]:
    weaknesses: list[str] = []
    if state.evidence_quality_score < 0.35:
        weaknesses.append("The available evidence is limited, which reduces how specific the feedback can be.")
    if state.safe_mode:
        weaknesses.append("Safe mode was activated because the system detected weak or uncertain evidence.")
    if state.rag_completed and state.rag_result.weak_retrieval:
        weaknesses.append("Retrieved grounding evidence was weak, so confidence should remain cautious.")
    if state.ml_completed and state.ml_result.disagreement_markers:
        weaknesses.append("ML disagreement markers suggest mixed signals in the submission evidence.")
    if not weaknesses:
        weaknesses.append("No major structural weakness was detected beyond the points already listed.")
    return weaknesses[:5]


def _base_suggestions(state: Phase12GraphState) -> list[str]:
    suggestions: list[str] = []
    if state.role == "student":
        suggestions.extend(
            [
                "Make each major claim more specific and support it with clearer evidence.",
                "Revise weaker sections so the reasoning and examples connect more directly.",
                "Use the checklist and learning path to focus the next revision pass.",
            ]
        )
    else:
        suggestions.extend(
            [
                "Cross-check the judgement against the retrieved rubric or moderation guidance.",
                "Tighten any feedback claims that are not clearly grounded in the submission evidence.",
                "Use the consistency notes to resolve any mismatch between tone, score, and rationale.",
            ]
        )
    return suggestions[:5]


def build_counterfactual(state: Phase12GraphState) -> str:
    """Return a compact counterfactual improvement explanation."""

    if state.role == "student":
        return (
            "If the submission added clearer evidence for its main claims and tightened weaker sections, "
            "the confidence-adjusted evaluation would likely move upward."
        )
    return (
        "If the moderation rationale were tied more explicitly to rubric evidence and contradictions were reduced, "
        "the confidence-adjusted moderation assessment would likely strengthen."
    )


def build_deterministic_critique(
    state: Phase12GraphState,
    draft_report: dict[str, Any],
) -> CritiqueReport:
    """Build a conservative validator critique without another model call."""

    concerns: list[CritiquePoint] = []
    if state.safe_mode:
        concerns.append(
            CritiquePoint(
                issue="safe_mode_active",
                severity="high",
                recommendation="Keep the final wording conservative and retain a needs-review flag.",
            )
        )
    if state.rag_completed and state.rag_result.weak_retrieval:
        concerns.append(
            CritiquePoint(
                issue="weak_grounding",
                severity="medium",
                recommendation="Avoid over-claiming and keep evidence references compact and explicit.",
            )
        )
    if len(draft_report.get("weaknesses") or []) > len(draft_report.get("suggestions") or []):
        concerns.append(
            CritiquePoint(
                issue="missing_actionability",
                severity="medium",
                recommendation="Add more concrete suggestions so the feedback is actionable.",
            )
        )
    contradiction_flags: list[str] = []
    summary = str(draft_report.get("summary") or "").lower()
    weaknesses = " ".join(str(x) for x in draft_report.get("weaknesses") or []).lower()
    if "strong" in summary and "limited" in weaknesses:
        contradiction_flags.append("The summary sounds more positive than the weaknesses suggest.")
    tone_flags = ["Safe mode indicates a fairness/overclaim risk."] if state.safe_mode else []
    confidence = max(0.2, min(0.95, (state.ml_confidence + max(state.evidence_quality_score, 0.25)) / 2.0))
    return CritiqueReport(
        overall_quality="weak" if concerns else "acceptable",
        concerns=concerns,
        tone_bias_flags=tone_flags,
        contradiction_flags=contradiction_flags,
        confidence_score=confidence,
        refinement_required=bool(concerns or contradiction_flags or tone_flags),
        validator_notes=[
            "Validator critique built deterministically from evidence, ML confidence, and grounding strength."
        ],
    )


def build_student_report(
    state: Phase12GraphState,
    *,
    critique: CritiqueReport | None = None,
) -> StudentReport:
    """Build a schema-valid student report without relying on an LLM."""

    strengths = _base_strengths(state)
    weaknesses = _base_weaknesses(state)
    suggestions = _base_suggestions(state)
    refs = rag_evidence_references(state)
    score = max(0.2, min(0.95, (state.ml_confidence + max(state.evidence_quality_score, 0.25)) / 2.0))
    if critique is not None and critique.refinement_required:
        score = max(0.1, score - 0.08)
    band = confidence_band(score)
    improvement_plan = ImprovementPlan(
        strengths=strengths[:2],
        weaknesses=weaknesses[:3],
        suggestions=suggestions[:3],
        confidence_score=score,
        evidence_references=refs,
        actions=[
            ImprovementAction(
                title="Clarify the weakest section",
                rationale="The current evidence profile suggests some parts need sharper support.",
                steps=[
                    "Identify the least-supported paragraph or claim.",
                    "Add clearer justification or an example.",
                    "Check that the revised section connects back to the main argument.",
                ],
                priority="high",
            )
        ],
        timeline="Use this as the next revision pass.",
    )
    learning_path = LearningPath(
        strengths=strengths[:2],
        weaknesses=weaknesses[:3],
        suggestions=suggestions[:3],
        confidence_score=score,
        evidence_references=refs,
        milestones=[
            LearningMilestone(
                title="Strengthen evidence use",
                objective="Make claims more specific and better supported.",
                activities=[
                    "Annotate each major claim with its supporting evidence.",
                    "Revise places where support is implied rather than explicit.",
                ],
            )
        ],
        recommended_practice=[
            "Do a targeted revision pass focused only on evidence and clarity.",
            "Check whether each paragraph has a clear point and supporting material.",
        ],
    )
    return StudentReport(
        summary=(
            "This report combines submission evidence, ML calibration signals, and grounded guidance "
            "to highlight the most useful next improvements."
        ),
        strengths=strengths,
        weaknesses=weaknesses,
        suggestions=suggestions,
        confidence_score=score,
        evidence_references=refs,
        issues=[
            {
                "title": item[:80],
                "evidence": item,
                "severity": "med" if idx < 2 else "low",
            }
            for idx, item in enumerate(weaknesses[:4])
        ],
        improvement_plan=improvement_plan,
        learning_path=learning_path,
        confidence=ReportConfidence(
            score=score,
            band=band,
            rationale="Confidence was calibrated from ML confidence, evidence quality, and grounding strength.",
        ),
        reasoning_summary=[
            "Submission evidence was reviewed first.",
            "ML signals were used as calibration rather than as a sole decision-maker.",
            "Grounded knowledge-base guidance was used where available.",
        ],
        counterfactual_explanation=build_counterfactual(state),
        safety=SafetySummary(
            needs_review=state.safe_mode or state.rag_result.weak_retrieval,
            reason=(
                "The system stayed conservative because evidence or grounding was limited."
                if state.safe_mode or state.rag_result.weak_retrieval
                else ""
            ),
        ),
    )


def build_professor_report(
    state: Phase12GraphState,
    *,
    critique: CritiqueReport | None = None,
) -> ProfessorModerationReport:
    """Build a schema-valid professor report without relying on an LLM."""

    strengths = _base_strengths(state)
    weaknesses = _base_weaknesses(state)
    suggestions = _base_suggestions(state)
    refs = rag_evidence_references(state)
    score = max(0.2, min(0.95, (state.ml_confidence + max(state.evidence_quality_score, 0.25)) / 2.0))
    if critique is not None and critique.refinement_required:
        score = max(0.1, score - 0.08)
    band = confidence_band(score)
    return ProfessorModerationReport(
        summary="This moderation report synthesizes submission evidence, ML calibration, and rubric/policy grounding.",
        feedback_explanation=(
            "The moderation view is grounded in the submission evidence first, then calibrated using ML and "
            "retrieved rubric or policy guidance where available."
        ),
        strengths=strengths,
        weaknesses=weaknesses,
        suggestions=suggestions,
        confidence_score=score,
        evidence_references=refs,
        moderation_notes=[
            "Treat the output as a structured moderation aid rather than a replacement for human judgement.",
            "Use the evidence references to support any final rubric or consistency decision.",
        ],
        rubric_alignment=[
            "Check that the stated judgement matches the cited evidence.",
            "Review whether the tone and confidence align with the underlying evidence quality.",
        ],
        confidence=ReportConfidence(
            score=score,
            band=band,
            rationale="Confidence was calibrated from evidence quality, ML consistency, and grounding strength.",
        ),
        reasoning_summary=[
            "Submission evidence was reviewed first.",
            "ML moderation signals were used as calibration.",
            "Retrieved rubric/policy grounding informed the moderation explanation.",
        ],
        counterfactual_explanation=build_counterfactual(state),
        safety=SafetySummary(
            needs_review=state.safe_mode or state.rag_result.weak_retrieval,
            reason=(
                "Manual moderation is recommended because the evidence or grounding is limited."
                if state.safe_mode or state.rag_result.weak_retrieval
                else ""
            ),
        ),
    )
