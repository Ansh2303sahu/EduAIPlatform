"""Deterministic report and critique builders for Phase 15/16."""

from __future__ import annotations

import re
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

_SPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CITATION_RE = re.compile(r"\([^)]+(?:19|20)\d{2}[a-z]?\)|\[[0-9]{1,3}\]|\bet al\.\b", re.IGNORECASE)
_HEADING_HINTS = (
    "chapter",
    "introduction",
    "background",
    "literature review",
    "methodology",
    "research methodology",
    "research design",
    "discussion",
    "conclusion",
)
_METHOD_HINTS = (
    "methodology",
    "research design",
    "design science research",
    "dsr",
    "evaluation framework",
    "requirements modelling",
    "evaluation",
)
_ARGUMENT_HINTS = (
    "argues",
    "documents",
    "demonstrates",
    "shows",
    "explores",
    "examines",
    "adopted",
    "selected",
    "justified",
)


def _clean_text(value: str | None, *, limit: int | None = None) -> str:
    text = _SPACE_RE.sub(" ", str(value or "").strip())
    if limit is None or len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _submission_text(state: Phase12GraphState) -> str:
    return _clean_text(state.extracted_text)


def _submission_lines(state: Phase12GraphState) -> list[str]:
    raw = [
        _clean_text(line, limit=180)
        for line in str(state.extracted_text or "").splitlines()
    ]
    return [line for line in raw if line]


def _submission_sentences(state: Phase12GraphState) -> list[str]:
    compact = _submission_text(state)
    if not compact:
        return []
    sentences = [
        _clean_text(sentence, limit=240)
        for sentence in _SENTENCE_SPLIT_RE.split(compact)
    ]
    return [sentence for sentence in sentences if len(sentence) >= 40]


def _mode_hint(state: Phase12GraphState) -> str:
    return str((state.rag_result.trace or {}).get("mode") or "").strip().lower()


def _primary_heading(state: Phase12GraphState) -> str:
    for line in _submission_lines(state)[:8]:
        lowered = line.lower()
        if any(hint in lowered for hint in _HEADING_HINTS):
            return line
        if len(line) <= 90 and line == line.title():
            return line
    compact = _submission_text(state)
    match = re.search(
        r"(chapter\s+\d+[^.]{0,80}|research methodology|research design|introduction|background and context|literature review)",
        compact,
        re.IGNORECASE,
    )
    return _clean_text(match.group(1), limit=100) if match else ""


def _focus_sentence(state: Phase12GraphState) -> str:
    sentences = _submission_sentences(state)
    if not sentences:
        return ""
    ranked = sorted(
        sentences[:8],
        key=lambda sentence: (
            sum(hint in sentence.lower() for hint in _METHOD_HINTS + _ARGUMENT_HINTS),
            bool(_CITATION_RE.search(sentence)),
            len(sentence),
        ),
        reverse=True,
    )
    return ranked[0]


def _citation_sentence(state: Phase12GraphState) -> str:
    for sentence in _submission_sentences(state)[:10]:
        if _CITATION_RE.search(sentence):
            return sentence
    return ""


def _quoted_snippet(text: str, *, limit: int = 140) -> str:
    cleaned = _clean_text(text, limit=limit)
    return f'"{cleaned}"' if cleaned else ""


def _student_priority_title(state: Phase12GraphState) -> str:
    text = _submission_text(state).lower()
    if any(term in text for term in _METHOD_HINTS):
        return "Explain how the chosen methodology was applied"
    if _mode_hint(state) == "chapter":
        return "Tighten the key claim in this chapter"
    return "Strengthen the least-supported claim"


def _submission_grounded_strengths(state: Phase12GraphState) -> list[str]:
    strengths: list[str] = []
    heading = _primary_heading(state)
    focus = _focus_sentence(state)
    cited = _citation_sentence(state)

    if heading:
        strengths.append(f"The submission clearly signals its focus in { _quoted_snippet(heading, limit=90) }.")
    if focus:
        strengths.append(f"The opening explanation gives the reviewer concrete material to assess, for example { _quoted_snippet(focus) }.")
    if cited and cited != focus:
        strengths.append(f"The chapter already anchors part of the discussion in cited literature, for example { _quoted_snippet(cited) }.")
    return [item for item in strengths if item][:3]


def _submission_grounded_weaknesses(state: Phase12GraphState) -> list[str]:
    weaknesses: list[str] = []
    focus = _focus_sentence(state)
    cited = _citation_sentence(state)
    text = _submission_text(state).lower()

    if focus:
        weaknesses.append(
            f"The claim in { _quoted_snippet(focus) } is still broader than the supporting explanation, so the reader cannot yet see the full claim-to-evidence chain."
        )
    if any(term in text for term in _METHOD_HINTS):
        weaknesses.append(
            "The methodology discussion names important choices, but it still needs clearer explanation of how those choices were applied in the project and how they shaped evaluation."
        )
    if cited:
        weaknesses.append(
            f"The cited support in { _quoted_snippet(cited) } would be stronger if each source were tied more explicitly to a concrete decision or criterion in the submission."
        )
    elif _mode_hint(state) in {"essay", "report", "chapter"}:
        weaknesses.append(
            "Several academic claims still need tighter signposting, so the argument flow is not yet as explicit as it could be."
        )

    if state.safe_mode:
        weaknesses.append("Safe mode was activated, so this draft should be treated as conservative rather than final.")
    if state.rag_completed and state.rag_result.weak_retrieval:
        weaknesses.append("Knowledge-base grounding was too weak to support confident external guidance for this submission.")
    return [item for item in weaknesses if item][:5]


def _submission_grounded_suggestions(state: Phase12GraphState) -> list[str]:
    suggestions: list[str] = []
    focus = _focus_sentence(state)
    cited = _citation_sentence(state)
    text = _submission_text(state).lower()

    if focus:
        suggestions.append(
            f"Revise the paragraph around { _quoted_snippet(focus, limit=110) } so it states the main claim, the reason for it, and the concrete evidence or method in one place."
        )
    if any(term in text for term in _METHOD_HINTS):
        suggestions.append(
            "After naming the chosen methodology, explain how it shaped the actual development, requirements modelling, and evaluation steps used in this work."
        )
    if cited:
        suggestions.append(
            "After each cited framework or source, add one sentence explaining exactly what decision, criterion, or design choice it justifies in your own submission."
        )
    if _mode_hint(state) in {"essay", "report", "chapter"}:
        suggestions.append(
            "Add clearer signposting at the start and end of key sections so the reader can follow the argument from claim to evidence to conclusion."
        )
    return [item for item in suggestions if item][:4]


def _student_summary(state: Phase12GraphState) -> str:
    focus = _focus_sentence(state)
    heading = _primary_heading(state)
    lead = _quoted_snippet(focus or heading, limit=130)
    if lead:
        return (
            f"The submission gives the reviewer a clear starting point through {lead}, but some important claims still need tighter explanation and follow-through. "
            "The next revision should make the link between each major claim, the supporting source or method, and the concrete project or chapter evidence much more explicit."
        )
    return (
        "The submission contains enough material for a conservative review, but some important claims still need tighter explanation and follow-through. "
        "The next revision should make the link between each major claim and its supporting evidence much more explicit."
    )


def _student_overall_judgment(state: Phase12GraphState) -> str:
    if _mode_hint(state) in {"essay", "report", "chapter"}:
        return "A clear academic direction is visible, but several important claims still need more explicit justification and follow-through."
    return "There is a clear direction in the work, but several important claims still need more explicit justification and follow-through."


def _student_priority_issue(state: Phase12GraphState, weaknesses: list[str], suggestions: list[str]) -> dict[str, str]:
    focus = _focus_sentence(state)
    return {
        "title": _student_priority_title(state),
        "why_it_matters": weaknesses[0] if weaknesses else (
            f"The passage around { _quoted_snippet(focus, limit=110) } still needs stronger justification."
            if focus
            else "At least one important claim still needs stronger justification."
        ),
        "how_to_fix_it": suggestions[0] if suggestions else (
            f"Revise the paragraph around { _quoted_snippet(focus, limit=110) } so the claim, justification, and evidence appear together."
            if focus
            else "Revise the weakest section so the claim, justification, and evidence appear together."
        ),
    }


def _base_strengths(state: Phase12GraphState) -> list[str]:
    strengths: list[str] = []
    strengths.extend(_submission_grounded_strengths(state))
    if len(state.extracted_text.strip()) > 800 and len(strengths) < 2:
        strengths.append("The submission provides enough written material for a targeted review.")
    if state.rag_completed and state.retrieved_chunks and len(strengths) < 3:
        strengths.append("Knowledge-base guidance was retrieved to ground the feedback.")
    if state.ml_completed and len(strengths) < 3:
        strengths.append("Phase 6 ML calibration signals were available as a secondary confidence check.")
    if not strengths:
        strengths.append("Some submission evidence was available for analysis.")
    return strengths[:4]


def _base_weaknesses(state: Phase12GraphState) -> list[str]:
    weaknesses: list[str] = []
    weaknesses.extend(_submission_grounded_weaknesses(state))
    if state.evidence_quality_score < 0.35 and not weaknesses:
        weaknesses.append("The available evidence is limited, which reduces how specific the feedback can be.")
    if state.ml_completed and state.ml_result and state.ml_result.disagreement_markers:
        weaknesses.append("ML disagreement markers suggest mixed signals in the submission evidence.")
    if not weaknesses:
        weaknesses.append("No major structural weakness was detected beyond the points already listed.")
    return weaknesses[:5]


def _base_suggestions(state: Phase12GraphState) -> list[str]:
    suggestions: list[str] = []
    suggestions.extend(_submission_grounded_suggestions(state))
    if state.role == "student":
        if len(suggestions) < 3:
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
                rationale="One of the weaker sections is not yet supported with enough clear explanation or evidence.",
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
        summary=_student_summary(state),
        overall_judgment=_student_overall_judgment(state),
        strengths=strengths,
        weaknesses=weaknesses,
        suggestions=suggestions,
        confidence_score=score,
        evidence_references=refs,
        strength_cards=[
            {"title": item[:80], "detail": item}
            for item in strengths[:3]
        ],
        weakness_cards=[
            {"title": item[:80], "detail": item, "severity": "medium" if idx < 2 else "low"}
            for idx, item in enumerate(weaknesses[:3])
        ],
        section_feedback=[
            {
                "section_name": "Main discussion",
                "what_works": strengths[0] if strengths else "",
                "what_needs_improvement": weaknesses[0] if weaknesses else "",
                "recommended_fix": suggestions[0] if suggestions else "",
            }
        ],
        priority_issue=_student_priority_issue(state, weaknesses, suggestions),
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
        confidence_explanation="Confidence depends on how much direct evidence was available from the submission, plus the strength of the grounded context.",
        evidence_coverage="The report is strongest where the submission contained concrete explanation, examples, or evidence, and weaker where the text stayed brief or implicit.",
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
        summary="The moderation picture is usable, but the final judgement should still be checked against the strongest and weakest evidenced sections.",
        evaluator_overview="Use this report to support moderation, not to replace close review of the submission evidence.",
        feedback_explanation=(
            "The moderation view is grounded in the submission evidence first and then calibrated using rubric or policy guidance where available. "
            "The main review task is to make sure the final judgement is no broader than the evidence supports."
        ),
        strengths=strengths,
        weaknesses=weaknesses,
        suggestions=suggestions,
        confidence_score=score,
        evidence_references=refs,
        strength_cards=[
            {"title": item[:80], "detail": item}
            for item in strengths[:3]
        ],
        concern_cards=[
            {"title": item[:80], "detail": item, "severity": "medium" if idx < 2 else "low"}
            for idx, item in enumerate(weaknesses[:3])
        ],
        section_observations=[
            {
                "section_name": "Overall submission",
                "observation": strengths[0] if strengths else "",
                "concern": weaknesses[0] if weaknesses else "",
                "next_step": suggestions[0] if suggestions else "",
            }
        ],
        marking_considerations=[
            "Check whether the final band language matches the strongest directly evidenced sections.",
            "Be cautious wherever the rationale depends on inference rather than explicit submission evidence.",
        ],
        moderation_notes=[
            "Treat the output as a structured moderation aid rather than a replacement for human judgement.",
            "Use the evidence references to support any final rubric or consistency decision.",
        ],
        action_recommendations=[
            "Recheck the weakest evidenced criterion before finalising the judgement.",
            "Confirm that the moderation explanation does not go beyond the available evidence.",
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
        confidence_explanation="Confidence depends on the strength of direct submission evidence, agreement with calibration signals, and the quality of the grounded moderation context.",
        evidence_coverage="Coverage is strongest where the submission clearly demonstrates the assessed criteria and weaker where the rationale depends on missing or indirect evidence.",
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
