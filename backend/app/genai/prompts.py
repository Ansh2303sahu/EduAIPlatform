"""Prompt builders for the Phase 15/16 generative pipeline."""

from __future__ import annotations

import json
from typing import Any

from app.genai.config import genai_settings
from app.genai.schemas import CritiqueReport, ProfessorModerationReport, StudentReport
from app.langgraph.state import Phase12GraphState


def _safe_json(value: Any, *, limit: int = 6000) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    return text[:limit]


def _generator_role_brief(state: Phase12GraphState) -> str:
    if state.role == "student":
        return (
            "You are writing a student-facing university learning report. "
            "Be constructive, specific, and evidence-grounded."
        )
    return (
        "You are writing a professor-facing moderation report. "
        "Be defensible, concise, and safe for audit or second marking."
    )


def _generator_quality_floor(state: Phase12GraphState) -> str:
    if state.role == "student":
        return (
            "Output quality floor:\n"
            f"- summary must explain the overall judgement in 2 to 4 sentences.\n"
            "- overall_judgment must state the overall academic or technical standard in direct student-facing language.\n"
            f"- strengths should contain up to {genai_settings.max_strengths} distinct points; include at least 3 when the evidence clearly supports them.\n"
            f"- weaknesses should contain up to {genai_settings.max_weaknesses} distinct points; prioritise the most important learning gaps.\n"
            f"- suggestions should contain up to {genai_settings.max_suggestions} concrete next steps.\n"
            "- strength_cards and weakness_cards should turn the best strengths and weaknesses into title/detail items when the evidence allows.\n"
            "- section_feedback should cover distinct sections or dimensions instead of repeating the same point.\n"
            "- priority_issue should identify the highest-leverage fix and explain why it matters.\n"
            "- improvement_plan.actions should contain 2 to 4 concrete actions with rationale and practical steps when evidence is available.\n"
            "- learning_path.milestones should contain 2 to 4 sequenced milestones when evidence is available.\n"
            f"- reasoning_summary should contain 3 to {genai_settings.max_reasoning_steps} short strings summarising how the judgement was formed.\n"
            "- confidence_explanation should explain confidence in terms of evidence coverage, not in terms of system status.\n"
            "- counterfactual_explanation should explain what missing or stronger evidence could change the report.\n"
            f"- evidence_references should contain only grounded references and should never exceed {genai_settings.max_evidence_references} items."
        )
    return (
        "Output quality floor:\n"
        "- summary must explain the overall moderation stance in 2 to 4 sentences.\n"
        "- evaluator_overview must give the marker-facing overall judgement in direct, defensible language.\n"
        f"- strengths should contain up to {genai_settings.max_strengths} distinct, moderation-relevant positives.\n"
        f"- weaknesses should contain up to {genai_settings.max_weaknesses} distinct, moderation-relevant gaps.\n"
        f"- suggestions should contain up to {genai_settings.max_suggestions} practical marker or moderation actions.\n"
        "- feedback_explanation should synthesise strongest evidence, weakest evidence, and the most important moderation risk.\n"
        "- rubric_alignment should list concrete rubric-aligned dimensions rather than generic labels.\n"
        "- section_observations, marking_considerations, and action_recommendations should give a reviewer concrete next checks.\n"
        "- moderation_notes should identify what a human marker should verify, challenge, or confirm.\n"
        f"- reasoning_summary should contain 3 to {genai_settings.max_reasoning_steps} short strings summarising how the moderation judgement was formed.\n"
        "- confidence_explanation should explain confidence in terms of evidence coverage, consistency, and grounding.\n"
        "- counterfactual_explanation should explain what missing or stronger evidence could change the judgement.\n"
        f"- evidence_references should contain only grounded references and should never exceed {genai_settings.max_evidence_references} items."
    )


def build_generator_prompt(state: Phase12GraphState) -> str:
    """Build the primary generation prompt for gemma3:4b."""

    schema = StudentReport if state.role == "student" else ProfessorModerationReport
    review_mode_line = (
        "Set safety.needs_review to true and lower confidence if evidence is weak, conflicting, or incomplete."
        if state.safe_mode
        else "Keep safety.needs_review false unless the evidence clearly requires escalation."
    )
    return (
        f"{_generator_role_brief(state)}\n"
        "Return exactly one valid JSON object and nothing else.\n"
        "Do not add markdown fences, commentary, or prose outside the JSON.\n"
        "Use only the provided submission evidence, ML context, and approved RAG context.\n"
        "Do not fabricate references, bibliography entries, rubric rules, plagiarism claims, or external facts.\n"
        "If evidence is partial, say so in the relevant fields instead of inventing support.\n\n"
        "Avoid system-centric phrasing such as describing the platform, pipeline, or model unless a confidence or safety field explicitly requires it.\n"
        "Submission evidence is primary. Start with the submission itself before using retrieved context.\n"
        "For student writing, prioritise paragraph clarity, claim specificity, evidence integration, citation consistency, and argument flow.\n"
        "Use retrieved context only as secondary support for referencing rules, structure expectations, or rubric alignment when it clearly matches the submission topic.\n"
        "If retrieved grounding is weak or rejected, ignore it and critique from the submission alone.\n"
        "Every criticism must quote or closely paraphrase a specific sentence, claim, or section from the submission, explain why it is weak or strong, and suggest a direct revision or review step.\n\n"
        f"{_generator_quality_floor(state)}\n\n"
        "Schema reminders:\n"
        "- Keep list items distinct rather than repetitive.\n"
        "- Prefer concise, information-dense strings over long generic paragraphs.\n"
        "- confidence.score must match the actual evidence strength.\n"
        "- safety.reason should briefly explain any escalation.\n"
        f"- {review_mode_line}\n\n"
        "Execution metadata:\n"
        f"- role: {state.role}\n"
        f"- analysis_type: {state.analysis_type.value}\n"
        f"- safe_mode: {state.safe_mode}\n\n"
        f"Evidence summary:\n{state.evidence_summary}\n\n"
        f"ML context:\n{(state.pipeline_context.ml_context_text or '')[:2600]}\n\n"
        f"RAG context:\n{(state.pipeline_context.rag.context_text or '')[:3000]}\n\n"
        f"Section plan:\n{_safe_json(state.section_plan, limit=1800)}\n\n"
        "Return JSON matching this schema shape:\n"
        f"{_safe_json(schema.model_json_schema(), limit=3600)}"
    )


def build_critic_prompt(state: Phase12GraphState, draft_report: dict[str, Any]) -> str:
    """Build the validator/critic prompt for phi3:mini."""

    return (
        "You are a strict validator reviewing a generated educational report.\n"
        "Return exactly one valid JSON object and nothing else.\n"
        "Do not rewrite the report. Diagnose whether it is grounded, fair, coherent, and schema-complete.\n"
        "Focus on unsupported claims, contradiction, duplicated points, weak prioritisation, tone mismatch, generic filler, and confidence that is too high for the evidence.\n"
        "Flag the draft when criticism is not tied to quoted or closely paraphrased submission content.\n"
        "If the draft omits important grounded content that should be present, flag that as a concern.\n"
        "Do not invent references or new evidence.\n"
        "Set refinement_required to true when the draft should be revised before it is shown to a user.\n\n"
        f"Role: {state.role}\n"
        f"Safe mode: {state.safe_mode}\n"
        f"Evidence summary:\n{state.evidence_summary}\n\n"
        f"RAG trace:\n{_safe_json(state.pipeline_context.rag.trace, limit=2000)}\n\n"
        f"Draft report:\n{_safe_json(draft_report, limit=3600)}\n\n"
        "Return JSON matching this schema shape:\n"
        f"{_safe_json(CritiqueReport.model_json_schema(), limit=2400)}"
    )


def build_refiner_prompt(
    state: Phase12GraphState,
    draft_report: dict[str, Any],
    critique: dict[str, Any],
) -> str:
    """Build the refinement prompt for the final gemma3:4b pass."""

    schema = StudentReport if state.role == "student" else ProfessorModerationReport
    return (
        "Refine the draft report using the critique.\n"
        "Return exactly one valid JSON object and nothing else.\n"
        "Preserve grounded content that is already good, but fix the concerns raised by the validator.\n"
        "Keep the output specific, compact, and evidence-grounded.\n"
        "Do not invent sources, bibliography entries, rubric rules, or missing evidence.\n"
        "If the critique identifies weak confidence or missing review escalation, adjust confidence and safety fields accordingly.\n\n"
        "Keep submission evidence primary and quote or closely paraphrase the relevant student text when making criticism or revision advice.\n\n"
        f"{_generator_quality_floor(state)}\n\n"
        f"Draft report:\n{_safe_json(draft_report, limit=3400)}\n\n"
        f"Critique:\n{_safe_json(critique, limit=2200)}\n\n"
        f"Evidence summary:\n{state.evidence_summary}\n\n"
        "Return JSON matching this schema shape:\n"
        f"{_safe_json(schema.model_json_schema(), limit=3400)}"
    )
