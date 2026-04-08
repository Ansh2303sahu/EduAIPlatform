"""Prompt builders for the Phase 15/16 generative pipeline."""

from __future__ import annotations

import json
from typing import Any

from app.genai.schemas import CritiqueReport, ProfessorModerationReport, StudentReport
from app.langgraph.state import Phase12GraphState


def _safe_json(value: Any, *, limit: int = 6000) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    return text[:limit]


def build_generator_prompt(state: Phase12GraphState) -> str:
    """Build the primary generation prompt for mistral."""

    schema = StudentReport if state.role == "student" else ProfessorModerationReport
    role_instructions = (
        "You are writing constructive student-facing academic feedback."
        if state.role == "student"
        else "You are writing professor-facing moderation feedback."
    )
    return (
        f"{role_instructions}\n"
        "Use the provided evidence only. Do not fabricate references, bibliography entries, "
        "or external facts. Do not claim plagiarism detection. Cite only retrieved evidence snippets.\n\n"
        f"Execution metadata:\n"
        f"- Role: {state.role}\n"
        f"- Analysis type: {state.analysis_type.value}\n"
        f"- Safe mode: {state.safe_mode}\n\n"
        f"Evidence summary:\n{state.evidence_summary}\n\n"
        f"ML context:\n{state.pipeline_context.ml_context_text[:2200]}\n\n"
        f"RAG context:\n{state.pipeline_context.rag.context_text[:2600]}\n\n"
        f"Section plan:\n{_safe_json(state.section_plan, limit=1600)}\n\n"
        f"Return JSON matching this schema shape:\n{_safe_json(schema.model_json_schema(), limit=3200)}\n\n"
        "Set safety.needs_review to true if the evidence is weak or inconsistent."
    )


def build_critic_prompt(state: Phase12GraphState, draft_report: dict[str, Any]) -> str:
    """Build the validator/critic prompt for phi3."""

    return (
        "You are a strict validator reviewing a generated educational report.\n"
        "Do not rewrite the report. Critique it for grounding, fairness, contradictions, "
        "score-vs-feedback mismatch, and unsupported claims.\n"
        "Do not invent references.\n\n"
        f"Role: {state.role}\n"
        f"Evidence summary: {state.evidence_summary}\n"
        f"RAG trace:\n{_safe_json(state.pipeline_context.rag.trace, limit=1800)}\n\n"
        f"Draft report:\n{_safe_json(draft_report, limit=3200)}\n\n"
        f"Return JSON matching this schema shape:\n{_safe_json(CritiqueReport.model_json_schema(), limit=2200)}"
    )


def build_refiner_prompt(
    state: Phase12GraphState,
    draft_report: dict[str, Any],
    critique: dict[str, Any],
) -> str:
    """Build the refinement prompt for the final mistral pass."""

    schema = StudentReport if state.role == "student" else ProfessorModerationReport
    return (
        "Refine the draft report using the critique. Keep the structure compact, grounded, "
        "and specific. Do not invent sources or bibliography entries.\n\n"
        f"Draft report:\n{_safe_json(draft_report, limit=3000)}\n\n"
        f"Critique:\n{_safe_json(critique, limit=2000)}\n\n"
        f"Evidence summary:\n{state.evidence_summary}\n\n"
        f"Return JSON matching this schema shape:\n{_safe_json(schema.model_json_schema(), limit=3000)}"
    )
