"""Planner node for the Phase 15/16 generative pipeline."""

from __future__ import annotations

from app.genai.schemas import PlanSection
from app.langgraph.schemas import Phase12NodeDescriptor
from app.langgraph.state import Phase12GraphState
from ._helpers import succeed_node

NODE_NAME = "planner"
NODE_SPEC = Phase12NodeDescriptor(
    name=NODE_NAME,
    description="Create a bounded section plan for the structured report.",
    wrapped_modules=["app.genai.schemas"],
    reads=["storage_payload", "pipeline_context.analysis_type"],
    writes=["section_plan", "storage_payload"],
)


def _student_plan(state: Phase12GraphState) -> list[PlanSection]:
    return [
        PlanSection(title="Summary", objective="State the core evaluation succinctly."),
        PlanSection(title="Strengths", objective="List the strongest grounded positives."),
        PlanSection(title="Weaknesses", objective="List the main grounded limitations."),
        PlanSection(title="Suggestions", objective="Provide actionable next steps."),
        PlanSection(title="Improvement Plan", objective="Turn suggestions into a short plan."),
        PlanSection(title="Learning Path", objective="Offer follow-on practice guidance."),
    ]


def _professor_plan(state: Phase12GraphState) -> list[PlanSection]:
    return [
        PlanSection(title="Summary", objective="State the moderation judgement succinctly."),
        PlanSection(title="Feedback Explanation", objective="Explain the judgement clearly."),
        PlanSection(title="Strengths", objective="List grounded positives."),
        PlanSection(title="Weaknesses", objective="List grounded limitations."),
        PlanSection(title="Suggestions", objective="Give moderation or feedback improvements."),
        PlanSection(title="Moderation Notes", objective="Surface trust and consistency notes."),
    ]


async def planner_node(state: Phase12GraphState) -> Phase12GraphState:
    """Create a deterministic section plan for the later LLM passes."""

    state.set_current_node(NODE_NAME)
    sections = _student_plan(state) if state.role == "student" else _professor_plan(state)
    state.section_plan = [section.title for section in sections]
    state.storage_payload["plan_sections"] = [section.model_dump(mode="json") for section in sections]
    return succeed_node(
        state,
        node_name=NODE_NAME,
        decision_type="planner",
        branch="planned",
        reason="A bounded section plan was created for structured generation.",
        detail={"sections": state.section_plan},
        confidence=max(state.evidence_quality_score, 0.4),
    )
