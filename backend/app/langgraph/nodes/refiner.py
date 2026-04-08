"""Refiner node for the Phase 15/16 self-critique loop."""

from __future__ import annotations

from app.genai.deterministic import build_professor_report, build_student_report
from app.genai.prompts import build_refiner_prompt
from app.langgraph.schemas import Phase12NodeDescriptor
from app.langgraph.state import Phase12GraphState
from ._genai_llm import invoke_json_prompt, primary_model_name
from ._helpers import succeed_node

NODE_NAME = "refiner"
NODE_SPEC = Phase12NodeDescriptor(
    name=NODE_NAME,
    description="Refine the draft using the critique while keeping the output schema-bounded.",
    wrapped_modules=["app.genai.prompts", "app.langchain.services.chain_factory"],
    reads=["draft_report", "critique_output"],
    writes=["repaired_report"],
)


async def refiner_node(state: Phase12GraphState) -> Phase12GraphState:
    """Refine the draft using the critique; degrade safely if needed."""

    state.set_current_node(NODE_NAME)
    critique = state.critique_output if isinstance(state.critique_output, dict) else {}
    refinement_required = bool(critique.get("refinement_required", False))

    if not refinement_required:
        state.repaired_report = dict(state.draft_report)
        return succeed_node(
            state,
            node_name=NODE_NAME,
            decision_type="refiner",
            branch="reuse_draft",
            reason="Validator critique did not require a refinement pass.",
            detail={"refinement_required": False},
            confidence=state.final_confidence or state.ml_confidence,
        )

    prompt_text = build_refiner_prompt(state, state.draft_report, critique)
    parsed, model_name, error = await invoke_json_prompt(
        role=state.role,
        prompt_text=prompt_text,
        model_name=primary_model_name(),
    )
    if parsed is None:
        state.add_warning("Refinement model output was unavailable or invalid; deterministic refinement was used.")
        refined = (
            build_student_report(state).model_dump(mode="json")
            if state.role == "student"
            else build_professor_report(state).model_dump(mode="json")
        )
        branch = "deterministic_refinement"
        reason = f"Refinement degraded safely: {error or 'invalid output'}"
    else:
        refined = parsed
        branch = "llm_refinement"
        reason = "Primary model refined the draft using the validator critique."

    state.repaired_report = refined
    state.pipeline_context.execution_meta.model_used = model_name
    return succeed_node(
        state,
        node_name=NODE_NAME,
        decision_type="refiner",
        branch=branch,
        reason=reason,
        detail={"model_used": model_name},
        confidence=max(state.ml_confidence, state.evidence_quality_score),
        safe_mode_triggered=state.safe_mode,
    )
