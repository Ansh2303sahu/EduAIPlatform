"""Critic node for the Phase 15/16 self-critique loop."""

from __future__ import annotations

from app.genai.deterministic import build_deterministic_critique
from app.genai.prompts import build_critic_prompt
from app.genai.schemas import CritiqueReport
from app.langgraph.schemas import Phase12NodeDescriptor
from app.langgraph.state import Phase12GraphState
from ._genai_llm import invoke_json_prompt, validator_model_name
from ._helpers import succeed_node

NODE_NAME = "critic"
NODE_SPEC = Phase12NodeDescriptor(
    name=NODE_NAME,
    description="Critique the initial draft with the validator model to surface grounding and fairness risks.",
    wrapped_modules=["app.genai.prompts", "app.langchain.services.chain_factory"],
    reads=["draft_report", "pipeline_context.rag", "pipeline_context.ml_result"],
    writes=["critique_output", "pipeline_context.execution_meta"],
)


async def critic_node(state: Phase12GraphState) -> Phase12GraphState:
    """Critique the first draft using phi3; degrade deterministically on failure."""

    state.set_current_node(NODE_NAME)
    prompt_text = build_critic_prompt(state, state.draft_report)
    parsed, model_name, error = await invoke_json_prompt(
        role=state.role,
        prompt_text=prompt_text,
        model_name=validator_model_name(),
    )

    if parsed is None:
        critique = build_deterministic_critique(state, state.draft_report)
        branch = "deterministic_critique"
        reason = f"Validator critique degraded safely: {error or 'invalid output'}"
    else:
        try:
            critique = CritiqueReport.model_validate(parsed)
            branch = "validator_critique"
            reason = "Validator model produced a structured critique."
        except Exception:
            critique = build_deterministic_critique(state, state.draft_report)
            branch = "deterministic_critique"
            reason = "Validator output did not match the critique schema; deterministic critique used."

    state.critique_output = critique.model_dump(mode="json")
    state.pipeline_context.execution_meta.agreement_score = critique.confidence_score
    state.storage_payload["validator_model"] = model_name
    return succeed_node(
        state,
        node_name=NODE_NAME,
        decision_type="critic",
        branch=branch,
        reason=reason,
        detail={"validator_model": model_name, "refinement_required": critique.refinement_required},
        confidence=critique.confidence_score,
        safe_mode_triggered=state.safe_mode,
    )
