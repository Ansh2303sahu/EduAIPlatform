"""Generator node for the Phase 15/16 self-critique loop."""

from __future__ import annotations

from app.genai.config import genai_settings
from app.genai.deterministic import build_professor_report, build_student_report
from app.genai.prompts import build_generator_prompt
from app.langgraph.schemas import Phase12NodeDescriptor
from app.langgraph.state import Phase12GraphState
from app.services import report_generation_support as support
from ._genai_llm import invoke_json_prompt, primary_model_name
from ._helpers import succeed_node

NODE_NAME = "generator"
NODE_SPEC = Phase12NodeDescriptor(
    name=NODE_NAME,
    description="Generate the first structured report draft using the secure llm-service chain boundary.",
    wrapped_modules=["app.genai.prompts", "app.langchain.services.chain_factory"],
    reads=["section_plan", "pipeline_context.ml_result", "pipeline_context.rag"],
    writes=["draft_report", "prompt_hash", "pipeline_context.execution_meta"],
)


async def generator_node(state: Phase12GraphState) -> Phase12GraphState:
    """Generate the first report draft; degrade deterministically on failure."""

    state.set_current_node(NODE_NAME)
    prompt_text = build_generator_prompt(state)
    state.prompt_hash = support.sha256_json(
        {"role": state.role, "prompt_version": genai_settings.prompt_version, "prompt_text": prompt_text}
    )
    parsed, model_name, error = await invoke_json_prompt(
        role=state.role,
        prompt_text=prompt_text,
        model_name=primary_model_name(),
    )
    state.pipeline_context.execution_meta.model_used = model_name
    state.pipeline_context.execution_meta.primary_model = model_name

    if parsed is None:
        state.add_warning("Primary generator output was unavailable or invalid; a deterministic draft was used.")
        draft = (
            build_student_report(state).model_dump(mode="json")
            if state.role == "student"
            else build_professor_report(state).model_dump(mode="json")
        )
        branch = "deterministic_draft"
        reason = f"Generation degraded safely: {error or 'invalid output'}"
    else:
        draft = parsed
        branch = "llm_draft"
        reason = "Primary model generated a structured draft."

    state.draft_report = draft
    return succeed_node(
        state,
        node_name=NODE_NAME,
        decision_type="generator",
        branch=branch,
        reason=reason,
        detail={"model_used": model_name, "has_draft": bool(draft)},
        confidence=max(state.ml_confidence, state.evidence_quality_score),
        safe_mode_triggered=state.safe_mode,
    )
