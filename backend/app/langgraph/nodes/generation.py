"""Generation node for Phase 12.

This node keeps `app.langchain.services.chain_factory` as the only generation
boundary to llm-service and reuses the existing prompt builders and fallback
runner from the Phase 10 LangChain pipelines.

Phase 14 addition: after the draft is generated, emit ``ai.assessment.requested``
to n8n to trigger the multi-model assessment workflow (OpenAI → Claude → Gemini).
The emit is fire-and-forget — failures are logged but never fail the pipeline.
"""

import logging

from app.langchain.prompts.professor import build_professor_prompt, build_professor_safe_prompt
from app.langchain.prompts.student import build_student_prompt, build_student_safe_prompt
from app.langchain.services.chain_factory import build_generation_chain, get_primary_model
from app.langchain.services.fallback_service import run_with_fallback

from ..adapters import report_support
from ..schemas import Phase12NodeDescriptor
from ..state import Phase12GraphState
from ..tracing.graph_trace import record_event

logger = logging.getLogger(__name__)

NODE_NAME = "generation"
NODE_SPEC = Phase12NodeDescriptor(
    name=NODE_NAME,
    description=(
        "Build prompts and call llm-service through the existing chain factory boundary. "
        "Emits ai.assessment.requested to n8n for Phase 14 multi-model pass (fire-and-forget)."
    ),
    wrapped_modules=[
        "app.langchain.prompts.student",
        "app.langchain.prompts.professor",
        "app.langchain.services.chain_factory",
        "app.langchain.services.fallback_service",
    ],
    reads=["pipeline_context"],
    writes=["pipeline_context.prompt_text", "pipeline_context.raw_llm_output", "pipeline_context.model_used"],
)


def _build_prompt(state: Phase12GraphState) -> str:
    restricted = state.pipeline_context.execution_mode.value != "normal"
    if state.role == "student":
        return (
            build_student_safe_prompt(state.pipeline_context)
            if restricted
            else build_student_prompt(state.pipeline_context)
        )
    return (
        build_professor_safe_prompt(state.pipeline_context)
        if restricted
        else build_professor_prompt(state.pipeline_context)
    )


async def _emit_assessment_requested(state: Phase12GraphState) -> None:
    """Fire-and-forget: emit ai.assessment.requested to n8n.

    Triggers the Phase 14 multi-model assessment workflow.  Any failure here
    is logged as a warning and swallowed — the Phase 12 pipeline continues
    regardless, persisting the single-model draft as usual.
    """
    try:
        from app.events.config import get_event_settings
        from app.events.emitter import send_event_to_n8n
        from app.events.schemas import EventType
        from app.schemas.assessment import AssessmentRequestedPayload

        cfg = get_event_settings()
        role = state.role

        # Build minimal payload — no raw text or PII beyond IDs
        payload = AssessmentRequestedPayload(
            file_id=state.pipeline_context.file_id,
            user_id=state.pipeline_context.user_id,
            role=role,
            submission_id=getattr(state.pipeline_context, "submission_id", ""),
            draft_confidence=float(
                getattr(state.pipeline_context.execution_meta, "final_confidence", 0.0)
                or getattr(state.pipeline_context, "ml_confidence", 0.0)
                or 0.0
            ),
            pipeline="phase12_langgraph",
        )

        webhook_url = (
            cfg.assessment_student_url
            if role == "student"
            else cfg.assessment_professor_url
        )

        result = await send_event_to_n8n(
            EventType.AI_ASSESSMENT_REQUESTED,
            payload,
            webhook_url=webhook_url,
            correlation_id=state.pipeline_context.request_id,
        )
        if result.success:
            logger.info(
                "assessment_requested_emitted",
                extra={
                    "event_id": result.event_id,
                    "file_id": payload.file_id,
                    "role": role,
                },
            )
        else:
            logger.warning(
                "assessment_requested_emit_failed",
                extra={
                    "event_id": result.event_id,
                    "error": result.error,
                    "http_status": result.http_status,
                },
            )
    except Exception as exc:
        # Never propagate — Phase 12 pipeline must not fail due to Phase 14
        logger.warning(
            "assessment_requested_emit_exception",
            extra={"error": f"{type(exc).__name__}: {exc}"},
        )


async def run(state: Phase12GraphState) -> Phase12GraphState:
    """Generate raw model output through the current Phase 10 generation stack."""

    prompt_text = _build_prompt(state)
    state.pipeline_context.prompt_text = prompt_text
    state.prompt_hash = report_support.sha256_json(
        {
            "role": state.role,
            "analysis_type": state.pipeline_context.analysis_type.value,
            "prompt_text": prompt_text,
        }
    )
    primary_model = get_primary_model(role=state.role)
    primary_chain = build_generation_chain(primary_model)
    raw_text, model_used = await run_with_fallback(
        primary_chain,
        {"prompt_text": prompt_text},
        role=state.role,
        request_id=state.pipeline_context.request_id,
    )
    state.pipeline_context.raw_llm_output = raw_text
    state.pipeline_context.model_used = model_used
    state.pipeline_context.execution_meta.model_used = model_used
    state.pipeline_context.execution_meta.fallback_used = model_used != primary_model
    record_event(
        state,
        NODE_NAME,
        {
            "model_used": model_used,
            "fallback_used": state.pipeline_context.execution_meta.fallback_used,
        },
    )

    # Phase 14: trigger multi-model assessment workflow (fire-and-forget)
    await _emit_assessment_requested(state)

    return state
