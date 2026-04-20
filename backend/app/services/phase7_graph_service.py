from __future__ import annotations

import time
from typing import Any, Literal

from app.core.deps import CurrentUser
from app.langchain.config import phase10_settings
from app.langchain.models import IngestionBundle
from app.langgraph.adapters import storage as graph_storage
from app.langgraph.graphs.professor_graph import get_professor_compiled_graph
from app.langgraph.graphs.student_graph import get_student_compiled_graph
from app.langgraph.schemas import Phase12ExecutionRequest
from app.langgraph.state import GraphExecutionStatus, Phase12GraphState
from app.langgraph.tracing.graph_trace import finalize_trace
from app.services import report_generation_support as support


GraphRole = Literal["student", "professor"]


class Phase7GraphService:
    """Run the existing student/professor Phase 12 graphs for the live Phase 7 flow."""

    async def run_generation(
        self,
        *,
        role: GraphRole,
        file_id: str,
        user: CurrentUser,
        file_row: dict[str, Any],
        ingestion_dict: dict[str, Any],
        input_hash: str,
        prompt_hash: str,
    ) -> Phase12GraphState:
        request = Phase12ExecutionRequest(
            file_id=file_id,
            user_id=str(user.id),
            role=role,
            correlation_id=str(file_row.get("submission_id") or file_id),
            submission_id=support.uuid_or_none(file_row.get("submission_id")) or "",
            user_email=user.email,
            raw_claims=user.raw_claims,
            file_metadata={
                "mime_type": file_row.get("mime_type"),
                "created_at": file_row.get("created_at"),
                "status": file_row.get("status"),
            },
        )
        state = Phase12GraphState.create(request)
        state.graph_name = (
            "phase12_student_graph" if role == "student" else "phase12_professor_graph"
        )
        state.graph_version = phase10_settings.chain_version
        state.file_row = dict(file_row or {})
        state.pipeline_context.ingestion = IngestionBundle.model_validate(ingestion_dict)
        state.pipeline_context.submission_id = request.submission_id
        state.pipeline_context.execution_meta.pipeline = "phase12_langgraph"
        state.pipeline_context.execution_meta.provider = phase10_settings.provider
        state.pipeline_context.execution_meta.primary_model = phase10_settings.llm_primary_label
        state.pipeline_context.execution_meta.fallback_model = phase10_settings.llm_fallback_label
        state.pipeline_context.execution_meta.chain_name = f"phase10_{role}_generation"
        state.pipeline_context.execution_meta.chain_version = phase10_settings.chain_version
        state.pipeline_context.execution_meta.student_prompt_version = phase10_settings.student_prompt_version
        state.pipeline_context.execution_meta.professor_prompt_version = phase10_settings.professor_prompt_version
        state.pipeline_context.execution_meta.schema_version = phase10_settings.schema_version
        state.input_hash = input_hash
        state.prompt_hash = prompt_hash

        submission_kind = support.detect_submission_kind(ingestion_dict)
        state.apply_submission_kind(submission_kind)

        started = time.perf_counter()
        compiled = (
            get_student_compiled_graph()
            if role == "student"
            else get_professor_compiled_graph()
        )
        raw = await compiled.ainvoke(state)
        final_state = raw if isinstance(raw, Phase12GraphState) else Phase12GraphState.model_validate(raw)

        total_ms = int((time.perf_counter() - started) * 1000)
        final_state.pipeline_context.timings_ms.setdefault("total", total_ms)
        final_state.pipeline_context.execution_meta.timings_ms = dict(final_state.pipeline_context.timings_ms)

        if final_state.final_status is None:
            terminal_status = (
                GraphExecutionStatus.COMPLETED
                if final_state.pipeline_context.report
                else GraphExecutionStatus.FAILED
            )
            finalize_trace(
                final_state,
                final_status=terminal_status.value,
                reason="Phase 7 live LangGraph execution completed.",
            )

        stored_row = final_state.storage_payload.get("stored_row")
        if not stored_row and final_state.pipeline_context.report:
            stored_row = await graph_storage.persist_ai_report(final_state)
            final_state.storage_payload["stored_row"] = stored_row

        return final_state
