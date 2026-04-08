"""Helpers for Phase 12 model_versions metadata.

This keeps LangGraph trace and engine metadata inside the existing
`model_versions` pattern used by the backend instead of introducing a new
response contract.
"""

from __future__ import annotations

from ..config import phase12_settings
from ..state import Phase12GraphState
from .graph_trace import build_execution_trace_metadata, trace_snapshot


def build_phase12_model_versions(state: Phase12GraphState) -> dict[str, object]:
    """Build Phase 12 model_versions metadata from the shared execution meta."""

    base = state.pipeline_context.execution_meta.to_model_versions_dict()
    trace_metadata = build_execution_trace_metadata(state)
    base["pipeline"] = "phase12_langgraph"
    base["graph"] = {
        "role": state.role,
        "graph_name": state.graph_name,
        "graph_version": state.graph_version,
        "prompt_version": state.prompt_version,
        "retrieval_policy_version": state.retrieval_policy_version,
        "tool_policy_version": state.tool_policy_version,
        "model_version": state.model_version,
        "schema_version": phase12_settings.trace_schema_version,
        "trace_enabled": phase12_settings.enable_trace_capture,
    }
    base["graph_trace_summary"] = state.export_compact_trace_summary()
    base["graph_execution"] = trace_metadata.model_dump(mode="json")
    if phase12_settings.persist_trace_to_model_versions:
        base["graph_trace"] = trace_snapshot(state)
    return base
