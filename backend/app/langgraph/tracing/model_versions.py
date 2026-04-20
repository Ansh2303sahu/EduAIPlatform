"""Helpers for Phase 12 model_versions metadata.

This keeps LangGraph trace and engine metadata inside the existing
`model_versions` pattern used by the backend instead of introducing a new
response contract.
"""

from __future__ import annotations

from typing import Any

from app.mcp.config import mcp_settings
from app.services.report_richness import (
    build_report_preview,
    extract_best_summary,
    report_low_content_quality,
)
from app.services.uuid_normalization import uuid_or_none

from ..config import phase12_settings
from ..state import Phase12GraphState
from .graph_trace import build_execution_trace_metadata, trace_snapshot


UUID_LIKE_KEYS = {
    "id",
    "file_id",
    "user_id",
    "report_id",
    "request_id",
    "execution_id",
    "trace_id",
    "run_id",
    "job_id",
    "session_id",
    "parent_id",
}


def _clean_text(value: object) -> str:
    return str(value or "").strip()


def _optional_text(value: object) -> str | None:
    text = _clean_text(value)
    return text or None


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_uuid_like_key(key: object) -> bool:
    if not isinstance(key, str):
        return False
    key = key.strip().lower()
    return key in UUID_LIKE_KEYS or key.endswith("_id")


def _normalize_uuid_like_blanks(value: Any, parent_key: str | None = None) -> Any:
    """Recursively convert blank UUID-like values to None.

    This does not attempt to validate UUID format. It only prevents empty-string
    identifiers from leaking into stored metadata and downstream insert payloads.
    """
    if isinstance(value, dict):
        return {
            k: _normalize_uuid_like_blanks(v, str(k))
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [_normalize_uuid_like_blanks(item, parent_key) for item in value]

    if isinstance(value, str) and _is_uuid_like_key(parent_key):
        return uuid_or_none(value)

    return value


def _confidence_band(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


def _confidence_bucket(score: float) -> int:
    return max(0, min(4, int(round(max(0.0, min(1.0, score)) * 4))))


def _report_llm_confidence(state: Phase12GraphState, final_confidence: float) -> float:
    report = state.pipeline_context.report or {}
    if not isinstance(report, dict):
        return final_confidence

    agreement = report.get("model_agreement") if isinstance(report.get("model_agreement"), dict) else {}
    confidence = report.get("confidence") if isinstance(report.get("confidence"), dict) else {}
    saw_zero_confidence = False
    for value in (
        agreement.get("llm_confidence"),
        confidence.get("overall"),
        confidence.get("score"),
        report.get("confidence_score"),
    ):
        score = _safe_float(value, default=-1.0)
        if score > 0.0:
            return max(0.0, min(1.0, score))
        saw_zero_confidence = saw_zero_confidence or score == 0.0
    if saw_zero_confidence and final_confidence <= 0.0:
        return 0.0
    return final_confidence


def _report_summary(state: Phase12GraphState) -> str:
    report = state.pipeline_context.report or {}
    return _clean_text(extract_best_summary(state.role, report))


def _ml_model_names(state: Phase12GraphState) -> list[str]:
    ml_result = state.pipeline_context.ml_result
    if ml_result is not None:
        model_metadata = ml_result.model_metadata or {}
        raw_models = model_metadata.get("models") or []
        names: list[str] = []

        for item in raw_models:
            if not isinstance(item, dict):
                continue

            source = _clean_text(item.get("source"))
            model = _clean_text(item.get("model"))
            version = _clean_text(item.get("version"))

            label = ":".join(part for part in (source, model) if part)
            if version:
                label = f"{label}@{version}" if label else version

            if label and label not in names:
                names.append(label)

        if names:
            return names

    if state.role == "student":
        return [
            "student.feedback_classifier_multimodal.v1",
            "student.confidence_model_multimodal.v1",
        ]
    return ["professor.rubric_suite_multimodal.v1"]


def _llm_route(primary_model: str, fallback_model: str, model_used: str) -> str:
    primary = _clean_text(primary_model)
    fallback = _clean_text(fallback_model)
    used = _clean_text(model_used)

    if primary and fallback and fallback != primary:
        return f"{primary} -> {fallback}"
    return used or primary or fallback


def _student_low_content_quality(report: dict[str, object]) -> bool:
    return report_low_content_quality("student", report)


def build_phase12_model_versions(state: Phase12GraphState) -> dict[str, object]:
    """Build Phase 12 model_versions metadata from the shared execution meta."""

    base = state.pipeline_context.execution_meta.to_model_versions_dict()
    base = _normalize_uuid_like_blanks(base)

    trace_metadata = build_execution_trace_metadata(state)
    timings = dict(state.pipeline_context.timings_ms or {})

    if "total" not in timings and state.total_latency_ms:
        timings["total"] = int(round(state.total_latency_ms))

    if timings:
        base["timings_ms"] = timings

    retrieval_debug = dict(base.get("retrieval_debug") or {})
    retrieval_debug.setdefault("chunk_count", len(state.retrieved_chunks))
    retrieval_debug.setdefault("confidence_score", _safe_float(state.rag_result.confidence_score))
    retrieval_debug.setdefault("confidence_label", _clean_text(state.rag_result.confidence_label) or "low")
    retrieval_debug.setdefault("weak_retrieval", bool(state.rag_result.weak_retrieval))
    retrieval_debug.setdefault("safe_review", bool(state.rag_result.safe_review))
    base["retrieval_debug"] = retrieval_debug

    final_status = state.final_status.value if state.final_status else state.status.value
    confidence_score = _safe_float(state.final_confidence)
    ml_confidence_score = _safe_float(state.ml_confidence)
    llm_confidence_score = _report_llm_confidence(state, confidence_score)

    base["agreement"] = {
        **dict(base.get("agreement") or {}),
        "final_confidence": confidence_score,
        "ml_confidence": ml_confidence_score,
        "llm_confidence": llm_confidence_score,
        "ml_bucket_0_to_4": _confidence_bucket(ml_confidence_score),
    }

    if state.model_version:
        base["llm_model_used"] = _clean_text(state.model_version)

    if state.role == "student":
        base["ml_models"] = {
            "feedback": "student.feedback_classifier_multimodal.v1",
            "confidence": "student.confidence_model_multimodal.v1",
        }
    else:
        base["ml_models"] = {"rubric_suite": "professor.rubric_suite_multimodal.v1"}

    base["pipeline"] = "phase12_langgraph"
    base["graph"] = {
        "role": state.role,
        "graph_name": _clean_text(state.graph_name),
        "graph_version": _clean_text(state.graph_version),
        "prompt_version": _clean_text(state.prompt_version),
        "retrieval_policy_version": _clean_text(state.retrieval_policy_version),
        "tool_policy_version": _clean_text(state.tool_policy_version),
        "model_version": _optional_text(state.model_version),
        "schema_version": _clean_text(phase12_settings.trace_schema_version),
        "trace_enabled": bool(phase12_settings.enable_trace_capture),
    }

    trace_summary = state.export_compact_trace_summary()
    base["graph_trace_summary"] = trace_summary
    base["graph_execution"] = _normalize_uuid_like_blanks(trace_metadata.model_dump(mode="json"))
    base["report_preview"] = build_report_preview(state.role, state.pipeline_context.report or {})

    base["langchain"] = {
        "available": bool(state.pipeline_context.report or state.pipeline_context.raw_llm_output),
        "pipeline": "phase12_langgraph",
        "chain_name": _clean_text(base.get("chain_name")),
        "chain_version": _clean_text(base.get("chain_version")),
        "prompt_version": _clean_text(state.prompt_version),
        "schema_version": _clean_text(base.get("schema_version")),
        "provider": _clean_text(base.get("provider")),
        "model_used": _clean_text(base.get("llm_model_used")),
        "primary_model": _clean_text(base.get("llm_primary")),
        "fallback_model": _clean_text(base.get("llm_fallback")),
        "fallback_used": bool(base.get("fallback_used", False)),
        "execution_mode": _clean_text(base.get("execution_mode")),
        "decision_source": _clean_text(base.get("decision_source")),
        "retrieval_mode": _clean_text(retrieval_debug.get("mode")),
        "retrieved_chunk_count": len(state.retrieved_chunks),
        "confidence_score": confidence_score,
        "summary": (
            f"LangChain generation completed with {len(state.retrieved_chunks)} retrieved chunk(s)."
            if state.retrieved_chunks
            else "LangChain generation completed without stored retrieved chunks."
        ),
    }

    base["langgraph"] = {
        "available": True,
        "pipeline": "phase12_langgraph",
        "graph_name": _clean_text(state.graph_name),
        "graph_version": _clean_text(state.graph_version),
        "prompt_version": _clean_text(state.prompt_version),
        "output_version": None,
        "final_status": _clean_text(final_status),
        "safe_mode": bool(state.safe_mode),
        "total_steps": int(state.total_steps or 0),
        "total_latency_ms": int(state.total_latency_ms or 0),
        "node_count": len(state.node_traces),
        "decision_count": len(state.decision_traces),
        "failure_count": len(state.failure_traces),
        "trace_summary": trace_summary,
        "warnings": list(state.warnings),
    }

    base["genai"] = {
        "available": bool(state.pipeline_context.report),
        "pipeline": "phase12_langgraph",
        "model_version": _clean_text(base.get("llm_model_used") or base.get("llm_primary")),
        "validator_model_version": None,
        "final_status": _clean_text(final_status),
        "confidence_score": confidence_score,
        "confidence_band": _confidence_band(confidence_score),
        "report_summary": _report_summary(state),
        "warning_count": len(state.warnings),
    }

    base["ml"] = {
        "available": bool(state.pipeline_context.ml_raw or state.pipeline_context.ml_result),
        "confidence_score": _safe_float(state.ml_confidence),
        "model_names": _ml_model_names(state),
        "source": "phase12_langgraph",
        "summary": (
            "Stored ML calibration signals were carried into the LangGraph run."
            if state.pipeline_context.ml_raw or state.pipeline_context.ml_result
            else "No ML calibration signal was stored for this LangGraph run."
        ),
    }

    base["llm"] = {
        "available": bool(base.get("llm_model_used") or base.get("llm_primary")),
        "model_used": _clean_text(base.get("llm_model_used")),
        "primary_model": _clean_text(base.get("llm_primary")),
        "fallback_model": _clean_text(base.get("llm_fallback")),
        "route": _llm_route(
            _clean_text(base.get("llm_primary")),
            _clean_text(base.get("llm_fallback")),
            _clean_text(base.get("llm_model_used")),
        ),
        "source": "phase12_langgraph",
    }

    graph_used = any(entry.node_name == "mcp_tools" for entry in state.node_traces)
    base["mcp"] = {
        "enabled": bool(mcp_settings.enabled),
        "orchestration_enabled": bool(mcp_settings.orchestration_enabled),
        "llm_enabled": bool(mcp_settings.llm_enabled),
        "graph_used": graph_used,
        "tool_call_count": int(state.tool_call_count or 0),
        "visible_tools": list(state.selected_tools),
        "summary": (
            "MCP tool orchestration executed inside the LangGraph run."
            if graph_used
            else "No MCP tool step executed for this LangGraph run."
        ),
    }

    quality_gate_degraded = (
        _student_low_content_quality(state.pipeline_context.report or {})
        if state.role == "student"
        else False
    )
    base["quality_gate"] = {
        "degraded_placeholder": quality_gate_degraded,
        "reason": "low_content_quality" if quality_gate_degraded else None,
    }

    if phase12_settings.persist_trace_to_model_versions:
        base["graph_trace"] = _normalize_uuid_like_blanks(trace_snapshot(state))

    return _normalize_uuid_like_blanks(base)
