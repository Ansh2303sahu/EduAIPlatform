"""Phase 15/16 service orchestration."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import HTTPException

from app.core.deps import CurrentUser
from app.events.config import get_event_settings
from app.genai.config import genai_settings
from app.genai.consistency import detect_report_contradictions
from app.genai.explainability import build_explanation_tab, build_sources_tab, confidence_band
from app.genai.fairness import build_fairness_tab
from app.genai.pdf_service import build_pdf_base64
from app.genai.schemas import (
    AICheckGenAI,
    AICheckLLM,
    AICheckLangChain,
    AICheckLangGraph,
    AICheckMCP,
    AICheckML,
    AICheckN8N,
    AICheckN8NIntegrations,
    AICheckRAG,
    AICheckResponse,
    AICheckSummary,
    AIReportGenerateIn,
    AIReportResponse,
    AuditResponse,
    AuditTab,
    CompareResponse,
    ExplainResponse,
    ExplanationTab,
    FairnessTab,
    PredictionTab,
    ProfessorModerationReport,
    SourcesTab,
    StoredSummary,
    StudentReport,
)
from app.langchain.enums import DecisionSource
from app.langchain.models import IngestionBundle, ValidationResult
from app.langchain.services.ml_context_builder import (
    normalize_professor_ml_context,
    normalize_student_ml_context,
)
from app.langchain.services.retrieval_packager import pack_professor_rag, pack_student_rag
from app.langgraph.graphs.professor_generative_graph import get_professor_generative_compiled_graph
from app.langgraph.graphs.student_generative_graph import get_student_generative_compiled_graph
from app.langgraph.schemas import Phase12ExecutionRequest
from app.langgraph.state import Phase12GraphState
from app.langgraph.tracing.model_versions import build_phase12_model_versions
from app.mcp.config import mcp_settings
from app.rag.store import build_storage_fields_from_rag_meta
from app.services.report_richness import extract_best_summary
from app.services import report_generation_support as support

logger = logging.getLogger("phase15_16.genai")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _final_status_label(state: Phase12GraphState) -> str:
    final_status = state.final_status.value if state.final_status else state.status.value
    if state.safe_mode and final_status in {"completed", "partial"}:
        return "safe_mode_completed"
    return final_status


def _stored_summary(row: dict[str, Any] | None) -> StoredSummary | None:
    if not isinstance(row, dict):
        return None
    return StoredSummary(
        id=str(row.get("id") or ""),
        file_id=str(row.get("file_id") or ""),
        submission_id=row.get("submission_id"),
        role=str(row.get("role") or ""),
        created_at=row.get("created_at"),
        needs_review=bool(row.get("needs_review", False)),
    )


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, number))


def _safe_confidence_band(score: float | None) -> str | None:
    if score is None:
        return None
    return confidence_band(score)


def _route_label(primary_model: str, fallback_model: str, model_used: str) -> str:
    primary = _as_text(primary_model)
    fallback = _as_text(fallback_model)
    used = _as_text(model_used)
    if primary and fallback and fallback != primary:
        return f"{primary} -> {fallback}"
    return used or primary or fallback


def _model_versions_section(row: dict[str, Any] | None, key: str) -> dict[str, Any]:
    return _as_dict(_as_dict(_as_dict(row).get("model_versions")).get(key))


def _rag_has_meaningful_data(rag_meta: dict[str, Any] | None) -> bool:
    rag = _as_dict(rag_meta)
    if not rag:
        return False

    citations = _as_list(rag.get("citations"))
    retrieved_chunks = _as_list(rag.get("retrieved_chunks"))
    trace = _as_dict(rag.get("trace"))
    score = _as_float(rag.get("confidence_score"))

    return bool(
        rag.get("enabled")
        or citations
        or retrieved_chunks
        or trace
        or bool(rag.get("safe_review"))
        or (score is not None and score > 0.0)
    )


def _rag_check_from_storage_row(
    row: dict[str, Any] | None,
    *,
    empty_summary: str,
) -> AICheckRAG | None:
    if not row:
        return None

    citations = _as_list(row.get("citations"))
    retrieved_chunks = _as_list(row.get("retrieved_chunks"))
    trace = _as_dict(row.get("rag_trace"))
    score = _as_float(row.get("retrieval_confidence"))
    label = _as_text(row.get("retrieval_confidence_label")) or _safe_confidence_band(score) or ""
    enabled = bool(citations or retrieved_chunks or trace or score is not None)
    if not enabled and not bool(row.get("safe_review")):
        return None

    summary = (
        f"{len(citations)} citations across {len(retrieved_chunks)} retrieved chunks."
        if enabled
        else empty_summary
    )
    return AICheckRAG(
        enabled=enabled,
        confidence_score=score,
        confidence_label=label,
        citations_count=len(citations),
        retrieved_chunk_count=len(retrieved_chunks),
        query=_as_text(trace.get("query")),
        collection_name=_as_text(trace.get("collection_name")),
        safe_review=bool(row.get("safe_review")),
        summary=summary,
    )


class GenAIService:
    """Unified generation, explain, compare, and audit service."""

    async def generate_student_report(
        self,
        body: AIReportGenerateIn,
        user: CurrentUser,
    ) -> AIReportResponse:
        return await self._generate(role="student", body=body, user=user)

    async def generate_professor_report(
        self,
        body: AIReportGenerateIn,
        user: CurrentUser,
    ) -> AIReportResponse:
        return await self._generate(role="professor", body=body, user=user)

    async def explain(
        self,
        *,
        file_id: str,
        role: Literal["student", "professor"],
        user: CurrentUser,
    ) -> ExplainResponse:
        row = await self._load_latest_row(file_id=file_id, role=role, user=user)
        phase = self._phase_payload(row)
        return ExplainResponse(file_id=file_id, role=role, explanation=phase["explanation"])

    async def compare(
        self,
        *,
        file_id: str,
        role: Literal["student", "professor"],
        user: CurrentUser,
    ) -> CompareResponse:
        row = await self._load_latest_row(file_id=file_id, role=role, user=user)
        phase = self._phase_payload(row)
        prediction = phase["prediction"]
        report = prediction.report
        confidence = report.confidence if hasattr(report, "confidence") else None
        evidence_count = len(phase["sources"].evidence_references)
        warning_count = len(phase["warnings"])
        consistency_count = len(phase["fairness"].consistency_findings)
        summary = (
            f"{genai_settings.primary_model} vs {genai_settings.validator_model} agreement is "
            f"{phase['fairness'].multi_model_agreement.agreement_band} "
            f"({phase['fairness'].multi_model_agreement.agreement_score:.2f}). "
            f"{evidence_count} evidence references and {warning_count} warnings were present; "
            f"{consistency_count} consistency findings were recorded."
        )
        return CompareResponse(
            file_id=file_id,
            role=role,
            fairness=phase["fairness"],
            comparison_summary=summary,
            confidence_score=float(getattr(confidence, "score", 0.0) or 0.0),
            confidence_band=str(getattr(confidence, "band", "low") or "low"),
            evidence_reference_count=evidence_count,
            warning_count=warning_count,
        )

    async def audit(
        self,
        *,
        file_id: str,
        role: Literal["student", "professor"],
        user: CurrentUser,
        include_pdf: bool = False,
    ) -> AuditResponse:
        row = await self._load_latest_row(file_id=file_id, role=role, user=user)
        phase = self._phase_payload(row)
        audit = phase["audit"]

        if include_pdf and genai_settings.pdf_enabled and not audit.pdf_base64:
            response = AIReportResponse(
                request_id=audit.request_id,
                prediction=phase["prediction"],
                explanation=phase["explanation"],
                sources=phase["sources"],
                warnings=list(phase["warnings"]),
                fairness=phase["fairness"],
                audit=audit,
                stored=_stored_summary(row),
            )
            try:
                audit = audit.model_copy(update={"pdf_base64": build_pdf_base64(response)})
            except Exception as exc:  # pragma: no cover - optional dependency path
                logger.warning("phase15_16 pdf generation failed file_id=%s error=%s", file_id, exc)

        return AuditResponse(file_id=file_id, role=role, audit=audit, warnings=list(phase["warnings"]))

    async def check(
        self,
        *,
        file_id: str,
        role: Literal["student", "professor"],
        user: CurrentUser,
    ) -> AICheckResponse:
        await support.load_file(file_id, user)
        rows = await support.get_rows(
            f"ai_reports?file_id=eq.{file_id}&role=eq.{role}&select=*&order=created_at.desc&limit=12"
        )
        genai_row, graph_row, langchain_row, baseline_row = self._select_check_rows(role, rows)
        phase = self._phase_payload_or_none(genai_row)

        return AICheckResponse(
            file_id=file_id,
            role=role,
            summary=self._build_check_summary(role, genai_row, baseline_row, phase),
            langchain=self._build_langchain_check(role, genai_row, langchain_row, baseline_row),
            langgraph=self._build_langgraph_check(graph_row, phase),
            genai=self._build_genai_check(role, genai_row, baseline_row, phase),
            rag=self._build_rag_check(genai_row, baseline_row),
            ml=self._build_ml_check(genai_row, baseline_row, phase),
            llm=self._build_llm_check(genai_row, baseline_row),
            mcp=self._build_mcp_check(role, graph_row),
            n8n=self._build_n8n_check(role, graph_row),
        )

    async def pdf(
        self,
        *,
        file_id: str,
        role: Literal["student", "professor"],
        user: CurrentUser,
    ) -> tuple[bytes, str]:
        row = await self._load_latest_row(file_id=file_id, role=role, user=user)
        phase = self._phase_payload(row)
        audit = phase["audit"]
        response = AIReportResponse(
            request_id=audit.request_id,
            prediction=phase["prediction"],
            explanation=phase["explanation"],
            sources=phase["sources"],
            warnings=list(phase["warnings"]),
            fairness=phase["fairness"],
            audit=audit,
            stored=_stored_summary(row),
        )
        from app.genai.pdf_service import build_pdf_bytes

        pdf_bytes = build_pdf_bytes(response)
        filename = f"eduaiplatform-{role}-report-{file_id}.pdf"
        return pdf_bytes, filename

    def _select_check_rows(
        self,
        role: Literal["student", "professor"],
        rows: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
        genai_row: dict[str, Any] | None = None
        graph_row: dict[str, Any] | None = None
        langchain_row: dict[str, Any] | None = None
        baseline_row: dict[str, Any] | None = None

        for row in rows:
            pipeline = _as_text(_as_dict(row.get("model_versions")).get("pipeline"))
            normalized = support.normalize_report_row(role, row)
            if genai_row is None and pipeline == genai_settings.pipeline_label:
                genai_row = row
            if graph_row is None and self._has_langgraph_metadata(row):
                graph_row = normalized
            if langchain_row is None and (pipeline == "phase10_langchain" or self._has_langchain_metadata(row)):
                langchain_row = normalized
            if baseline_row is None:
                baseline_row = normalized

        return genai_row, graph_row, langchain_row, baseline_row

    def _phase_payload_or_none(self, row: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(row, dict):
            return None
        try:
            return self._phase_payload(row)
        except Exception as exc:
            logger.warning(
                "phase15_16 check metadata parse failed file_id=%s error=%s",
                row.get("file_id"),
                exc,
            )
            return None

    def _prediction_summary(
        self,
        role: Literal["student", "professor"],
        prediction: PredictionTab | None,
    ) -> str:
        if prediction is None:
            return ""
        report = prediction.report.model_dump(mode="json") if hasattr(prediction.report, "model_dump") else {}
        return _as_text(extract_best_summary(role, report))

    def _phase7_summary(
        self,
        role: Literal["student", "professor"],
        row: dict[str, Any] | None,
    ) -> str:
        report = _as_dict(_as_dict(row).get("report_json"))
        return _as_text(extract_best_summary(role, report))

    def _build_check_summary(
        self,
        role: Literal["student", "professor"],
        genai_row: dict[str, Any] | None,
        baseline_row: dict[str, Any] | None,
        phase: dict[str, Any] | None,
    ) -> AICheckSummary:
        if genai_row and phase:
            prediction = phase["prediction"]
            report = prediction.report
            report_confidence = getattr(report, "confidence", None)
            score = _as_float(getattr(report_confidence, "score", None))
            band = _as_text(getattr(report_confidence, "band", None)) or _safe_confidence_band(score)
            safety = getattr(report, "safety", None)
            return AICheckSummary(
                selected_pipeline=genai_settings.pipeline_label,
                status=_as_text(phase["audit"].final_status) or "completed",
                created_at=genai_row.get("created_at"),
                needs_review=bool(getattr(safety, "needs_review", False) or genai_row.get("needs_review")),
                confidence_score=score,
                confidence_band=band or None,
                report_summary=self._prediction_summary(role, prediction),
            )

        if baseline_row:
            report = _as_dict(baseline_row.get("report_json"))
            model_versions = _as_dict(baseline_row.get("model_versions"))
            stored_genai = _as_dict(model_versions.get("genai"))
            stored_langgraph = _as_dict(model_versions.get("langgraph"))
            model_agreement = _as_dict(report.get("model_agreement"))
            agreement = _as_dict(model_versions.get("agreement"))
            confidence_score = _as_float(_as_dict(report.get("confidence")).get("overall"))
            if confidence_score is None:
                confidence_score = _as_float(model_agreement.get("final_confidence"))
            if confidence_score is None:
                confidence_score = _as_float(agreement.get("final_confidence"))
            return AICheckSummary(
                selected_pipeline=_as_text(model_versions.get("pipeline")) or "phase7",
                status=_as_text(stored_genai.get("final_status"))
                or _as_text(stored_langgraph.get("final_status"))
                or "stored",
                created_at=baseline_row.get("created_at"),
                needs_review=bool(
                    baseline_row.get("needs_review")
                    or _as_dict(report.get("safety")).get("needs_review")
                ),
                confidence_score=confidence_score,
                confidence_band=_safe_confidence_band(confidence_score),
                report_summary=self._phase7_summary(role, baseline_row),
            )

        return AICheckSummary(
            report_summary="No LangGraph or GenAI report has been stored for this file yet.",
        )

    def _build_langchain_check(
        self,
        role: Literal["student", "professor"],
        genai_row: dict[str, Any] | None,
        langchain_row: dict[str, Any] | None,
        baseline_row: dict[str, Any] | None,
    ) -> AICheckLangChain:
        row = next(
            (
                candidate
                for candidate in (langchain_row, genai_row, baseline_row)
                if self._has_langchain_metadata(candidate)
            ),
            None,
        )
        if not row:
            return AICheckLangChain(
                summary="No stored LangChain execution metadata was found for this file yet.",
            )

        model_versions = _as_dict(row.get("model_versions"))
        stored_meta = _as_dict(model_versions.get("langchain"))
        if stored_meta:
            return AICheckLangChain(
                available=bool(stored_meta.get("available", False)),
                pipeline=_as_text(stored_meta.get("pipeline")) or _as_text(model_versions.get("pipeline")),
                chain_name=_as_text(stored_meta.get("chain_name")),
                chain_version=_as_text(stored_meta.get("chain_version")),
                prompt_version=_as_text(stored_meta.get("prompt_version")),
                schema_version=_as_text(stored_meta.get("schema_version")),
                provider=_as_text(stored_meta.get("provider")),
                model_used=_as_text(stored_meta.get("model_used")),
                primary_model=_as_text(stored_meta.get("primary_model")),
                fallback_model=_as_text(stored_meta.get("fallback_model")),
                fallback_used=bool(stored_meta.get("fallback_used", False)),
                execution_mode=_as_text(stored_meta.get("execution_mode")),
                decision_source=_as_text(stored_meta.get("decision_source")),
                discrepancy_flag=stored_meta.get("discrepancy_flag")
                if isinstance(stored_meta.get("discrepancy_flag"), bool)
                else None,
                retrieval_mode=_as_text(stored_meta.get("retrieval_mode")),
                retrieved_chunk_count=int(stored_meta.get("retrieved_chunk_count") or 0),
                confidence_score=_as_float(stored_meta.get("confidence_score")),
                summary=_as_text(stored_meta.get("summary")),
            )

        report = _as_dict(row.get("report_json"))
        retrieval_debug = _as_dict(model_versions.get("retrieval_debug"))
        pipeline = _as_text(model_versions.get("pipeline"))
        chain_name = _as_text(model_versions.get("chain_name"))
        chain_version = _as_text(model_versions.get("chain_version"))
        prompt_version = _as_text(
            model_versions.get("student_prompt_version")
            if role == "student"
            else model_versions.get("professor_prompt_version")
        )
        primary_model = _as_text(model_versions.get("llm_primary"))
        fallback_model = _as_text(model_versions.get("llm_fallback"))
        model_used = _as_text(model_versions.get("llm_model_used"))
        fallback_used = bool(model_versions.get("fallback_used")) or bool(
            model_used and fallback_model and model_used == fallback_model
        )
        execution_mode = _as_text(model_versions.get("execution_mode"))
        if not execution_mode and (fallback_used or bool(retrieval_debug.get("weak_retrieval"))):
            execution_mode = "fallback"
        if not execution_mode:
            execution_mode = "normal"
        confidence_score = _as_float(_as_dict(model_versions.get("agreement")).get("final_confidence"))
        if confidence_score is None:
            confidence_score = _as_float(_as_dict(report.get("confidence")).get("overall"))
        if confidence_score is None:
            confidence_score = _as_float(_as_dict(report.get("model_agreement")).get("final_confidence"))
        discrepancy_flag_raw = model_versions.get("discrepancy_flag")
        discrepancy_flag = discrepancy_flag_raw if isinstance(discrepancy_flag_raw, bool) else None
        retrieved_chunk_count = int(retrieval_debug.get("chunk_count") or 0)

        if pipeline == "phase10_langchain":
            summary = (
                f"Stored LangChain output is available with {retrieved_chunk_count} retrieved chunk(s)."
                if retrieved_chunk_count
                else "Stored LangChain output is available for this file."
            )
        elif pipeline == genai_settings.pipeline_label:
            summary = (
                "LangChain foundation metadata was carried into the stored LangGraph/GenAI run."
            )
        else:
            summary = "LangChain-style execution metadata is available from the stored report."

        return AICheckLangChain(
            available=True,
            pipeline=pipeline,
            chain_name=chain_name,
            chain_version=chain_version,
            prompt_version=prompt_version,
            schema_version=_as_text(model_versions.get("schema_version")),
            provider=_as_text(model_versions.get("provider")),
            model_used=model_used,
            primary_model=primary_model,
            fallback_model=fallback_model,
            fallback_used=fallback_used,
            execution_mode=execution_mode,
            decision_source=_as_text(model_versions.get("decision_source")),
            discrepancy_flag=discrepancy_flag,
            retrieval_mode=_as_text(retrieval_debug.get("mode")),
            retrieved_chunk_count=retrieved_chunk_count,
            confidence_score=confidence_score,
            summary=summary,
        )

    def _has_langchain_metadata(self, row: dict[str, Any] | None) -> bool:
        model_versions = _as_dict(_as_dict(row).get("model_versions"))
        pipeline = _as_text(model_versions.get("pipeline"))
        return bool(
            pipeline == "phase10_langchain"
            or _as_dict(model_versions.get("langchain")).get("available")
            or _as_text(model_versions.get("chain_name"))
            or _as_text(model_versions.get("chain_version"))
            or _as_dict(model_versions.get("retrieval_debug"))
            or _as_dict(model_versions.get("prompt_debug"))
        )

    def _has_langgraph_metadata(self, row: dict[str, Any] | None) -> bool:
        model_versions = _as_dict(_as_dict(row).get("model_versions"))
        pipeline = _as_text(model_versions.get("pipeline"))
        return bool(
            pipeline in {"phase12_langgraph", genai_settings.pipeline_label}
            or _as_dict(model_versions.get("langgraph")).get("available")
            or _as_dict(model_versions.get("graph"))
            or _as_dict(model_versions.get("graph_execution"))
            or _as_dict(model_versions.get("graph_trace"))
        )

    def _build_langgraph_check(
        self,
        graph_row: dict[str, Any] | None,
        phase: dict[str, Any] | None,
    ) -> AICheckLangGraph:
        if not graph_row:
            return AICheckLangGraph()

        model_versions = _as_dict(graph_row.get("model_versions"))
        stored_meta = _as_dict(model_versions.get("langgraph"))
        if stored_meta:
            graph_trace = _as_dict(model_versions.get("graph_trace"))
            trace_nodes = _as_list(graph_trace.get("node_entries"))
            stored_node_count = int(stored_meta.get("node_count") or len(trace_nodes) or 0)
            stored_total_steps = int(stored_meta.get("total_steps") or 0) or stored_node_count
            return AICheckLangGraph(
                available=bool(stored_meta.get("available", False)),
                pipeline=_as_text(stored_meta.get("pipeline")) or _as_text(model_versions.get("pipeline")),
                graph_name=_as_text(stored_meta.get("graph_name")),
                graph_version=_as_text(stored_meta.get("graph_version")) or genai_settings.graph_version,
                prompt_version=_as_text(stored_meta.get("prompt_version")),
                output_version=_as_text(stored_meta.get("output_version")),
                final_status=_as_text(stored_meta.get("final_status")),
                safe_mode=bool(stored_meta.get("safe_mode", False)),
                total_steps=stored_total_steps,
                total_latency_ms=float(stored_meta.get("total_latency_ms") or 0.0),
                node_count=stored_node_count,
                decision_count=int(stored_meta.get("decision_count") or 0),
                failure_count=int(stored_meta.get("failure_count") or 0),
                trace_summary=_as_text(stored_meta.get("trace_summary")),
                warnings=[_as_text(item) for item in _as_list(stored_meta.get("warnings")) if _as_text(item)],
            )

        graph = _as_dict(model_versions.get("graph"))
        execution = _as_dict(model_versions.get("graph_execution"))
        trace = _as_dict(model_versions.get("graph_trace"))
        audit = phase["audit"] if phase else None
        node_entries = _as_list(trace.get("node_entries"))
        decision_entries = _as_list(trace.get("decision_entries"))
        failure_entries = _as_list(trace.get("failure_entries"))
        total_latency = execution.get("total_latency_ms")
        if total_latency is None:
            total_latency = _as_dict(model_versions.get("timings_ms")).get("total") or 0.0

        final_status = _as_text(execution.get("final_status"))
        if not final_status and audit is not None:
            final_status = _as_text(audit.final_status)

        safe_mode = bool(execution.get("safe_mode"))
        if not safe_mode and audit is not None:
            safe_mode = _as_text(audit.final_status).startswith("safe_mode")

        return AICheckLangGraph(
            available=True,
            pipeline=_as_text(model_versions.get("pipeline")),
            graph_name=_as_text(graph.get("graph_name")),
            graph_version=_as_text(graph.get("graph_version")) or genai_settings.graph_version,
            prompt_version=_as_text(graph.get("prompt_version")) or _as_text(getattr(audit, "prompt_version", "")),
            output_version=_as_text(getattr(audit, "output_version", "")),
            final_status=final_status,
            safe_mode=safe_mode,
            total_steps=int(execution.get("total_steps") or len(node_entries)),
            total_latency_ms=float(total_latency or 0.0),
            node_count=len(node_entries),
            decision_count=len(decision_entries),
            failure_count=len(failure_entries),
            trace_summary=_as_text(model_versions.get("graph_trace_summary")) or _as_text(trace.get("summary")),
            warnings=list(phase["warnings"]) if phase else [],
        )

    def _build_genai_check(
        self,
        role: Literal["student", "professor"],
        genai_row: dict[str, Any] | None,
        baseline_row: dict[str, Any] | None,
        phase: dict[str, Any] | None,
    ) -> AICheckGenAI:
        if not genai_row or not phase:
            stored_meta = _model_versions_section(baseline_row, "genai")
            if stored_meta:
                return AICheckGenAI(
                    available=bool(stored_meta.get("available", False)),
                    pipeline=_as_text(stored_meta.get("pipeline")) or _as_text(
                        _as_dict(_as_dict(baseline_row).get("model_versions")).get("pipeline")
                    ),
                    model_version=_as_text(stored_meta.get("model_version")),
                    validator_model_version=_as_text(stored_meta.get("validator_model_version")),
                    final_status=_as_text(stored_meta.get("final_status")),
                    confidence_score=_as_float(stored_meta.get("confidence_score")),
                    confidence_band=_as_text(stored_meta.get("confidence_band")) or None,
                    report_summary=_as_text(stored_meta.get("report_summary")) or self._phase7_summary(role, baseline_row),
                    warning_count=int(stored_meta.get("warning_count") or 0),
                )
            return AICheckGenAI()

        prediction = phase["prediction"]
        report = prediction.report
        report_confidence = getattr(report, "confidence", None)
        score = _as_float(getattr(report_confidence, "score", None))
        band = _as_text(getattr(report_confidence, "band", None)) or _safe_confidence_band(score)

        return AICheckGenAI(
            available=True,
            pipeline=genai_settings.pipeline_label,
            model_version=_as_text(phase["audit"].model_version),
            validator_model_version=_as_text(phase["audit"].validator_model_version),
            final_status=_as_text(phase["audit"].final_status),
            confidence_score=score,
            confidence_band=band or None,
            report_summary=self._prediction_summary(role, prediction),
            warning_count=len(phase["warnings"]),
        )

    def _build_rag_check(
        self,
        genai_row: dict[str, Any] | None,
        baseline_row: dict[str, Any] | None,
    ) -> AICheckRAG:
        rag_meta = _as_dict(_as_dict(genai_row).get("rag_meta"))
        rag_trace = _as_dict(_as_dict(genai_row).get("rag_trace"))
        if _rag_has_meaningful_data(rag_meta):
            citations = _as_list(rag_meta.get("citations"))
            retrieved_chunks = _as_list(rag_meta.get("retrieved_chunks"))
            trace = _as_dict(rag_meta.get("trace")) or rag_trace
            score = _as_float(rag_meta.get("confidence_score"))
            label = _as_text(rag_meta.get("confidence_label")) or _safe_confidence_band(score) or ""
            enabled = bool(rag_meta.get("enabled") or citations or retrieved_chunks or trace)
            summary = (
                f"{len(citations)} citations across {len(retrieved_chunks)} retrieved chunks."
                if enabled
                else "RAG grounding metadata was not stored for this run."
            )
            return AICheckRAG(
                enabled=enabled,
                confidence_score=score,
                confidence_label=label,
                citations_count=len(citations),
                retrieved_chunk_count=len(retrieved_chunks),
                query=_as_text(trace.get("query")),
                collection_name=_as_text(trace.get("collection_name")),
                safe_review=bool(rag_meta.get("safe_review")),
                summary=summary,
            )

        genai_storage_rag = _rag_check_from_storage_row(
            genai_row,
            empty_summary="RAG grounding metadata was not stored for this run.",
        )
        if genai_storage_rag is not None:
            return genai_storage_rag

        baseline_storage_rag = _rag_check_from_storage_row(
            baseline_row,
            empty_summary="No RAG evidence was attached to the stored baseline report.",
        )
        if baseline_storage_rag is not None:
            return baseline_storage_rag

        return AICheckRAG(summary="RAG has not been run or stored for this file yet.")

    def _build_ml_check(
        self,
        genai_row: dict[str, Any] | None,
        baseline_row: dict[str, Any] | None,
        phase: dict[str, Any] | None,
    ) -> AICheckML:
        if baseline_row:
            stored_meta = _model_versions_section(baseline_row, "ml")
            if stored_meta:
                return AICheckML(
                    available=bool(stored_meta.get("available", False)),
                    confidence_score=_as_float(stored_meta.get("confidence_score")),
                    model_names=[_as_text(item) for item in _as_list(stored_meta.get("model_names")) if _as_text(item)],
                    source=_as_text(stored_meta.get("source")) or _as_text(_as_dict(baseline_row.get("model_versions")).get("pipeline")),
                    summary=_as_text(stored_meta.get("summary")),
                )
            report = _as_dict(baseline_row.get("report_json"))
            model_versions = _as_dict(baseline_row.get("model_versions"))
            ml_models = _as_dict(model_versions.get("ml_models"))
            model_names = [
                value
                for raw in ml_models.values()
                if (value := _as_text(raw))
            ]
            ml_score = _as_float(_as_dict(report.get("model_agreement")).get("ml_confidence"))
            if ml_score is None:
                bucket = _as_dict(model_versions.get("agreement")).get("ml_bucket_0_to_4")
                if isinstance(bucket, (int, float)):
                    ml_score = max(0.0, min(1.0, float(bucket) / 4.0))
            summary = (
                f"ML calibration is available from {len(model_names)} stored model signal(s)."
                if model_names
                else "ML calibration is available from the stored baseline report."
            )
            return AICheckML(
                available=True,
                confidence_score=ml_score,
                model_names=model_names,
                source=_as_text(model_versions.get("pipeline")) or "phase7",
                summary=summary,
            )

        if genai_row and phase:
            feature_importance = getattr(phase["explanation"], "feature_importance", [])
            used_ml = any(
                _as_text(getattr(item, "feature", "")).startswith("ml:")
                for item in feature_importance
            )
            return AICheckML(
                available=used_ml,
                source=genai_settings.pipeline_label if used_ml else "",
                summary=(
                    "ML calibration contributed to the stored LangGraph explanation."
                    if used_ml
                    else "No explicit ML calibration metadata was stored for this GenAI run."
                ),
            )

        return AICheckML(summary="ML calibration has not been surfaced for this file yet.")

    def _build_llm_check(
        self,
        genai_row: dict[str, Any] | None,
        baseline_row: dict[str, Any] | None,
    ) -> AICheckLLM:
        row = genai_row or baseline_row
        if not row:
            return AICheckLLM()

        model_versions = _as_dict(row.get("model_versions"))
        stored_meta = _as_dict(model_versions.get("llm"))
        if stored_meta:
            route = _as_text(stored_meta.get("route"))
            return AICheckLLM(
                available=bool(stored_meta.get("available", False) or route),
                model_used=_as_text(stored_meta.get("model_used")),
                primary_model=_as_text(stored_meta.get("primary_model")),
                fallback_model=_as_text(stored_meta.get("fallback_model")),
                route=route,
                source=_as_text(stored_meta.get("source")) or _as_text(model_versions.get("pipeline")) or "stored_report",
            )

        primary_model = _as_text(model_versions.get("llm_primary"))
        fallback_model = _as_text(model_versions.get("llm_fallback"))
        model_used = _as_text(model_versions.get("llm_model_used"))
        route = _route_label(primary_model, fallback_model, model_used)

        return AICheckLLM(
            available=bool(route),
            model_used=model_used,
            primary_model=primary_model,
            fallback_model=fallback_model,
            route=route,
            source=_as_text(model_versions.get("pipeline")) or "stored_report",
        )

    def _build_mcp_check(
        self,
        role: Literal["student", "professor"],
        graph_row: dict[str, Any] | None,
    ) -> AICheckMCP:
        visible_tools: list[str] = []
        if mcp_settings.enabled:
            try:
                import app.mcp  # noqa: F401
                from app.mcp.enums import ToolRole
                from app.mcp.registry import list_tools

                role_enum = ToolRole(role)
                visible_tools = [
                    defn.tool_name
                    for defn in list_tools(include_disabled=False)
                    if role_enum in defn.allowed_roles
                ][:8]
            except Exception as exc:
                logger.warning("phase15_16 mcp tool listing failed role=%s error=%s", role, exc)

        stored_meta = _model_versions_section(graph_row, "mcp")
        trace = _as_dict(_as_dict(_as_dict(graph_row).get("model_versions")).get("graph_trace"))
        node_entries = _as_list(trace.get("node_entries"))
        mcp_steps = [
            entry
            for entry in node_entries
            if _as_text(_as_dict(entry).get("node_name")) == "mcp_tools"
        ]
        graph_used = bool(stored_meta.get("graph_used", False)) or len(mcp_steps) > 0

        if stored_meta:
            return AICheckMCP(
                enabled=bool(stored_meta.get("enabled", mcp_settings.enabled)),
                orchestration_enabled=bool(stored_meta.get("orchestration_enabled", mcp_settings.orchestration_enabled)),
                llm_enabled=bool(stored_meta.get("llm_enabled", mcp_settings.llm_enabled)),
                graph_used=graph_used,
                tool_call_count=max(int(stored_meta.get("tool_call_count") or 0), len(mcp_steps)),
                visible_tools=visible_tools
                or [_as_text(item) for item in _as_list(stored_meta.get("visible_tools")) if _as_text(item)],
                summary=_as_text(stored_meta.get("summary")),
            )

        if not mcp_settings.enabled:
            summary = "MCP is disabled in backend configuration."
        elif graph_used:
            summary = "An MCP bridge node was recorded in the stored LangGraph trace."
        elif visible_tools:
            summary = (
                f"MCP is enabled with {len(visible_tools)} visible tool(s) for this role, "
                "but no MCP step was recorded for this file."
            )
        else:
            summary = "MCP is enabled, but no visible tools were resolved for this role."

        return AICheckMCP(
            enabled=mcp_settings.enabled,
            orchestration_enabled=mcp_settings.orchestration_enabled,
            llm_enabled=mcp_settings.llm_enabled,
            graph_used=graph_used,
            tool_call_count=len(mcp_steps),
            visible_tools=visible_tools,
            summary=summary,
        )

    def _build_n8n_check(
        self,
        role: Literal["student", "professor"],
        graph_row: dict[str, Any] | None,
    ) -> AICheckN8N:
        cfg = get_event_settings()
        trace = _as_dict(_as_dict(_as_dict(graph_row).get("model_versions")).get("graph_trace"))
        node_entries = _as_list(trace.get("node_entries"))
        generation_bridge_active = any(
            _as_text(_as_dict(entry).get("node_name")) == "generation"
            for entry in node_entries
        )

        integrations = AICheckN8NIntegrations(
            assessment=bool(
                (
                    cfg.n8n_webhook_path_assessment_student
                    if role == "student"
                    else cfg.n8n_webhook_path_assessment_professor
                ).strip()
            ),
            file_upload=bool(cfg.n8n_webhook_path_file_upload.strip()),
            low_confidence=bool(cfg.n8n_webhook_path_low_confidence.strip()),
            pipeline_failure=bool(cfg.n8n_webhook_path_pipeline_failure.strip()),
        )
        configured = bool(cfg.n8n_base_url.strip() and cfg.n8n_webhook_hmac_secret.strip())

        if configured and generation_bridge_active and integrations.assessment:
            summary = (
                "The LangGraph generation node is wired to emit an assessment request to n8n "
                "for this role when the webhook bridge is reachable."
            )
        elif configured:
            summary = "n8n webhook bridges are configured, but no stored generation trace was found for this file."
        else:
            summary = "n8n base URL or webhook signing secret is not fully configured."

        return AICheckN8N(
            configured=configured,
            generation_bridge_active=generation_bridge_active,
            integrations=integrations,
            summary=summary,
        )

    async def _generate(
        self,
        *,
        role: Literal["student", "professor"],
        body: AIReportGenerateIn,
        user: CurrentUser,
    ) -> AIReportResponse:
        support.rate_limit(user.id)
        state = await self._prepare_state(role=role, file_id=body.file_id, user=user)
        graph = (
            get_student_generative_compiled_graph()
            if role == "student"
            else get_professor_generative_compiled_graph()
        )
        raw = await graph.ainvoke(state)
        final_state = raw if isinstance(raw, Phase12GraphState) else Phase12GraphState.model_validate(raw)
        response = self._build_response(final_state)
        stored = await self._persist(final_state, response)
        return response.model_copy(update={"stored": stored})

    async def _prepare_state(
        self,
        *,
        role: Literal["student", "professor"],
        file_id: str,
        user: CurrentUser,
    ) -> Phase12GraphState:
        request = Phase12ExecutionRequest(
            file_id=file_id,
            user_id=user.id,
            role=role,
            correlation_id=str(uuid.uuid4()),
            submission_id="",
            user_email=user.email,
            raw_claims=user.raw_claims,
            file_metadata={},
        )
        state = Phase12GraphState.create(request)
        state.graph_version = genai_settings.graph_version
        state.pipeline_context.execution_meta.pipeline = genai_settings.pipeline_label
        state.pipeline_context.execution_meta.student_prompt_version = genai_settings.prompt_version
        state.pipeline_context.execution_meta.professor_prompt_version = genai_settings.prompt_version
        state.pipeline_context.execution_meta.chain_version = genai_settings.graph_version
        state.pipeline_context.execution_meta.schema_version = genai_settings.schema_version
        state.pipeline_context.execution_meta.primary_model = genai_settings.primary_model
        state.pipeline_context.execution_meta.fallback_model = genai_settings.validator_model
        state.pipeline_context.execution_meta.decision_source = DecisionSource.HYBRID

        file_row = await support.load_file(file_id, user)
        ingestion_dict = await support.build_ingestion_bundle(file_id, user)
        ingestion_bundle = IngestionBundle.model_validate(ingestion_dict)

        state.file_row = file_row
        state.pipeline_context.ingestion = ingestion_bundle
        state.pipeline_context.submission_id = str(file_row.get("submission_id") or "")
        state.media_metadata = {
            "mime_type": file_row.get("mime_type"),
            "status": file_row.get("status"),
        }

        submission_form = support.classify_submission_form(ingestion_dict)
        submission_kind = support.detect_submission_kind(ingestion_dict)
        state.apply_submission_kind(submission_kind)
        state.input_hash = support.sha256_json(
            {
                "file_id": file_id,
                "submission_id": state.pipeline_context.submission_id,
                "submission_kind": submission_kind,
                "submission_form": submission_form,
                "ingestion": ingestion_bundle.model_dump(mode="json"),
            }
        )

        try:
            if role == "student":
                ml_raw = await support.call_ai_student_multimodal(user, ingestion_dict)
                ml_result = normalize_student_ml_context(ml_raw)
            else:
                ml_raw = await support.call_ai_professor_multimodal(user, ingestion_dict)
                ml_result = normalize_professor_ml_context(ml_raw)
            state.pipeline_context.ml_raw = ml_raw
            state.pipeline_context.ml_result = ml_result
            state.pipeline_context.ml_context_text = ml_result.context_text
        except Exception as exc:
            logger.warning("phase15_16 ml preparation failed file_id=%s role=%s error=%s", file_id, role, exc)
            state.add_warning(
                "ML calibration was unavailable, so the generation path used a more conservative explanation."
            )

        try:
            rag_seed = {
                "submission_id": state.pipeline_context.submission_id,
                "ingestion": ingestion_bundle.model_dump(mode="json"),
                "ml": state.pipeline_context.ml_result.model_dump(mode="json")
                if state.pipeline_context.ml_result
                else {},
                "analysis_type": state.pipeline_context.analysis_type.value,
                "submission_type": state.pipeline_context.submission_kind,
                "mode": submission_form,
                "submission_form": submission_form,
            }
            _, rag_context = (
                pack_student_rag(rag_seed)
                if role == "student"
                else pack_professor_rag(rag_seed)
            )
            state.pipeline_context.rag = rag_context
            state.pipeline_context.execution_meta.retrieval_debug = rag_context.trace
        except Exception as exc:
            logger.warning("phase15_16 rag preparation failed file_id=%s role=%s error=%s", file_id, role, exc)
            state.add_warning(
                "RAG grounding was unavailable, so only submission evidence and ML signals were used."
            )

        return state

    def _build_response(self, state: Phase12GraphState) -> AIReportResponse:
        report_model = (
            StudentReport.model_validate(state.final_report)
            if state.role == "student"
            else ProfessorModerationReport.model_validate(state.final_report)
        )
        counterfactual = str(report_model.counterfactual_explanation or "")
        explanation = build_explanation_tab(
            state,
            hybrid_summary=(
                "The report combines submission evidence, Phase 6 ML calibration signals, "
                "and retrieved knowledge-base guidance."
            ),
            counterfactual=counterfactual,
        )
        sources = build_sources_tab(state)
        consistency = detect_report_contradictions(report_model.model_dump(mode="json"))
        validator_confidence = 0.0
        if isinstance(state.critique_output, dict):
            try:
                validator_confidence = float(state.critique_output.get("confidence_score") or 0.0)
            except (TypeError, ValueError):
                validator_confidence = 0.0
        fairness = build_fairness_tab(
            report=report_model.model_dump(mode="json"),
            counterfactual=counterfactual,
            consistency_findings=consistency,
            primary_model=genai_settings.primary_model,
            validator_model=genai_settings.validator_model,
            primary_confidence=state.final_confidence,
            validator_confidence=validator_confidence,
        )
        audit = AuditTab(
            request_id=state.pipeline_context.request_id,
            execution_id=state.execution_id,
            role=state.role,
            graph_version=genai_settings.graph_version,
            prompt_version=genai_settings.prompt_version,
            output_version=genai_settings.output_version,
            model_version=state.model_version or genai_settings.primary_model,
            validator_model_version=genai_settings.validator_model,
            timestamp=_utc_now(),
            final_status=_final_status_label(state),
        )
        return AIReportResponse(
            request_id=state.pipeline_context.request_id,
            prediction=PredictionTab(report_type=state.role, report=report_model),
            explanation=explanation,
            sources=sources,
            warnings=list(state.warnings),
            fairness=fairness,
            audit=audit,
        )

    async def _persist(self, state: Phase12GraphState, response: AIReportResponse) -> StoredSummary | None:
        model_versions = build_phase12_model_versions(state)
        model_versions["pipeline"] = genai_settings.pipeline_label
        model_versions["phase15_16"] = {
            "output_version": genai_settings.output_version,
            "prediction": response.prediction.model_dump(mode="json"),
            "explanation": response.explanation.model_dump(mode="json"),
            "sources": response.sources.model_dump(mode="json"),
            "warnings": list(response.warnings),
            "fairness": response.fairness.model_dump(mode="json"),
            "audit": response.audit.model_dump(mode="json", exclude={"pdf_base64"}),
        }

        state.pipeline_context.report = response.prediction.report.model_dump(mode="json")
        state.pipeline_context.validation_result = ValidationResult.ok(repaired=bool(state.repaired_report))
        file_id = support.uuid_or_none(state.file_id)
        submission_id = support.uuid_or_none(state.submission_id)
        row: dict[str, Any] = {
            "file_id": file_id,
            "submission_id": submission_id,
            "role": state.role,
            "report_json": state.pipeline_context.report,
            "report_hash": support.sha256_json(state.pipeline_context.report),
            "prompt_hash": state.prompt_hash or support.sha256_json({"prompt_version": genai_settings.prompt_version}),
            "input_hash": state.input_hash,
            "model_versions": model_versions,
            "needs_review": bool(response.prediction.report.safety.needs_review),
        }
        rag = state.pipeline_context.rag.model_dump(mode="json") if state.pipeline_context.rag else {}
        row.update(build_storage_fields_from_rag_meta(rag))
        stored = await support.post_row("ai_reports", row)
        return _stored_summary(stored)

    async def _load_latest_row(
        self,
        *,
        file_id: str,
        role: Literal["student", "professor"],
        user: CurrentUser,
    ) -> dict[str, Any]:
        await support.load_file(file_id, user)
        rows = await support.get_rows(
            f"ai_reports?file_id=eq.{file_id}&role=eq.{role}&select=*&order=created_at.desc&limit=10"
        )
        for row in rows:
            pipeline = ((row.get("model_versions") or {}).get("pipeline")) or ""
            if pipeline == genai_settings.pipeline_label:
                return row
        raise HTTPException(status_code=404, detail="No Phase 15/16 report found for this file.")

    def _phase_payload(self, row: dict[str, Any]) -> dict[str, Any]:
        phase = ((row.get("model_versions") or {}).get("phase15_16")) or {}
        if not phase:
            raise HTTPException(status_code=404, detail="Stored Phase 15/16 metadata is missing.")
        return {
            "prediction": PredictionTab.model_validate(phase.get("prediction") or {}),
            "explanation": ExplanationTab.model_validate(phase.get("explanation") or {}),
            "sources": SourcesTab.model_validate(phase.get("sources") or {}),
            "warnings": list(phase.get("warnings") or []),
            "fairness": FairnessTab.model_validate(phase.get("fairness") or {}),
            "audit": AuditTab.model_validate(phase.get("audit") or {}),
        }
