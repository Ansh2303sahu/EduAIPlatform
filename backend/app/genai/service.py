"""Phase 15/16 service orchestration."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import HTTPException

from app.core.deps import CurrentUser
from app.genai.config import genai_settings
from app.genai.consistency import detect_report_contradictions
from app.genai.explainability import build_explanation_tab, build_sources_tab
from app.genai.fairness import build_fairness_tab
from app.genai.pdf_service import build_pdf_base64
from app.genai.schemas import (
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
        state.pipeline_context.execution_meta.decision_source = DecisionSource.HYBRID.value

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

        submission_kind = support.detect_submission_kind(ingestion_dict)
        state.apply_submission_kind(submission_kind)
        state.input_hash = support.sha256_json(
            {
                "file_id": file_id,
                "submission_id": state.pipeline_context.submission_id,
                "submission_kind": submission_kind,
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
                "mode": state.pipeline_context.submission_kind,
                "query": (
                    "student writing feedback evidence referencing"
                    if role == "student" and submission_kind != "project"
                    else "student architecture implementation testing security"
                    if role == "student"
                    else "professor rubric moderation consistency policy"
                ),
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
        row: dict[str, Any] = {
            "file_id": state.file_id,
            "submission_id": state.submission_id,
            "role": state.role,
            "report_json": state.pipeline_context.report,
            "report_hash": support.sha256_json(state.pipeline_context.report),
            "prompt_hash": state.prompt_hash or support.sha256_json({"prompt_version": genai_settings.prompt_version}),
            "input_hash": state.input_hash,
            "model_versions": model_versions,
            "needs_review": bool(response.prediction.report.safety.needs_review),
        }
        rag = state.pipeline_context.rag.model_dump(mode="json") if state.pipeline_context.rag else {}
        if rag:
            row["rag_meta"] = rag
            row["rag_trace"] = rag.get("trace") or {}
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
