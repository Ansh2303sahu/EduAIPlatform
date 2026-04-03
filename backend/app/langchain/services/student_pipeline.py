"""
Student generation pipeline for Phase 10.

This service keeps the existing Phase 7-compatible return shape while adding
the richer orchestration needed by the LangChain-based Phase 10 flow:

- sanitize raw submission text
- normalize Phase 6 ML signals
- package student retrieval context
- choose normal vs safe prompting
- invoke the llm-service-backed LangChain chain
- repair / parse / validate / normalize JSON
- emit execution metadata and storage-safe payloads
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Mapping, Sequence

from app.langchain.config import phase10_settings
from app.langchain.enums import (
    AnalysisType,
    ChainRole,
    DecisionSource,
    ExecutionMode,
    PipelineMode,
    PipelineRole,
    PipelineStage,
    execution_mode_from_pipeline_mode,
)
from app.langchain.models import (
    ExecutionMetadata,
    IngestionBundle,
    PipelineContext,
    RagContext,
    ValidationResult,
)
from app.langchain.parsers.json_repair import try_parse_json
from app.langchain.parsers.normalizer import normalize_student_payload
from app.langchain.parsers.validators import validate_student_payload
from app.langchain.prompts.student import build_student_prompt, build_student_safe_prompt
from app.langchain.services.chain_factory import build_generation_chain, get_primary_model
from app.langchain.services.execution_logger import Phase10ExecutionLogger
from app.langchain.services.fallback_service import (
    build_student_fallback_payload,
    run_repair_with_fallback,
    run_with_fallback,
)
from app.langchain.services.ml_context_builder import normalize_student_ml_context
from app.langchain.services.prompt_sanitizer import sanitize_input
from app.langchain.services.retrieval_packager import (
    pack_student_rag,
    package_retrieval_chunks,
    package_retrieval_payload,
)
from app.rag.store import build_storage_fields_from_rag_meta

logger = logging.getLogger("phase10.student_pipeline")


class StudentPipeline:
    """
    Full student report generation pipeline.

    The service is independently callable and accepts direct ingestion, Phase 6
    ML output, and optional Phase 9 retrieval artifacts without depending on
    the FastAPI router layer.
    """

    async def run(
        self,
        *,
        file_id: str,
        submission_id: str,
        ingestion_dict: dict[str, Any],
        ml_dict: dict[str, Any],
        submission_kind: str = "academic",
        user_id: str = "",
        file_metadata: Mapping[str, Any] | None = None,
        rag_chunks: Sequence[Mapping[str, Any]] | None = None,
        rag_payload: Mapping[str, Any] | None = None,
        rag_citations: Sequence[Mapping[str, Any]] | None = None,
        extra_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute the full student pipeline and return a Phase 7-compatible result
        envelope plus richer execution metadata for direct callers.
        """
        request_id = str(uuid.uuid4())
        t_all = time.perf_counter()
        extra = dict(extra_context or {})

        sanitized_ingestion, injected, injection_reason = _sanitize_ingestion(ingestion_dict)

        ctx = PipelineContext(
            request_id=request_id,
            file_id=file_id,
            submission_id=submission_id,
            role=PipelineRole.STUDENT,
            analysis_type=_analysis_type_for_submission(submission_kind),
            mode=PipelineMode.NORMAL,
            execution_mode=ExecutionMode.NORMAL,
            submission_kind=submission_kind,
            ingestion=IngestionBundle(**sanitized_ingestion),
            injection_detected=injected,
            injection_reason=injection_reason,
            ml_raw=dict(ml_dict or {}),
        )

        execution_logger = Phase10ExecutionLogger(
            request_id,
            "student",
            file_id=file_id,
            user_id=str(user_id or ""),
            execution_mode=ExecutionMode.NORMAL,
            decision_source=DecisionSource.LLM,
            weak_retrieval=False,
            low_confidence=False,
        )

        t_ml = time.perf_counter()
        ctx.ml_result = normalize_student_ml_context(ml_dict)
        ctx.ml_context_text = ctx.ml_result.context_text
        ctx.timings_ms["ml"] = int((time.perf_counter() - t_ml) * 1000)

        low_confidence = _is_low_confidence(ctx)
        ctx.mode = _initial_mode(injected=injected, low_confidence=low_confidence)
        ctx.execution_mode = execution_mode_from_pipeline_mode(ctx.mode)

        t_rag = time.perf_counter()
        ctx.rag = self._package_retrieval(
            ctx,
            rag_payload=rag_payload,
            rag_chunks=rag_chunks,
            rag_citations=rag_citations,
            extra_context=extra,
        )
        ctx.timings_ms["rag"] = int((time.perf_counter() - t_rag) * 1000)

        retrieval_weak = _is_retrieval_weak(ctx.rag)
        if phase10_settings.enable_safe_mode and (retrieval_weak or low_confidence or injected):
            ctx.mode = PipelineMode.RESTRICTED
            ctx.execution_mode = execution_mode_from_pipeline_mode(ctx.mode)

        execution_logger.set_flags(
            weak_retrieval=retrieval_weak,
            low_confidence=low_confidence,
        )

        if retrieval_weak and low_confidence:
            logger.info(
                "phase10 student hard fallback request_id=%s weak_retrieval=%s low_confidence=%s",
                request_id,
                retrieval_weak,
                low_confidence,
            )
            return await self._finalize_fallback_result(
                ctx,
                execution_logger,
                reason="weak_retrieval",
                detail="Phase 6 confidence was also low, so generation was skipped.",
                stage=PipelineStage.RETRIEVE,
                t_all=t_all,
                file_metadata=file_metadata,
            )

        t_prompt = time.perf_counter()
        ctx.prompt_text = self._build_prompt(ctx)
        ctx.timings_ms["prompt"] = int((time.perf_counter() - t_prompt) * 1000)

        t_gen = time.perf_counter()
        raw_text, model_used, model_error = await self._generate(ctx, execution_logger)
        ctx.timings_ms["llm"] = int((time.perf_counter() - t_gen) * 1000)
        ctx.raw_llm_output = raw_text
        ctx.model_used = model_used
        ctx.fallback_used = bool(
            execution_logger.record.fallback_triggered
            or model_used == phase10_settings.llm_fallback_label
        )
        if ctx.fallback_used and not model_error:
            ctx.execution_mode = ExecutionMode.FALLBACK

        if model_error:
            return await self._finalize_fallback_result(
                ctx,
                execution_logger,
                reason="ollama_unavailable",
                detail=model_error,
                stage=PipelineStage.GENERATE,
                t_all=t_all,
                file_metadata=file_metadata,
            )

        t_parse = time.perf_counter()
        report_dict, validation_result, llm_ok, fallback_reason = await self._parse_validate_normalize(
            ctx,
            execution_logger,
        )
        ctx.timings_ms["parse_validate"] = int((time.perf_counter() - t_parse) * 1000)
        ctx.report = report_dict
        ctx.validation_result = validation_result
        ctx.llm_ok = llm_ok

        if fallback_reason:
            return await self._finalize_fallback_result(
                ctx,
                execution_logger,
                reason=fallback_reason,
                detail="The generated output could not be safely validated.",
                stage=validation_result.stage,
                t_all=t_all,
                file_metadata=file_metadata,
                preserve_timings=True,
            )

        _apply_student_safety(
            ctx,
            injected=injected,
            retrieval_weak=retrieval_weak,
            low_confidence=low_confidence,
        )
        _penalize_empty_output(ctx)
        student_discrepancy_flag = _run_student_disagreement_check(ctx)
        _apply_report_quality_gate(ctx)

        ctx.rag_meta = _build_rag_meta(ctx.rag)
        ctx.timings_ms["total"] = int((time.perf_counter() - t_all) * 1000)

        decision_source = _determine_decision_source(ctx)
        execution_logger.set_model_used(ctx.model_used)
        execution_logger.set_execution_mode(ctx.execution_mode)
        execution_logger.set_decision_source(decision_source)

        ctx.execution_meta = _build_execution_meta(ctx, decision_source=decision_source)
        execution_payload = await execution_logger.persist(
            duration_ms=ctx.timings_ms["total"],
            raw_output=ctx.raw_llm_output,
        )

        logger.info(
            "phase10 student_pipeline complete request_id=%s model=%s mode=%s "
            "injected=%s retrieval_weak=%s llm_ok=%s discrepancy=%s timings_ms=%s",
            request_id,
            ctx.model_used,
            ctx.mode.value,
            injected,
            retrieval_weak,
            llm_ok,
            student_discrepancy_flag,
            ctx.timings_ms,
        )

        result = _build_result_payload(
            ctx,
            execution_payload=execution_payload,
            file_metadata=file_metadata,
        )
        result["discrepancy_flag"] = student_discrepancy_flag
        return result

    async def _generate(
        self,
        ctx: PipelineContext,
        execution_logger: Phase10ExecutionLogger,
    ) -> tuple[str, str, str]:
        primary_model = get_primary_model(role="student")
        primary_chain = build_generation_chain(primary_model)
        try:
            raw, model_label = await run_with_fallback(
                primary_chain,
                {"prompt_text": ctx.prompt_text},
                role="student",
                request_id=ctx.request_id,
                execution_logger=execution_logger,
            )
            if model_label == phase10_settings.llm_fallback_label:
                execution_logger.set_execution_mode(ExecutionMode.FALLBACK)
            return raw, model_label, ""
        except RuntimeError as exc:
            logger.error(
                "phase10 student generation failed request_id=%s error=%s",
                ctx.request_id,
                exc,
            )
            return "", "", str(exc)

    async def _parse_validate_normalize(
        self,
        ctx: PipelineContext,
        execution_logger: Phase10ExecutionLogger,
    ) -> tuple[dict[str, Any], ValidationResult, bool, str]:
        raw_text = ctx.raw_llm_output
        repaired = False
        last_parse_error = ""

        for attempt in range(phase10_settings.llm_max_retries + 1):
            parsed, err = try_parse_json(raw_text)
            if parsed is not None:
                raw_validation = validate_student_payload(parsed)
                if not raw_validation.valid:
                    execution_logger.mark_validation_failure("; ".join(raw_validation.errors[:2]))

                normalized = normalize_student_payload(parsed)
                normalized_dump = normalized.model_dump(exclude={"rag_meta"})
                final_validation = validate_student_payload(normalized_dump)

                if final_validation.valid:
                    warnings = list(raw_validation.errors)
                    if warnings:
                        warnings.append(
                            "Missing or malformed student fields were normalized with safe defaults."
                        )
                    validation = ValidationResult(
                        valid=True,
                        warnings=warnings,
                        stage=PipelineStage.VALIDATE,
                        repaired=bool(repaired or not raw_validation.valid),
                    )
                    return normalized_dump, validation, True, ""

                execution_logger.mark_validation_failure("; ".join(final_validation.errors[:2]))
                validation = ValidationResult(
                    valid=False,
                    errors=final_validation.errors,
                    warnings=raw_validation.errors,
                    stage=PipelineStage.VALIDATE,
                    repaired=bool(repaired or not raw_validation.valid),
                )
                return normalized_dump, validation, False, "validation_failure"

            last_parse_error = err or "Unable to parse student JSON output."
            execution_logger.mark_parse_failure(last_parse_error)

            if (
                attempt < phase10_settings.llm_max_retries
                and phase10_settings.enable_json_repair
            ):
                target = (
                    "student_project_review"
                    if ctx.submission_kind == "project"
                    else "student"
                )
                raw_text = await run_repair_with_fallback(
                    raw_text,
                    target,
                    request_id=ctx.request_id,
                    execution_logger=execution_logger,
                )
                repaired = True
                continue

            break

        validation = ValidationResult(
            valid=False,
            errors=[last_parse_error or "The student output could not be parsed."],
            stage=PipelineStage.PARSE,
            repaired=repaired,
        )
        return {}, validation, False, "malformed_output"

    def _package_retrieval(
        self,
        ctx: PipelineContext,
        *,
        rag_payload: Mapping[str, Any] | None,
        rag_chunks: Sequence[Mapping[str, Any]] | None,
        rag_citations: Sequence[Mapping[str, Any]] | None,
        extra_context: Mapping[str, Any],
    ) -> RagContext:
        base_payload = self._build_base_llm_payload(ctx)

        resolved_payload = _first_mapping(
            rag_payload,
            extra_context.get("rag_payload"),
            extra_context.get("phase9_rag_payload"),
        )
        if resolved_payload:
            packaged = package_retrieval_payload(resolved_payload, role="student")
            return RagContext.model_validate(packaged)

        resolved_chunks = _first_sequence(
            rag_chunks,
            extra_context.get("rag_chunks"),
            extra_context.get("retrieved_chunks"),
            extra_context.get("phase9_chunks"),
        )
        if resolved_chunks:
            packaged = package_retrieval_chunks(
                role="student",
                retrieved_chunks=resolved_chunks,
                citations=_first_sequence(
                    rag_citations,
                    extra_context.get("rag_citations"),
                    extra_context.get("citations"),
                ),
                confidence_score=_coerce_float(
                    extra_context.get("rag_confidence_score"),
                    default=0.0,
                ),
                confidence_label=str(
                    extra_context.get("rag_confidence_label") or "low"
                ),
                safe_review=bool(extra_context.get("rag_safe_review", False)),
                trace=_first_mapping(extra_context.get("rag_trace")),
                instruction=str(extra_context.get("rag_instruction") or ""),
                context_text=str(extra_context.get("rag_context_text") or ""),
            )
            return RagContext.model_validate(packaged)

        _, rag_ctx = pack_student_rag(base_payload)
        return rag_ctx

    def _build_base_llm_payload(self, ctx: PipelineContext) -> dict[str, Any]:
        return {
            "submission_id": ctx.submission_id,
            "ingestion": ctx.ingestion.model_dump(),
            "ml": ctx.ml_raw,
            "query": _derive_query(ctx),
            "top_k": 6 if ctx.submission_kind == "project" else 5,
            "mode": ctx.mode.value,
            "analysis_type": ctx.analysis_type.value,
            "analysis_focus": _analysis_focus(ctx.submission_kind),
            "submission_type": ctx.submission_kind,
            "safety_flags": {
                "injection_detected": ctx.injection_detected,
                "low_confidence": _is_low_confidence(ctx),
            },
        }

    @staticmethod
    def _build_prompt(ctx: PipelineContext) -> str:
        if ctx.mode == PipelineMode.RESTRICTED:
            return build_student_safe_prompt(ctx)
        return build_student_prompt(ctx)

    async def _finalize_fallback_result(
        self,
        ctx: PipelineContext,
        execution_logger: Phase10ExecutionLogger,
        *,
        reason: str,
        detail: str,
        stage: PipelineStage,
        t_all: float,
        file_metadata: Mapping[str, Any] | None,
        preserve_timings: bool = False,
    ) -> dict[str, Any]:
        ctx.fallback_used = True
        ctx.llm_ok = False
        ctx.mode = PipelineMode.RESTRICTED
        ctx.execution_mode = ExecutionMode.FALLBACK
        ctx.rag_meta = _build_rag_meta(ctx.rag)

        native_fallback = build_student_fallback_payload(
            reason,
            detail=detail,
            citations=ctx.rag.citations,
        )
        ctx.report = _student_report_from_native_fallback(ctx, native_fallback, reason=reason, detail=detail)
        ctx.validation_result = ValidationResult(
            valid=False,
            errors=[detail or reason],
            stage=stage,
            repaired=preserve_timings and bool(ctx.raw_llm_output),
        )
        ctx.timings_ms["total"] = int((time.perf_counter() - t_all) * 1000)

        execution_logger.mark_fallback_triggered(reason)
        execution_logger.set_execution_mode(ExecutionMode.FALLBACK)
        execution_logger.set_decision_source(DecisionSource.FALLBACK)

        ctx.execution_meta = _build_execution_meta(
            ctx,
            decision_source=DecisionSource.FALLBACK,
        )
        execution_payload = await execution_logger.persist(
            duration_ms=ctx.timings_ms["total"],
            raw_output=ctx.raw_llm_output,
        )

        logger.warning(
            "phase10 student fallback request_id=%s reason=%s stage=%s detail=%s",
            ctx.request_id,
            reason,
            stage.value,
            detail,
        )

        return _build_result_payload(
            ctx,
            execution_payload=execution_payload,
            file_metadata=file_metadata,
        )


def _sanitize_ingestion(
    ingestion_dict: Mapping[str, Any],
) -> tuple[dict[str, Any], bool, str]:
    sanitized_text, text_injected, text_reason = sanitize_input(
        str(ingestion_dict.get("text_content") or "")
    )
    sanitized_ocr, ocr_injected, ocr_reason = sanitize_input(
        str(ingestion_dict.get("ocr_text") or "")
    )
    sanitized_audio, audio_injected, audio_reason = sanitize_input(
        str(ingestion_dict.get("audio_transcript") or "")
    )

    sanitized = dict(ingestion_dict)
    sanitized["text_content"] = sanitized_text
    sanitized["ocr_text"] = sanitized_ocr
    sanitized["audio_transcript"] = sanitized_audio

    injected = bool(text_injected or ocr_injected or audio_injected)
    injection_reason = next(
        (reason for reason in (text_reason, ocr_reason, audio_reason) if reason),
        "",
    )
    return sanitized, injected, injection_reason


def _analysis_type_for_submission(submission_kind: str) -> AnalysisType:
    return (
        AnalysisType.STUDENT_PROJECT
        if submission_kind == "project"
        else AnalysisType.STUDENT_ACADEMIC
    )


def _initial_mode(*, injected: bool, low_confidence: bool) -> PipelineMode:
    return PipelineMode.RESTRICTED if (injected or low_confidence) else PipelineMode.NORMAL


def _analysis_focus(submission_kind: str) -> list[str]:
    if submission_kind == "project":
        return [
            "project aim",
            "technical stack",
            "architecture",
            "implementation quality",
            "security",
            "testing",
            "limitations",
        ]
    return [
        "task response",
        "structure",
        "evidence",
        "critical analysis",
        "clarity",
        "referencing",
        "academic quality",
    ]


def _derive_query(ctx: PipelineContext) -> str:
    text = (ctx.ingestion.text_content or "")[:500]
    prefix = "project" if ctx.submission_kind == "project" else "academic"
    return f"{prefix} student submission review: {text[:200]}"


def _is_low_confidence(ctx: PipelineContext) -> bool:
    bucket = _coerce_int(
        ctx.ml_raw.get("confidence_0_to_4"),
        default=int(round((ctx.ml_result.confidence_score if ctx.ml_result else 0.5) * 4)),
    )
    score = ctx.ml_result.confidence_score if ctx.ml_result is not None else bucket / 4.0
    return bucket <= 1 or score <= 0.35


def _is_retrieval_weak(rag: RagContext) -> bool:
    if _bool_attr(rag, "safe_review") or _bool_attr(rag, "weak_retrieval"):
        return True
    if not _bool_attr(rag, "enabled"):
        return False
    return str(getattr(rag, "confidence_label", "low")).lower() == "low"


def _apply_student_safety(
    ctx: PipelineContext,
    *,
    injected: bool,
    retrieval_weak: bool,
    low_confidence: bool,
) -> None:
    reasons: list[str] = []
    if injected:
        reasons.append("potential prompt-injection content was detected")
    if retrieval_weak:
        reasons.append("grounding evidence was limited")
    if low_confidence:
        reasons.append("Phase 6 confidence was low")
    if ctx.validation_result.warnings:
        reasons.append("some generated fields were normalized conservatively")

    needs_review = bool(
        injected
        or retrieval_weak
        or low_confidence
        or ctx.report.get("safety", {}).get("needs_review", False)
    )
    ctx.report.setdefault("safety", {})["needs_review"] = needs_review

    if needs_review and not str(ctx.report["safety"].get("reason") or "").strip():
        if reasons:
            ctx.report["safety"]["reason"] = (
                "Manual review recommended because " + ", ".join(reasons) + "."
            )
        else:
            ctx.report["safety"]["reason"] = "Manual review recommended."


def _penalize_empty_output(ctx: PipelineContext) -> None:
    """Deflate over-inflated LLM confidence when the generated report is mostly empty."""
    report = ctx.report
    summary = str(report.get("summary") or "").strip()
    issues = [x for x in (report.get("issues") or []) if x]
    strengths = [x for x in (report.get("strengths") or []) if x]
    improvement = [x for x in (report.get("improvement_plan") or []) if x]
    checklist = [x for x in (report.get("checklist") or []) if x]

    filled_sections = sum([
        len(summary) > 30,
        len(issues) >= 1,
        len(strengths) >= 1,
        len(improvement) >= 1,
        len(checklist) >= 1,
    ])

    if filled_sections >= 3:
        return  # report is substantive — no penalty needed

    agreement = report.setdefault("model_agreement", {})
    confidence = report.setdefault("confidence", {})
    current_final = float(agreement.get("final_confidence") or 0.0)
    current_overall = float(confidence.get("overall") or 0.0)

    cap = 0.35 if filled_sections == 2 else (0.20 if filled_sections == 1 else 0.10)
    if current_final > cap:
        agreement["final_confidence"] = cap
        agreement["llm_confidence"] = min(float(agreement.get("llm_confidence") or 0.0), cap)
    if current_overall > cap:
        confidence["overall"] = cap

    if filled_sections <= 1:
        ctx.report.setdefault("safety", {})["needs_review"] = True
        if not str(ctx.report["safety"].get("reason") or "").strip():
            ctx.report["safety"]["reason"] = (
                "Generated report has insufficient content for reliable assessment."
            )


def _run_student_disagreement_check(ctx: PipelineContext) -> bool:
    """
    Detect ML/LLM confidence disagreement for student reports.

    When Phase 6 ML predicts high confidence but the LLM self-reports low
    confidence (or vice versa), flag for review and downgrade final_confidence
    to a hybrid midpoint.  Returns True if material disagreement was found.
    """
    ml_bucket = _coerce_int(
        ctx.ml_raw.get("confidence_0_to_4"),
        default=int(round((ctx.ml_result.confidence_score if ctx.ml_result else 0.5) * 4)),
    )
    ml_conf = ml_bucket / 4.0

    agreement = ctx.report.get("model_agreement") or {}
    llm_conf = float(
        agreement.get("final_confidence") or agreement.get("llm_confidence") or 0.0
    )

    gap = abs(ml_conf - llm_conf)
    if gap <= 0.4:
        return False

    ctx.report.setdefault("safety", {})["needs_review"] = True
    if not str(ctx.report["safety"].get("reason") or "").strip():
        ctx.report["safety"]["reason"] = (
            f"Manual review recommended: ML confidence ({ml_conf:.2f}) and "
            f"LLM confidence ({llm_conf:.2f}) differ materially."
        )

    # Downgrade to a hybrid midpoint, cap at 0.5 to avoid false High band
    hybrid_conf = round(min(0.5, (ml_conf + llm_conf) / 2.0), 3)
    ctx.report.setdefault("model_agreement", {})["final_confidence"] = hybrid_conf
    return True


def _apply_report_quality_gate(ctx: PipelineContext) -> None:
    """
    Reject high-confidence empty reports before storage.

    If the student report has no substantive content, force needs_review and
    floor final_confidence so a High band is never awarded to an empty output.
    """
    report = ctx.report
    summary = str(report.get("summary") or "").strip()
    issues = [x for x in (report.get("issues") or []) if x]
    improvement = [x for x in (report.get("improvement_plan") or []) if x]
    checklist = [x for x in (report.get("checklist") or []) if x]

    has_content = (
        len(summary) > 50
        or len(issues) >= 1
        or len(improvement) >= 1
        or len(checklist) >= 1
    )
    if has_content:
        return

    ctx.report.setdefault("safety", {})["needs_review"] = True
    agreement = ctx.report.setdefault("model_agreement", {})
    if float(agreement.get("final_confidence") or 0.0) > 0.15:
        agreement["final_confidence"] = 0.15
    ctx.fallback_used = True


def _build_rag_meta(rag: RagContext) -> dict[str, Any]:
    return {
        "enabled": _bool_attr(rag, "enabled"),
        "confidence_score": _coerce_float(getattr(rag, "confidence_score", 0.0), default=0.0),
        "confidence_label": str(getattr(rag, "confidence_label", "low")),
        "safe_review": _bool_attr(rag, "safe_review"),
        "chunk_count": _coerce_int(getattr(rag, "chunk_count", 0), default=0),
        "weak_retrieval": _bool_attr(rag, "weak_retrieval"),
        "citations": list(getattr(rag, "citations", []) or []),
        "retrieved_chunks": list(getattr(rag, "retrieved_chunks", []) or []),
        "trace": dict(getattr(rag, "trace", {}) or {}),
    }


def _student_report_from_native_fallback(
    ctx: PipelineContext,
    native_payload: Any,
    *,
    reason: str,
    detail: str,
) -> dict[str, Any]:
    ml_confidence = ctx.ml_result.confidence_score if ctx.ml_result is not None else 0.0
    final_confidence = _coerce_float(getattr(native_payload, "confidence", 0.0), default=0.0)
    llm_confidence = 0.0
    safety_reason = detail.strip() or _fallback_reason_text(reason)

    strength_rows = [
        {
            "title": value[:200],
            "evidence": "A conservative fallback response was used, so detailed supporting evidence was not generated.",
        }
        for value in list(getattr(native_payload, "strengths", []) or [])
        if str(value).strip()
    ]
    issue_rows = [
        {
            "title": getattr(issue, "title", "Manual review recommended"),
            "evidence": getattr(issue, "description", safety_reason),
            "severity": getattr(issue, "severity", "high"),
        }
        for issue in list(getattr(native_payload, "issues", []) or [])
    ] or [
        {
            "title": "Manual review recommended",
            "evidence": safety_reason,
            "severity": "high",
        }
    ]
    improvement_rows = [
        {
            "action": value[:300],
            "why": "The fallback response kept recommendations conservative.",
            "how": "Use manual review or regenerate once evidence quality improves.",
            "priority": min(index + 1, 10),
        }
        for index, value in enumerate(list(getattr(native_payload, "improvement_plan", []) or []))
        if str(value).strip()
    ]
    checklist_rows = [
        {
            "item": getattr(item, "item", "Manual review completed"),
            "done": bool(getattr(item, "done", False)),
        }
        for item in list(getattr(native_payload, "checklist", []) or [])
    ]

    return {
        "summary": getattr(
            native_payload,
            "summary",
            "Automated student feedback was limited and a conservative fallback response was returned.",
        ),
        "issues": issue_rows,
        "strengths": strength_rows,
        "architecture_review": {
            "overview": "Not assessed.",
            "backend": "Not assessed.",
            "frontend": "Not assessed.",
            "database": "Not assessed.",
            "security": "Not assessed.",
        },
        "implementation_review": {
            "features_built": [],
            "technical_quality": "Not assessed.",
            "integration_quality": "Not assessed.",
        },
        "evaluation_review": {
            "testing_present": "Not assessed.",
            "limitations": "Evidence was too limited for a reliable automated review.",
            "academic_quality": "Not assessed.",
        },
        "improvement_plan": improvement_rows,
        "checklist": checklist_rows,
        "confidence": {
            "mode": "restricted",
            "overall": final_confidence,
        },
        "model_agreement": {
            "ml_confidence": ml_confidence,
            "llm_confidence": llm_confidence,
            "final_confidence": final_confidence,
        },
        "safety": {
            "needs_review": True,
            "reason": safety_reason,
        },
    }


def _fallback_reason_text(reason: str) -> str:
    messages = {
        "ollama_unavailable": "The model call failed, so the pipeline returned a conservative fallback response.",
        "malformed_output": "The generated output could not be parsed into reliable JSON.",
        "validation_failure": "The generated output could not be validated safely.",
        "weak_retrieval": "Retrieved evidence was too weak for a reliable grounded response.",
        "low_confidence": "Phase 6 confidence was too low for a trustworthy automated review.",
        "internal_error": "An internal pipeline error prevented a complete automated response.",
    }
    return messages.get(reason, messages["internal_error"])


def _determine_decision_source(ctx: PipelineContext) -> DecisionSource:
    if ctx.fallback_used or not ctx.llm_ok:
        return DecisionSource.FALLBACK
    if ctx.ml_result is None:
        return DecisionSource.LLM
    has_ml_signals = bool(
        ctx.ml_result.predicted_label
        or ctx.ml_result.predicted_band
        or ctx.ml_result.confidence_score > 0.0
    )
    if has_ml_signals:
        return DecisionSource.HYBRID
    return DecisionSource.LLM


def _build_execution_meta(
    ctx: PipelineContext,
    *,
    decision_source: DecisionSource,
) -> ExecutionMetadata:
    agreement_score = _coerce_float(
        ((ctx.report.get("model_agreement") or {}).get("final_confidence")),
        default=_coerce_float(((ctx.report.get("confidence") or {}).get("overall")), default=0.0),
    )
    return ExecutionMetadata(
        request_id=ctx.request_id,
        provider=phase10_settings.provider,
        primary_model=phase10_settings.llm_primary_label,
        fallback_model=phase10_settings.llm_fallback_label,
        model_used=ctx.model_used,
        fallback_used=ctx.fallback_used,
        execution_mode=ctx.execution_mode,
        decision_source=decision_source,
        role=ChainRole.STUDENT,
        chain_version=phase10_settings.chain_version,
        student_prompt_version=phase10_settings.student_prompt_version,
        professor_prompt_version=phase10_settings.professor_prompt_version,
        schema_version=phase10_settings.schema_version,
        timings_ms=dict(ctx.timings_ms),
        agreement_score=agreement_score,
    )


def _build_result_payload(
    ctx: PipelineContext,
    *,
    execution_payload: Mapping[str, Any],
    file_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    storage_fields = build_storage_fields_from_rag_meta(ctx.rag_meta)
    storage_payload = {
        "file_id": ctx.file_id,
        "submission_id": ctx.submission_id,
        "role": ctx.role.value,
        "report_json": ctx.report,
        "needs_review": bool((ctx.report.get("safety") or {}).get("needs_review", False)),
        "model_versions": ctx.execution_meta.to_model_versions_dict(),
        **storage_fields,
    }

    return {
        "request_id": ctx.request_id,
        "report": ctx.report,
        "rag_meta": ctx.rag_meta,
        "ml": ctx.ml_raw,
        "model_used": ctx.model_used,
        "llm_ok": ctx.llm_ok,
        "mode": ctx.mode.value,
        "storage_fields": storage_fields,
        "storage_payload": storage_payload,
        "timings_ms": ctx.timings_ms,
        "validation_status": ctx.validation_result.model_dump(),
        "execution_metadata": dict(execution_payload),
        "execution_mode": ctx.execution_mode.value,
        "decision_source": _determine_decision_source(ctx).value,
        "fallback_used": ctx.fallback_used,
        "file_metadata": dict(file_metadata or {}),
    }


def _first_mapping(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _first_sequence(*values: Any) -> list[Mapping[str, Any]]:
    for value in values:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            items = [item for item in value if isinstance(item, Mapping)]
            if items:
                return [dict(item) for item in items]
    return []


def _bool_attr(obj: Any, name: str) -> bool:
    value = getattr(obj, name, False)
    return value if isinstance(value, bool) else False


def _coerce_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default
