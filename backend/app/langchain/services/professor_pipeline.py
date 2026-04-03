"""
Professor generation pipeline for Phase 10.

Professor moderation logic remains separate from student feedback logic. This
pipeline keeps the existing Phase 7-compatible response shape while adding the
full LangChain orchestration, disagreement checks, and execution metadata
expected by Phase 10.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import Counter
from typing import Any, Mapping, Sequence

from app.core.config import settings
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
from app.langchain.parsers.normalizer import normalize_professor_payload
from app.langchain.parsers.validators import validate_professor_payload
from app.langchain.prompts.professor import (
    build_professor_prompt,
    build_professor_safe_prompt,
)
from app.langchain.services.chain_factory import build_generation_chain, get_primary_model
from app.langchain.services.execution_logger import Phase10ExecutionLogger
from app.langchain.services.fallback_service import (
    build_professor_fallback_payload,
    run_repair_with_fallback,
    run_with_fallback,
)
from app.langchain.services.ml_context_builder import normalize_professor_ml_context
from app.langchain.services.prompt_sanitizer import sanitize_input
from app.langchain.services.retrieval_packager import (
    pack_professor_rag,
    package_retrieval_chunks,
    package_retrieval_payload,
)
from app.rag.store import build_storage_fields_from_rag_meta

logger = logging.getLogger("phase10.professor_pipeline")

_PROFESSOR_BAND_RANKS = {
    "needs review": -1,
    "ineffective": 0,
    "poor": 0,
    "fail": 0,
    "adequate": 1,
    "pass": 1,
    "satisfactory": 1,
    "good": 2,
    "merit": 2,
    "effective": 2,
    "very good": 3,
    "distinction": 3,
    "excellent": 3,
}


class ProfessorPipeline:
    """
    Full professor report generation pipeline.

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
        Execute the full professor pipeline and return a Phase 7-compatible
        result envelope plus richer execution metadata for direct callers.
        """
        request_id = str(uuid.uuid4())
        t_all = time.perf_counter()
        extra = dict(extra_context or {})

        sanitized_ingestion, injected, injection_reason = _sanitize_ingestion(ingestion_dict)

        ctx = PipelineContext(
            request_id=request_id,
            file_id=file_id,
            submission_id=submission_id,
            role=PipelineRole.PROFESSOR,
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
            "professor",
            file_id=file_id,
            user_id=str(user_id or ""),
            execution_mode=ExecutionMode.NORMAL,
            decision_source=DecisionSource.LLM,
            weak_retrieval=False,
            low_confidence=False,
        )

        t_ml = time.perf_counter()
        ctx.ml_result = normalize_professor_ml_context(ml_dict)
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
                "phase10 professor hard fallback request_id=%s weak_retrieval=%s low_confidence=%s",
                request_id,
                retrieval_weak,
                low_confidence,
            )
            return await self._finalize_fallback_result(
                ctx,
                execution_logger,
                reason="weak_retrieval",
                detail="Retrieved evidence was weak and professor moderation confidence was low.",
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
                detail="The generated professor output could not be safely validated.",
                stage=validation_result.stage,
                t_all=t_all,
                file_metadata=file_metadata,
                preserve_timings=True,
            )

        moderation_checks = _run_professor_moderation_checks(
            ctx,
            injected=injected,
            retrieval_weak=retrieval_weak,
            low_confidence=low_confidence,
        )
        _apply_professor_quality_gate(ctx)
        ctx.rag_meta = _build_rag_meta(ctx.rag)
        ctx.timings_ms["moderation"] = moderation_checks["timing_ms"]
        ctx.timings_ms["total"] = int((time.perf_counter() - t_all) * 1000)

        decision_source = _determine_decision_source(ctx)
        execution_logger.set_model_used(ctx.model_used)
        execution_logger.set_execution_mode(ctx.execution_mode)
        execution_logger.set_decision_source(decision_source)

        ctx.execution_meta = _build_execution_meta(
            ctx,
            decision_source=decision_source,
            agreement_score=_coerce_float(moderation_checks["agreement_score"], default=0.0),
        )
        execution_payload = await execution_logger.persist(
            duration_ms=ctx.timings_ms["total"],
            raw_output=ctx.raw_llm_output,
        )

        logger.info(
            "phase10 professor_pipeline complete request_id=%s model=%s mode=%s "
            "injected=%s retrieval_weak=%s llm_ok=%s discrepancy=%s timings_ms=%s",
            request_id,
            ctx.model_used,
            ctx.mode.value,
            injected,
            retrieval_weak,
            llm_ok,
            moderation_checks["discrepancy_flag"],
            ctx.timings_ms,
        )

        return _build_result_payload(
            ctx,
            execution_payload=execution_payload,
            moderation_checks=moderation_checks,
            file_metadata=file_metadata,
        )

    async def _generate(
        self,
        ctx: PipelineContext,
        execution_logger: Phase10ExecutionLogger,
    ) -> tuple[str, str, str]:
        primary_model = get_primary_model(role="professor")
        primary_chain = build_generation_chain(primary_model)
        try:
            raw, model_label = await run_with_fallback(
                primary_chain,
                {"prompt_text": ctx.prompt_text},
                role="professor",
                request_id=ctx.request_id,
                execution_logger=execution_logger,
            )
            if model_label == phase10_settings.llm_fallback_label:
                execution_logger.set_execution_mode(ExecutionMode.FALLBACK)
            return raw, model_label, ""
        except RuntimeError as exc:
            logger.error(
                "phase10 professor generation failed request_id=%s error=%s",
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
                raw_validation = validate_professor_payload(parsed)
                if not raw_validation.valid:
                    execution_logger.mark_validation_failure("; ".join(raw_validation.errors[:2]))

                normalized = normalize_professor_payload(parsed)
                normalized_dump = normalized.model_dump(exclude={"rag_meta"})
                final_validation = validate_professor_payload(normalized_dump)

                if final_validation.valid:
                    warnings = list(raw_validation.errors)
                    if warnings:
                        warnings.append(
                            "Missing or malformed professor fields were normalized with safe defaults."
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

            last_parse_error = err or "Unable to parse professor JSON output."
            execution_logger.mark_parse_failure(last_parse_error)

            if (
                attempt < phase10_settings.llm_max_retries
                and phase10_settings.enable_json_repair
            ):
                raw_text = await run_repair_with_fallback(
                    raw_text,
                    "professor",
                    request_id=ctx.request_id,
                    execution_logger=execution_logger,
                )
                repaired = True
                continue

            break

        validation = ValidationResult(
            valid=False,
            errors=[last_parse_error or "The professor output could not be parsed."],
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
            packaged = package_retrieval_payload(resolved_payload, role="professor")
            return RagContext.model_validate(packaged)

        resolved_chunks = _first_sequence(
            rag_chunks,
            extra_context.get("rag_chunks"),
            extra_context.get("retrieved_chunks"),
            extra_context.get("phase9_chunks"),
        )
        if resolved_chunks:
            packaged = package_retrieval_chunks(
                role="professor",
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

        _, rag_ctx = pack_professor_rag(base_payload)
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
            "official_only": (
                True
                if (ctx.submission_kind == "academic" or settings.rag_require_official_for_professor)
                else None
            ),
            "safety_flags": {
                "injection_detected": ctx.injection_detected,
                "low_confidence": _is_low_confidence(ctx),
            },
        }

    @staticmethod
    def _build_prompt(ctx: PipelineContext) -> str:
        if ctx.mode == PipelineMode.RESTRICTED:
            return build_professor_safe_prompt(ctx)
        return build_professor_prompt(ctx)

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

        native_fallback = build_professor_fallback_payload(
            reason,
            detail=detail,
            citations=ctx.rag.citations,
        )
        ctx.report = _professor_report_from_native_fallback(
            native_fallback,
            reason=reason,
            detail=detail,
        )
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
            agreement_score=0.0,
        )
        execution_payload = await execution_logger.persist(
            duration_ms=ctx.timings_ms["total"],
            raw_output=ctx.raw_llm_output,
        )

        moderation_checks = {
            "discrepancy_flag": bool(getattr(native_fallback, "discrepancy_flag", False)),
            "high_disagreement": bool(getattr(native_fallback, "discrepancy_flag", False)),
            "needs_review": True,
            "llm_band": "needs review",
            "ml_band": _normalize_band_label(str(ctx.ml_raw.get("rubric_band") or "")),
            "band_gap": 0,
            "agreement_score": 0.0,
            "timing_ms": 0,
            "notes": list(getattr(native_fallback, "moderation_notes", []) or []),
        }

        logger.warning(
            "phase10 professor fallback request_id=%s reason=%s stage=%s detail=%s",
            ctx.request_id,
            reason,
            stage.value,
            detail,
        )

        return _build_result_payload(
            ctx,
            execution_payload=execution_payload,
            moderation_checks=moderation_checks,
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
        AnalysisType.PROFESSOR_PROJECT
        if submission_kind == "project"
        else AnalysisType.PROFESSOR_ACADEMIC
    )


def _initial_mode(*, injected: bool, low_confidence: bool) -> PipelineMode:
    return PipelineMode.RESTRICTED if (injected or low_confidence) else PipelineMode.NORMAL


def _analysis_focus(submission_kind: str) -> list[str]:
    if submission_kind == "project":
        return [
            "project scope",
            "architecture",
            "technical implementation",
            "security",
            "testing",
            "limitations",
            "moderation risk",
        ]
    return [
        "structure",
        "argument quality",
        "evidence use",
        "critical analysis",
        "clarity",
        "referencing",
        "moderation risk",
    ]


def _derive_query(ctx: PipelineContext) -> str:
    text = (ctx.ingestion.text_content or "")[:500]
    prefix = "project" if ctx.submission_kind == "project" else "academic"
    return f"{prefix} professor moderation review: {text[:200]}"


def _is_low_confidence(ctx: PipelineContext) -> bool:
    consistency = str(ctx.ml_raw.get("moderation_consistency", "")).strip().lower()
    confidence_score = ctx.ml_result.confidence_score if ctx.ml_result is not None else 0.6
    disagreement_markers = set(ctx.ml_result.disagreement_markers if ctx.ml_result else [])
    return bool(
        consistency == "low"
        or confidence_score <= 0.4
        or "phase6_uncertain" in disagreement_markers
    )


def _is_retrieval_weak(rag: RagContext) -> bool:
    if _bool_attr(rag, "safe_review") or _bool_attr(rag, "weak_retrieval"):
        return True
    if not _bool_attr(rag, "enabled"):
        return False
    return str(getattr(rag, "confidence_label", "low")).lower() == "low"


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


def _run_professor_moderation_checks(
    ctx: PipelineContext,
    *,
    injected: bool,
    retrieval_weak: bool,
    low_confidence: bool,
) -> dict[str, Any]:
    t0 = time.perf_counter()

    report = ctx.report
    ml_band = _normalize_band_label(
        str(
            (ctx.ml_result.predicted_band if ctx.ml_result is not None else "")
            or ctx.ml_raw.get("rubric_band")
            or ""
        )
    )
    llm_band = _derive_llm_professor_band(report)
    band_gap = _band_gap(ml_band, llm_band)
    disagreement_markers = set(ctx.ml_result.disagreement_markers if ctx.ml_result else [])
    moderation_consistency = str(ctx.ml_raw.get("moderation_consistency", "med")).strip().lower()

    high_disagreement = bool(
        band_gap >= 2
        or (
            band_gap >= 1
            and (
                moderation_consistency == "low"
                or disagreement_markers
            )
        )
    )
    discrepancy_flag = high_disagreement

    moderation_notes = _normalize_moderation_notes(report.get("moderation_notes"))
    appended_notes: list[dict[str, str]] = []
    review_reasons: list[str] = []

    if discrepancy_flag:
        review_reasons.append("ML and LLM band signals diverged materially")
        appended_notes.append(
            {
                "risk": "ML/LLM discrepancy",
                "note": (
                    f"Phase 6 suggested {ml_band or 'unknown'} while the generated rubric summary "
                    f"implied {llm_band or 'unknown'}."
                ),
            }
        )
    if retrieval_weak and low_confidence:
        review_reasons.append("retrieval evidence was weak and moderation confidence was low")
        appended_notes.append(
            {
                "risk": "Weak evidence base",
                "note": "Retrieved context was limited while Phase 6 moderation confidence was low.",
            }
        )
    if injected:
        review_reasons.append("prompt-injection content was detected upstream")
        appended_notes.append(
            {
                "risk": "Prompt-injection risk",
                "note": "Embedded instructions in the submission were ignored during moderation.",
            }
        )
    if ctx.validation_result.warnings:
        review_reasons.append("some fields were normalized conservatively")
        appended_notes.append(
            {
                "risk": "Normalization applied",
                "note": "The professor payload needed safe normalization before storage.",
            }
        )

    report["moderation_notes"] = _merge_moderation_notes(moderation_notes, appended_notes)

    needs_review = bool(
        discrepancy_flag
        or (retrieval_weak and low_confidence)
        or injected
        or ctx.fallback_used
        or not ctx.validation_result.valid
        or report.get("safety", {}).get("needs_review", False)
    )
    report.setdefault("safety", {})["needs_review"] = needs_review
    if needs_review and not str(report["safety"].get("reason") or "").strip():
        if review_reasons:
            report["safety"]["reason"] = (
                "Manual review recommended because " + ", ".join(review_reasons) + "."
            )
        else:
            report["safety"]["reason"] = "Manual review recommended."

    agreement_score = _agreement_score_for_professor(
        band_gap=band_gap,
        retrieval_weak=retrieval_weak,
        low_confidence=low_confidence,
        discrepancy_flag=discrepancy_flag,
        llm_ok=ctx.llm_ok,
        ml_confidence=(ctx.ml_result.confidence_score if ctx.ml_result is not None else 0.6),
    )

    return {
        "discrepancy_flag": discrepancy_flag,
        "high_disagreement": high_disagreement,
        "needs_review": needs_review,
        "llm_band": llm_band,
        "ml_band": ml_band,
        "band_gap": band_gap,
        "agreement_score": agreement_score,
        "timing_ms": int((time.perf_counter() - t0) * 1000),
        "notes": [note["note"] for note in appended_notes],
    }


def _normalize_moderation_notes(value: Any) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    if isinstance(value, list):
        for index, item in enumerate(value):
            if isinstance(item, dict):
                risk = str(item.get("risk") or item.get("title") or f"Risk {index + 1}").strip()
                note = str(item.get("note") or item.get("details") or item.get("description") or "").strip()
                if risk or note:
                    normalized.append(
                        {
                            "risk": risk or f"Risk {index + 1}",
                            "note": note or "A moderation note was not provided.",
                        }
                    )
            elif isinstance(item, str) and item.strip():
                normalized.append({"risk": f"Risk {index + 1}", "note": item.strip()})
    return normalized


def _merge_moderation_notes(
    existing: list[dict[str, str]],
    extra: list[dict[str, str]],
) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in [*existing, *extra]:
        risk = str(item.get("risk") or "").strip()
        note = str(item.get("note") or "").strip()
        key = (risk.lower(), note.lower())
        if not risk or not note or key in seen:
            continue
        seen.add(key)
        merged.append({"risk": risk, "note": note})
    return merged


def _normalize_band_label(value: str) -> str:
    label = str(value or "").strip().lower()
    if not label:
        return ""
    aliases = {
        "needsreview": "needs review",
        "need review": "needs review",
        "dist": "distinction",
    }
    return aliases.get(label.replace("-", " "), label.replace("-", " "))


def _band_rank(label: str) -> int | None:
    normalized = _normalize_band_label(label)
    if normalized == "":
        return None
    return _PROFESSOR_BAND_RANKS.get(normalized)


def _apply_professor_quality_gate(ctx: PipelineContext) -> None:
    """
    Reject high-confidence empty professor reports before storage.

    If rubric_breakdown is empty and feedback_explanation is trivially short,
    force needs_review and lower confidence so the result is not stored with a
    falsely confident disposition.
    """
    report = ctx.report
    rubric = [x for x in (report.get("rubric_breakdown") or []) if x]
    explanation = str(report.get("feedback_explanation") or "").strip()

    has_content = len(rubric) >= 1 or len(explanation) > 50
    if has_content:
        return

    ctx.report.setdefault("safety", {})["needs_review"] = True
    if not str(ctx.report["safety"].get("reason") or "").strip():
        ctx.report["safety"]["reason"] = (
            "Generated professor report has insufficient rubric or feedback content for reliable assessment."
        )
    ctx.fallback_used = True


def _band_gap(ml_band: str, llm_band: str) -> int:
    ml_rank = _band_rank(ml_band)
    llm_rank = _band_rank(llm_band)
    if ml_rank is None or llm_rank is None:
        return 0
    return abs(ml_rank - llm_rank)


def _derive_llm_professor_band(report: Mapping[str, Any]) -> str:
    rows = report.get("rubric_breakdown")
    if not isinstance(rows, list) or not rows:
        return ""

    for row in rows:
        if isinstance(row, Mapping):
            criterion = str(row.get("criterion") or "").strip().lower()
            if "overall" in criterion:
                return _normalize_band_label(str(row.get("band") or ""))

    labels = [
        _normalize_band_label(str(row.get("band") or ""))
        for row in rows
        if isinstance(row, Mapping) and str(row.get("band") or "").strip()
    ]
    labels = [label for label in labels if label]
    if not labels:
        return ""

    counts = Counter(labels)
    return max(labels, key=lambda item: (counts[item], _band_rank(item) or -99, -labels.index(item)))


def _agreement_score_for_professor(
    *,
    band_gap: int,
    retrieval_weak: bool,
    low_confidence: bool,
    discrepancy_flag: bool,
    llm_ok: bool,
    ml_confidence: float,
) -> float:
    score = 0.8 if llm_ok else 0.25
    score = min(score, max(0.0, ml_confidence))
    score -= min(0.45, band_gap * 0.2)
    if retrieval_weak:
        score -= 0.15
    if low_confidence:
        score -= 0.15
    if discrepancy_flag:
        score -= 0.15
    return round(max(0.0, min(1.0, score)), 3)


def _professor_report_from_native_fallback(
    native_payload: Any,
    *,
    reason: str,
    detail: str,
) -> dict[str, Any]:
    safety_reason = detail.strip() or _fallback_reason_text(reason)

    rubric_rows = [
        {
            "criterion": getattr(row, "criterion", "Overall academic quality"),
            "band": getattr(row, "band", "Needs review"),
            "justification": getattr(row, "justification", safety_reason),
        }
        for row in list(getattr(native_payload, "rubric_breakdown", []) or [])
    ] or [
        {
            "criterion": "Overall academic quality",
            "band": "Needs review",
            "justification": safety_reason,
        }
    ]

    moderation_rows = [
        {
            "risk": f"Risk {index + 1}",
            "note": str(item).strip(),
        }
        for index, item in enumerate(list(getattr(native_payload, "moderation_notes", []) or []))
        if str(item).strip()
    ] or [
        {
            "risk": "Pipeline fallback",
            "note": safety_reason,
        }
    ]

    feedback_lines = [
        str(item).strip()
        for item in list(getattr(native_payload, "feedback_reasoning", []) or [])
        if str(item).strip()
    ]
    feedback_explanation = " ".join(feedback_lines).strip() or (
        "Automated moderation output was limited and a conservative fallback response was returned."
    )

    return {
        "rubric_breakdown": rubric_rows,
        "feedback_explanation": feedback_explanation,
        "moderation_notes": moderation_rows,
        "safety": {
            "needs_review": True,
            "reason": safety_reason,
        },
    }


def _fallback_reason_text(reason: str) -> str:
    messages = {
        "ollama_unavailable": "The model call failed, so the pipeline returned a conservative moderation fallback response.",
        "malformed_output": "The generated output could not be parsed into reliable JSON.",
        "validation_failure": "The generated output could not be validated safely.",
        "weak_retrieval": "Retrieved evidence was too weak for a reliable moderation judgment.",
        "low_confidence": "Phase 6 moderation confidence was too low for a trustworthy automated evaluation.",
        "internal_error": "An internal pipeline error prevented a complete automated moderation response.",
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
    agreement_score: float,
) -> ExecutionMetadata:
    return ExecutionMetadata(
        request_id=ctx.request_id,
        provider=phase10_settings.provider,
        primary_model=phase10_settings.llm_primary_label,
        fallback_model=phase10_settings.llm_fallback_label,
        model_used=ctx.model_used,
        fallback_used=ctx.fallback_used,
        execution_mode=ctx.execution_mode,
        decision_source=decision_source,
        role=ChainRole.PROFESSOR,
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
    moderation_checks: Mapping[str, Any],
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
        "discrepancy_flag": bool(moderation_checks.get("discrepancy_flag", False)),
        "moderation_checks": dict(moderation_checks),
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
