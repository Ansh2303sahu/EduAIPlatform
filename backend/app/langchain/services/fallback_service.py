"""
Fallback and retry service for Phase 10.

This module does two things:

1. It provides conservative, schema-valid fallback payload builders for the
   Phase 10 native student/professor schemas.
2. It preserves the existing generation / JSON-repair retry helpers used by
   the current LangChain pipelines.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

from langchain_core.runnables import Runnable

from app.langchain.config import phase10_settings
from app.langchain.enums import ExecutionMode
from app.langchain.parsers.json_repair import build_repair_prompt, try_parse_json
from app.langchain.schemas import (
    CitationOut,
    IssueOut,
    Phase10ProfessorOut,
    Phase10StudentOut,
    RubricBandOut,
    StudentChecklistItem,
)
from app.langchain.services.chain_factory import (
    build_generation_chain,
    build_repair_chain,
    get_fallback_model,
)

if TYPE_CHECKING:
    from app.langchain.services.execution_logger import Phase10ExecutionLogger

logger = logging.getLogger("phase10.fallback")

FallbackReason = Literal[
    "ollama_unavailable",
    "malformed_output",
    "validation_failure",
    "weak_retrieval",
    "low_confidence",
    "internal_error",
]

_DEFAULT_REASON: FallbackReason = "internal_error"

_REASON_MESSAGES: dict[FallbackReason, str] = {
    "ollama_unavailable": (
        "The automated generation service was unavailable, so a conservative "
        "fallback response was returned instead of speculative feedback."
    ),
    "malformed_output": (
        "The model returned output that could not be parsed reliably into the "
        "expected schema."
    ),
    "validation_failure": (
        "The generated response did not satisfy the expected schema after "
        "normalization and validation."
    ),
    "weak_retrieval": (
        "Retrieved evidence was too sparse or too weak to support a reliable "
        "grounded response."
    ),
    "low_confidence": (
        "Upstream confidence signals were too low for a trustworthy automated "
        "assessment."
    ),
    "internal_error": (
        "An internal processing error prevented a complete automated response."
    ),
}

_REASON_CONFIDENCE: dict[FallbackReason, float] = {
    "ollama_unavailable": 0.1,
    "malformed_output": 0.08,
    "validation_failure": 0.08,
    "weak_retrieval": 0.12,
    "low_confidence": 0.1,
    "internal_error": 0.05,
}


def _normalize_reason(reason: str | None) -> FallbackReason:
    candidate = str(reason or "").strip().lower()
    if candidate in _REASON_MESSAGES:
        return candidate  # type: ignore[return-value]
    return _DEFAULT_REASON


def _reason_message(reason: FallbackReason, detail: str = "") -> str:
    base = _REASON_MESSAGES[reason]
    suffix = str(detail or "").strip()
    return f"{base} {suffix}".strip() if suffix else base


def _coerce_citations(citations: Sequence[Mapping[str, Any]] | None) -> list[CitationOut]:
    items: list[CitationOut] = []
    for item in citations or []:
        try:
            items.append(
                CitationOut.model_validate(
                    {
                        "source": str(item.get("source") or item.get("title") or ""),
                        "snippet": str(item.get("snippet") or item.get("section") or ""),
                        "relevance": float(item.get("relevance") or item.get("score") or 0.0),
                    }
                )
            )
        except Exception:
            continue
    return items


def build_student_fallback_payload(
    reason: FallbackReason | str,
    *,
    detail: str = "",
    citations: Sequence[Mapping[str, Any]] | None = None,
) -> Phase10StudentOut:
    """
    Build a conservative Phase 10-native student fallback payload.
    """
    normalized_reason = _normalize_reason(reason)
    message = _reason_message(normalized_reason, detail)
    confidence = _REASON_CONFIDENCE[normalized_reason]

    return Phase10StudentOut(
        summary=(
            "Automated student feedback was limited, so the platform returned a "
            "safe fallback response for manual follow-up."
        ),
        strengths=[],
        issues=[
            IssueOut(
                title="Manual review recommended",
                description=message,
                severity="high" if normalized_reason != "weak_retrieval" else "med",
            )
        ],
        improvement_plan=[
            "Review the original submission manually before relying on the automated result.",
            "Regenerate after improving evidence quality or resolving the underlying system issue.",
        ],
        checklist=[
            StudentChecklistItem(item="Manual review completed", done=False),
            StudentChecklistItem(item="Fallback trigger reviewed", done=False),
        ],
        confidence=confidence,
        citations=_coerce_citations(citations),
        safe_mode=True,
        fallback_used=True,
    )


def build_professor_fallback_payload(
    reason: FallbackReason | str,
    *,
    detail: str = "",
    citations: Sequence[Mapping[str, Any]] | None = None,
) -> Phase10ProfessorOut:
    """
    Build a conservative Phase 10-native professor fallback payload.
    """
    normalized_reason = _normalize_reason(reason)
    message = _reason_message(normalized_reason, detail)
    confidence = _REASON_CONFIDENCE[normalized_reason]

    return Phase10ProfessorOut(
        rubric_breakdown=[
            RubricBandOut(
                criterion="Overall academic quality",
                band="Needs review",
                justification=message,
            )
        ],
        feedback_reasoning=[
            "The platform returned a conservative fallback response rather than a full professor evaluation.",
            message,
        ],
        moderation_notes=[
            "Manual moderation is recommended before any band judgement is relied upon.",
            "Fallback mode was triggered, so this output should be treated as provisional only.",
        ],
        needs_review=True,
        confidence=confidence,
        citations=_coerce_citations(citations),
        safe_mode=True,
        fallback_used=True,
        discrepancy_flag=normalized_reason in {"malformed_output", "validation_failure", "low_confidence"},
    )


def build_fallback_payload(
    role: str,
    reason: FallbackReason | str,
    *,
    detail: str = "",
    citations: Sequence[Mapping[str, Any]] | None = None,
) -> Phase10StudentOut | Phase10ProfessorOut:
    """
    Role-aware wrapper around the student/professor fallback builders.
    """
    if str(role).strip().lower() == "professor":
        return build_professor_fallback_payload(reason, detail=detail, citations=citations)
    return build_student_fallback_payload(reason, detail=detail, citations=citations)


def _infer_chain_failure_reason(exc: BaseException) -> FallbackReason:
    message = str(exc).lower()
    if phase10_settings.provider.lower() == "ollama":
        if any(token in message for token in ("connection refused", "failed to connect", "timeout", "ollama")):
            return "ollama_unavailable"
    return "internal_error"


async def _ainvoke_with_optional_callbacks(
    chain: Runnable,
    inputs: dict[str, Any],
    execution_logger: Phase10ExecutionLogger | None,
) -> str:
    if execution_logger is None or not phase10_settings.enable_execution_logs:
        return await chain.ainvoke(inputs)
    return await chain.ainvoke(inputs, config={"callbacks": [execution_logger]})


async def run_with_fallback(
    primary_chain: Runnable,
    inputs: dict[str, Any],
    *,
    role: str = "student",
    request_id: str = "",
    execution_logger: Phase10ExecutionLogger | None = None,
) -> tuple[str, str]:
    """
    Invoke the primary chain, then retry once with the fallback model on error.

    Returns ``(raw_text, model_label)`` where ``model_label`` is either
    ``phase10_settings.llm_primary_label`` or ``phase10_settings.llm_fallback_label``.
    """
    try:
        raw = await _ainvoke_with_optional_callbacks(primary_chain, inputs, execution_logger)
        if execution_logger is not None:
            execution_logger.set_model_used(phase10_settings.llm_primary_label)
        return raw, phase10_settings.llm_primary_label
    except Exception as exc:
        logger.warning(
            "phase10 primary chain failed request_id=%s error=%s - trying fallback",
            request_id,
            exc,
        )
        if execution_logger is not None:
            execution_logger.mark_retry()
            execution_logger.mark_fallback_triggered(_infer_chain_failure_reason(exc))

    try:
        fallback_model = get_fallback_model(role=role)
        fallback_chain = build_generation_chain(fallback_model)
        raw = await _ainvoke_with_optional_callbacks(fallback_chain, inputs, execution_logger)
        if execution_logger is not None:
            execution_logger.set_model_used(phase10_settings.llm_fallback_label)
            execution_logger.set_execution_mode(ExecutionMode.FALLBACK)
        return raw, phase10_settings.llm_fallback_label
    except Exception as exc:
        logger.error("phase10 fallback chain also failed request_id=%s error=%s", request_id, exc)
        if execution_logger is not None:
            execution_logger.mark_retry()
            execution_logger.mark_fallback_triggered(_infer_chain_failure_reason(exc))
        raise RuntimeError(f"Both primary and fallback LangChain chains failed: {exc}") from exc


async def run_repair_with_fallback(
    raw_bad_output: str,
    target: str,
    *,
    request_id: str = "",
    max_attempts: int = 2,
    execution_logger: Phase10ExecutionLogger | None = None,
) -> str:
    """
    Ask the fallback model to repair malformed JSON output.

    Returns the last raw string whether or not it became valid JSON.
    """
    repair_prompt_text = build_repair_prompt(raw_bad_output, target)
    role = "professor" if "professor" in str(target).lower() else "student"
    fallback_model = get_fallback_model(role=role)
    repair_chain = build_repair_chain(fallback_model)

    last_output = raw_bad_output
    for attempt in range(1, max_attempts + 1):
        try:
            last_output = await _ainvoke_with_optional_callbacks(
                repair_chain,
                {"repair_prompt": repair_prompt_text},
                execution_logger,
            )
            if execution_logger is not None:
                execution_logger.mark_retry()

            parsed, err = try_parse_json(last_output)
            if parsed is not None:
                logger.info(
                    "phase10 json_repair success request_id=%s attempt=%d/%d",
                    request_id,
                    attempt,
                    max_attempts,
                )
                return last_output

            logger.warning(
                "phase10 json_repair attempt=%d/%d still invalid: %s",
                attempt,
                max_attempts,
                err,
            )
            if execution_logger is not None:
                execution_logger.mark_parse_failure(err or "json repair attempt remained invalid")
            repair_prompt_text = build_repair_prompt(last_output, target)
        except Exception as exc:
            logger.warning("phase10 json_repair attempt=%d/%d exception: %s", attempt, max_attempts, exc)
            if execution_logger is not None:
                execution_logger.mark_retry()
                execution_logger.mark_parse_failure(str(exc))

    return last_output
