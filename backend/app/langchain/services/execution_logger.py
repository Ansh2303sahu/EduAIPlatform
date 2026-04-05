"""
Execution logging abstraction for Phase 10 LangChain runs.

The logger is deliberately conservative:
- it never logs full prompts
- it never logs raw private document content by default
- raw LLM output logging is opt-in via ``PHASE10_STORE_RAW_OUTPUT``

It can be used in two ways:
- as a LangChain callback handler
- as a small metadata collector with ``build_payload()``, ``emit()``, and
  ``persist()``
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from pydantic import BaseModel, Field

from app.langchain.config import phase10_settings
from app.langchain.enums import DecisionSource, ExecutionMode
from app.services.audit import audit_log

logger = logging.getLogger("phase10.execution")


def _clip_text(value: str, limit: int = 1200) -> str:
    text = value if isinstance(value, str) else str(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _prompt_version_for_role(role: str) -> str:
    return (
        phase10_settings.professor_prompt_version
        if str(role).strip().lower() == "professor"
        else phase10_settings.student_prompt_version
    )


class Phase10ExecutionRecord(BaseModel):
    """Structured execution metadata captured for one Phase 10 run."""

    request_id: str = ""
    role: str = ""
    file_id: str = ""
    user_id: str = ""
    model_used: str = ""
    chain_version: str = ""
    chain_name: str = ""
    prompt_version: str = ""
    schema_version: str = ""
    analysis_type: str = ""
    submission_kind: str = ""
    execution_mode: str = ExecutionMode.NORMAL.value
    retry_count: int = 0
    parse_failures: int = 0
    validation_failures: int = 0
    weak_retrieval: bool = False
    low_confidence: bool = False
    decision_source: str = DecisionSource.LLM.value
    fallback_triggered: bool = False
    fallback_reason: str = ""
    retrieval_mode: str = ""
    query_summary: str = ""
    title_hint: str = ""
    prompt_chars: int = 0
    prompt_trimmed: bool = False
    retrieved_doc_count: int = 0
    selected_doc_titles: list[str] = Field(default_factory=list)
    selected_categories: list[str] = Field(default_factory=list)
    degraded_retrieval_mode: bool = False
    duration_ms: int = 0
    parse_failure_details: list[str] = Field(default_factory=list)
    validation_failure_details: list[str] = Field(default_factory=list)
    raw_output_excerpt: str | None = None


class Phase10ExecutionLogger(BaseCallbackHandler):
    """
    Structured logging callback for LangChain execution.

    Attach to any runnable via ``config={"callbacks": [logger]}``, or use it
    manually as a metadata collector around a pipeline run.
    """

    def __init__(
        self,
        request_id: str,
        role: str,
        *,
        file_id: str = "",
        user_id: str = "",
        model_used: str = "",
        chain_version: str | None = None,
        prompt_version: str | None = None,
        schema_version: str | None = None,
        execution_mode: str | ExecutionMode = ExecutionMode.NORMAL,
        decision_source: str | DecisionSource = DecisionSource.LLM,
        retry_count: int = 0,
        parse_failures: int = 0,
        validation_failures: int = 0,
        weak_retrieval: bool = False,
        low_confidence: bool = False,
        fallback_triggered: bool = False,
        store_raw_output: bool | None = None,
        enable_logging: bool | None = None,
    ) -> None:
        super().__init__()
        self.record = Phase10ExecutionRecord(
            request_id=request_id,
            role=str(role),
            file_id=file_id,
            user_id=user_id,
            model_used=model_used,
            chain_version=chain_version or phase10_settings.chain_version,
            prompt_version=prompt_version or _prompt_version_for_role(role),
            schema_version=schema_version or phase10_settings.schema_version,
            execution_mode=(
                execution_mode.value if isinstance(execution_mode, ExecutionMode) else str(execution_mode)
            ),
            retry_count=max(0, int(retry_count)),
            parse_failures=max(0, int(parse_failures)),
            validation_failures=max(0, int(validation_failures)),
            weak_retrieval=bool(weak_retrieval),
            low_confidence=bool(low_confidence),
            decision_source=(
                decision_source.value if isinstance(decision_source, DecisionSource) else str(decision_source)
            ),
            fallback_triggered=bool(fallback_triggered),
        )
        self._created_at = time.perf_counter()
        self._run_start_times: dict[str, float] = {}
        self._prompt_chars = 0
        self._store_raw_output = (
            phase10_settings.store_raw_output if store_raw_output is None else bool(store_raw_output)
        )
        self._enable_logging = (
            phase10_settings.enable_execution_logs if enable_logging is None else bool(enable_logging)
        )

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def bind_context(
        self,
        *,
        file_id: str | None = None,
        user_id: str | None = None,
        model_used: str | None = None,
        chain_name: str | None = None,
        analysis_type: str | None = None,
        submission_kind: str | None = None,
    ) -> "Phase10ExecutionLogger":
        if file_id is not None:
            self.record.file_id = file_id
        if user_id is not None:
            self.record.user_id = user_id
        if model_used is not None:
            self.record.model_used = model_used
        if chain_name is not None:
            self.record.chain_name = str(chain_name)
        if analysis_type is not None:
            self.record.analysis_type = str(analysis_type)
        if submission_kind is not None:
            self.record.submission_kind = str(submission_kind)
        return self

    def set_model_used(self, model_used: str) -> "Phase10ExecutionLogger":
        self.record.model_used = str(model_used or "")
        return self

    def set_execution_mode(self, mode: str | ExecutionMode) -> "Phase10ExecutionLogger":
        self.record.execution_mode = mode.value if isinstance(mode, ExecutionMode) else str(mode)
        return self

    def set_decision_source(self, source: str | DecisionSource) -> "Phase10ExecutionLogger":
        self.record.decision_source = source.value if isinstance(source, DecisionSource) else str(source)
        return self

    def set_prompt_metrics(
        self,
        *,
        prompt_text: str | None = None,
        prompt_chars: int | None = None,
        prompt_trimmed: bool | None = None,
    ) -> "Phase10ExecutionLogger":
        if prompt_chars is not None:
            self.record.prompt_chars = max(0, int(prompt_chars))
        elif prompt_text is not None:
            self.record.prompt_chars = len(prompt_text)
        if prompt_trimmed is not None:
            self.record.prompt_trimmed = bool(prompt_trimmed)
        return self

    def set_retrieval_summary(
        self,
        *,
        retrieval_mode: str | None = None,
        query_summary: str | None = None,
        title_hint: str | None = None,
        retrieved_doc_count: int | None = None,
        selected_doc_titles: list[str] | None = None,
        selected_categories: list[str] | None = None,
        degraded_retrieval_mode: bool | None = None,
    ) -> "Phase10ExecutionLogger":
        if retrieval_mode is not None:
            self.record.retrieval_mode = str(retrieval_mode)
        if query_summary is not None:
            self.record.query_summary = _clip_text(query_summary, 240)
        if title_hint is not None:
            self.record.title_hint = _clip_text(title_hint, 180)
        if retrieved_doc_count is not None:
            self.record.retrieved_doc_count = max(0, int(retrieved_doc_count))
        if selected_doc_titles is not None:
            self.record.selected_doc_titles = [
                _clip_text(title, 140)
                for title in selected_doc_titles
                if str(title).strip()
            ][:6]
        if selected_categories is not None:
            self.record.selected_categories = [
                _clip_text(category, 80)
                for category in selected_categories
                if str(category).strip()
            ][:8]
        if degraded_retrieval_mode is not None:
            self.record.degraded_retrieval_mode = bool(degraded_retrieval_mode)
        return self

    def set_flags(
        self,
        *,
        weak_retrieval: bool | None = None,
        low_confidence: bool | None = None,
    ) -> "Phase10ExecutionLogger":
        if weak_retrieval is not None:
            self.record.weak_retrieval = bool(weak_retrieval)
        if low_confidence is not None:
            self.record.low_confidence = bool(low_confidence)
        return self

    def mark_retry(self, count: int = 1) -> "Phase10ExecutionLogger":
        self.record.retry_count += max(1, int(count))
        return self

    def mark_parse_failure(self, detail: str = "") -> "Phase10ExecutionLogger":
        self.record.parse_failures += 1
        if detail:
            self.record.parse_failure_details.append(_clip_text(detail, 300))
        return self

    def mark_validation_failure(self, detail: str = "") -> "Phase10ExecutionLogger":
        self.record.validation_failures += 1
        if detail:
            self.record.validation_failure_details.append(_clip_text(detail, 300))
        return self

    def mark_fallback_triggered(self, reason: str = "") -> "Phase10ExecutionLogger":
        self.record.fallback_triggered = True
        if reason:
            self.record.fallback_reason = str(reason)
        return self

    # ------------------------------------------------------------------
    # Serialization / output
    # ------------------------------------------------------------------

    def build_payload(
        self,
        *,
        duration_ms: int | None = None,
        raw_output: str | None = None,
    ) -> dict[str, Any]:
        if duration_ms is None:
            duration_ms = int((time.perf_counter() - self._created_at) * 1000)
        self.record.duration_ms = max(0, int(duration_ms))

        if self._store_raw_output and raw_output:
            self.record.raw_output_excerpt = _clip_text(raw_output, 1200)
        else:
            self.record.raw_output_excerpt = None

        return self.record.model_dump(exclude_none=True)

    def emit(
        self,
        *,
        duration_ms: int | None = None,
        raw_output: str | None = None,
    ) -> dict[str, Any]:
        payload = self.build_payload(duration_ms=duration_ms, raw_output=raw_output)
        if not self._enable_logging:
            return payload

        logger.info("phase10 execution_summary=%s", payload)

        if self.record.user_id:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    audit_log(
                        actor_user_id=self.record.user_id,
                        action="phase10.execution",
                        entity_type="file" if self.record.file_id else "phase10_run",
                        entity_id=self.record.file_id or self.record.request_id,
                        metadata=payload,
                    )
                )
            except RuntimeError:
                pass
        return payload

    async def persist(
        self,
        *,
        duration_ms: int | None = None,
        raw_output: str | None = None,
    ) -> dict[str, Any]:
        payload = self.build_payload(duration_ms=duration_ms, raw_output=raw_output)
        if not self._enable_logging:
            return payload

        if self.record.user_id:
            await audit_log(
                actor_user_id=self.record.user_id,
                action="phase10.execution",
                entity_type="file" if self.record.file_id else "phase10_run",
                entity_id=self.record.file_id or self.record.request_id,
                metadata=payload,
            )
        else:
            logger.info("phase10 execution_summary=%s", payload)
        return payload

    # ------------------------------------------------------------------
    # LangChain callback events
    # ------------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        key = str(run_id)
        self._run_start_times[key] = time.perf_counter()
        self._prompt_chars += sum(len(prompt) for prompt in prompts)
        if self._prompt_chars > self.record.prompt_chars:
            self.record.prompt_chars = self._prompt_chars
        if self._enable_logging:
            logger.debug(
                "phase10 llm_start request_id=%s role=%s run_id=%s prompt_chars=%d",
                self.record.request_id,
                self.record.role,
                key,
                self._prompt_chars,
            )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        key = str(run_id)
        started = self._run_start_times.pop(key, time.perf_counter())
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        llm_output = response.llm_output or {}
        model_name = str(
            llm_output.get("model_name")
            or llm_output.get("model")
            or self.record.model_used
            or ""
        )
        if model_name:
            self.record.model_used = model_name
        if self._enable_logging:
            logger.debug(
                "phase10 llm_end request_id=%s role=%s run_id=%s elapsed_ms=%d",
                self.record.request_id,
                self.record.role,
                key,
                elapsed_ms,
            )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        key = str(run_id)
        started = self._run_start_times.pop(key, time.perf_counter())
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        if self._enable_logging:
            logger.warning(
                "phase10 llm_error request_id=%s role=%s run_id=%s elapsed_ms=%d error=%s",
                self.record.request_id,
                self.record.role,
                key,
                elapsed_ms,
                error,
            )

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if self._enable_logging:
            logger.debug(
                "phase10 chain_start request_id=%s role=%s run_id=%s",
                self.record.request_id,
                self.record.role,
                str(run_id),
            )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if self._enable_logging:
            logger.debug(
                "phase10 chain_end request_id=%s role=%s run_id=%s",
                self.record.request_id,
                self.record.role,
                str(run_id),
            )

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if self._enable_logging:
            logger.warning(
                "phase10 chain_error request_id=%s role=%s run_id=%s error=%s",
                self.record.request_id,
                self.record.role,
                str(run_id),
                error,
            )
