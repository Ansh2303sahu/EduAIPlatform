"""
Phase 11 / 11.2 MCP execution gateway.

``execute_tool`` is the single controlled entry point for all tool invocations.
No tool handler is ever called directly from a prompt, router, or external
service.

Execution flow (Phase 11.2 additions marked with ✦)
----------------------------------------------------
1.  Assign a unique ``request_id``.
2.  Resolve the tool from the registry → ``UnknownToolError`` / ``DisabledToolError``.
3.  Enforce synchronous access policy → ``PolicyDeniedError``.
4. ✦ Enforce async ownership policy → ``PolicyDeniedError``.
5.  Enforce per-tool rate limit → ``RateLimitError``.
6.  Validate the raw payload against ``defn.input_model`` → ``InputValidationError``.
7. ✦ Cache hit check (idempotent tools only) → return cached envelope immediately.
8.  Emit ``mcp.tool.started`` audit event (best-effort).
9.  Run handler with timeout wrapper → ``ToolTimeoutError``.
10. Validate handler output against ``defn.output_model`` → ``OutputValidationError``.
11. Emit ``mcp.tool.succeeded`` audit event (best-effort).
12. ✦ Store result in cache (idempotent tools only).
13. ✦ Record success metrics.
14. Return ``MCPSuccessEnvelope`` with ``ExplainabilityMeta``.

On any exception:
- ✦ Record failure metrics.
- Emit ``mcp.tool.failed`` audit event (best-effort).
- Return ``MCPFailureEnvelope`` — never re-raise to the API layer.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from pydantic import ValidationError

from app.mcp.audit import emit_tool_failed, emit_tool_started, emit_tool_succeeded
from app.mcp.handler_result import HandlerResult
from app.mcp.config import mcp_settings
from app.mcp.errors import (
    InputValidationError,
    MCPError,
    OutputValidationError,
    PolicyDeniedError,
    RateLimitError,
    ToolTimeoutError,
)
from app.mcp.policies import enforce_ownership_policy, enforce_policy
from app.mcp.rate_limits import check_rate_limit
from app.mcp.registry import resolve_tool
from app.mcp.response import build_failure, build_success
from app.mcp.schemas import (
    ExplainabilityMeta,
    MCPExecuteRequest,
    MCPFailureEnvelope,
    MCPSuccessEnvelope,
)
from app.mcp.timeouts import run_with_timeout

logger = logging.getLogger("mcp.executor")


def _record_success(tool_name: str, meta: ExplainabilityMeta) -> None:
    if not mcp_settings.metrics_enabled:
        return
    from app.mcp.metrics import record_success
    record_success(
        tool_name,
        llm_used=meta.llm_used,
        fallback_used=(meta.llm_fallback_used or meta.deterministic_fallback),
        cache_hit=meta.cache_hit,
    )


def _record_failure(tool_name: str, exc: Exception) -> None:
    if not mcp_settings.metrics_enabled:
        return
    from app.mcp.metrics import record_failure
    record_failure(
        tool_name,
        error_code=getattr(exc, "error_code", "mcp.internal_error"),
        timeout=isinstance(exc, ToolTimeoutError),
        policy_denied=isinstance(exc, PolicyDeniedError),
        rate_limited=isinstance(exc, RateLimitError),
    )


def _record_latency(tool_name: str, execution_ms: float) -> None:
    if not mcp_settings.metrics_enabled:
        return
    from app.mcp.metrics import record_latency
    record_latency(tool_name, execution_ms)


async def execute_tool(
    req: MCPExecuteRequest,
) -> MCPSuccessEnvelope | MCPFailureEnvelope:
    """
    Controlled execution gateway for MCP tools.

    Always returns an envelope — never raises.  The API layer wraps this with
    HTTP 200 so callers always receive a structured response.
    """
    request_id = str(uuid.uuid4())
    ctx = req.context
    tool_name = req.tool_name
    t_start = time.perf_counter()

    defn = None
    try:
        # --- Step 2: resolve ---
        defn = resolve_tool(tool_name)

        # --- Step 3: synchronous policy ---
        enforce_policy(defn, ctx)

        # --- Step 4: async ownership ---
        await enforce_ownership_policy(ctx)

        # --- Step 5: rate limit ---
        check_rate_limit(
            tool_name,
            ctx.user_id,
            max_calls=mcp_settings.default_rate_limit_calls,
            window_seconds=mcp_settings.default_rate_limit_window_seconds,
        )

        # --- Step 6: validate input ---
        try:
            validated_input = defn.input_model.model_validate(req.payload)
        except ValidationError as exc:
            raise InputValidationError(f"Input validation failed: {exc}") from exc

        # --- Step 7: cache hit check ---
        if defn.supports_idempotency and mcp_settings.cache_ttl_seconds > 0:
            from app.mcp.cache import get as cache_get
            cached_result = cache_get(
                tool_name,
                defn.version,
                ctx.role,
                ctx.user_id,
                ctx.file_id,
                ctx.submission_id,
                req.payload,
                ttl_seconds=mcp_settings.cache_ttl_seconds,
            )
            if cached_result is not None:
                execution_ms = (time.perf_counter() - t_start) * 1000
                meta = ExplainabilityMeta(
                    cache_hit=True,
                    llm_used=False,
                    confidence_note="Result served from cache.",
                )
                _record_success(tool_name, meta)
                _record_latency(tool_name, execution_ms)
                logger.info(
                    "mcp.executor: cache hit tool=%r role=%s user=%s exec_ms=%.1f",
                    tool_name, ctx.role, ctx.user_id, execution_ms,
                )
                return build_success(
                    defn=defn,
                    result=cached_result,
                    request_id=request_id,
                    correlation_id=ctx.correlation_id,
                    execution_ms=execution_ms,
                    meta=meta,
                )

        # --- Step 8: audit started ---
        if mcp_settings.audit_enabled:
            await emit_tool_started(
                tool_name=tool_name,
                tool_version=defn.version,
                ctx=ctx,
                request_id=request_id,
            )

        # --- Step 9: run handler with timeout ---
        async def _invoke() -> Any:
            return await defn.handler(validated_input, ctx)  # type: ignore[union-attr]

        raw_output = await run_with_timeout(
            _invoke,
            timeout_seconds=defn.timeout_seconds,
            tool_name=tool_name,
        )

        # --- Step 10: validate output ---
        # Unwrap HandlerResult if the handler returned one.
        if isinstance(raw_output, HandlerResult):
            handler_result = raw_output
            raw_model = handler_result.output
        else:
            handler_result = None
            raw_model = raw_output

        try:
            raw_dict = (
                raw_model.model_dump()
                if hasattr(raw_model, "model_dump")
                else raw_model
            )
            validated_output = defn.output_model.model_validate(raw_dict)
        except ValidationError as exc:
            raise OutputValidationError(f"Output validation failed: {exc}") from exc

        execution_ms = (time.perf_counter() - t_start) * 1000

        # --- Step 11: audit succeeded ---
        if mcp_settings.audit_enabled:
            await emit_tool_succeeded(
                tool_name=tool_name,
                tool_version=defn.version,
                ctx=ctx,
                request_id=request_id,
                execution_ms=execution_ms,
            )

        # Build explainability meta from HandlerResult or plain output dict.
        output_dict = validated_output.model_dump()
        meta = _build_meta_from_handler(handler_result)

        # --- Step 12: cache store ---
        if defn.supports_idempotency and mcp_settings.cache_ttl_seconds > 0:
            from app.mcp.cache import put as cache_put
            cache_put(
                tool_name,
                defn.version,
                ctx.role,
                ctx.user_id,
                ctx.file_id,
                ctx.submission_id,
                req.payload,
                output_dict,
            )

        # --- Step 13: metrics ---
        _record_success(tool_name, meta)
        _record_latency(tool_name, execution_ms)

        logger.info(
            "mcp.executor: success tool=%r role=%s user=%s exec_ms=%.1f "
            "llm_used=%s fallback=%s cache_hit=%s",
            tool_name, ctx.role, ctx.user_id, execution_ms,
            meta.llm_used, meta.llm_fallback_used, meta.cache_hit,
        )

        # --- Step 14: return ---
        return build_success(
            defn=defn,
            result=output_dict,
            request_id=request_id,
            correlation_id=ctx.correlation_id,
            execution_ms=execution_ms,
            meta=meta,
        )

    except Exception as exc:
        execution_ms = (time.perf_counter() - t_start) * 1000

        if not isinstance(exc, MCPError):
            logger.exception(
                "mcp.executor: unexpected error tool=%r role=%s user=%s",
                tool_name, ctx.role, ctx.user_id,
            )
        else:
            logger.warning(
                "mcp.executor: controlled failure tool=%r error_code=%s role=%s "
                "user=%s message=%s",
                tool_name, exc.error_code, ctx.role, ctx.user_id, str(exc),
            )

        _record_failure(tool_name, exc)
        _record_latency(tool_name, execution_ms)

        error_code = exc.error_code if isinstance(exc, MCPError) else "mcp.internal_error"

        if mcp_settings.audit_enabled:
            await emit_tool_failed(
                tool_name=tool_name,
                tool_version=defn.version if defn is not None else None,
                ctx=ctx,
                request_id=request_id,
                error_code=error_code,
                execution_ms=execution_ms,
            )

        return build_failure(
            tool_name=tool_name,
            tool_version=defn.version if defn is not None else None,
            request_id=request_id,
            correlation_id=ctx.correlation_id,
            error=exc,
            execution_ms=execution_ms,
        )


def _build_meta_from_handler(
    handler_result: HandlerResult | None,
) -> ExplainabilityMeta:
    """
    Build ``ExplainabilityMeta`` from a ``HandlerResult`` if one was returned.

    When a handler returns a plain ``BaseModel`` (not a ``HandlerResult``),
    defaults are used (deterministic_fallback=True for backward compatibility).
    """
    if handler_result is None:
        return ExplainabilityMeta(
            cache_hit=False,
            deterministic_fallback=True,
            confidence_note="",
        )
    return ExplainabilityMeta(
        cache_hit=False,
        llm_used=handler_result.llm_used,
        llm_fallback_used=handler_result.llm_fallback_used,
        model_used=handler_result.model_used,
        deterministic_fallback=handler_result.deterministic_fallback,
        confidence_note=handler_result.confidence_note,
    )
