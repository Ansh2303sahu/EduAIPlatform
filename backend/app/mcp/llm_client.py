"""
Phase 11.2 MCP shared LLM client.

All MCP tools that need LLM generation go through this module.  It mirrors
the existing backend pattern (llm_service_url + llm_service_secret from
app.core.config) and calls the existing ``/llm/generate`` endpoint on
llm-service — the same controlled path used by Phase 10 LangChain pipelines.

Key guarantees
--------------
- A single ``POST /llm/generate`` call per request; no ad-hoc per-tool HTTP.
- ``LLMCallResult`` carries both the raw text response and a ``model_used``
  string so callers can log and surface explainability metadata.
- On any transport failure, timeout, or non-2xx response the function returns
  ``LLMCallResult(ok=False, ...)`` — never raises — so tools can implement
  their own deterministic fallback without try/except overhead.
- JSON extraction and cheap repair are delegated to ``_extract_json`` so tools
  that need structured output share one parsing path.
- ``max_output_chars`` is enforced before the response leaves this module.
- Internal prompts and secrets are never returned to callers.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.core.config import settings

logger = logging.getLogger("mcp.llm_client")

# Maximum character length for any text sent to the LLM — hard-coded safety cap.
_MAX_INPUT_CHARS: int = 20_000
# Maximum characters of raw LLM response that are ever returned.
_MAX_OUTPUT_CHARS: int = 6_000
# HTTP timeout for the llm-service call (the tool-level asyncio timeout is
# handled by the executor; this is the httpx socket-level safety net).
_HTTP_TIMEOUT: float = 40.0

# Cheap regex repairs for common local-model JSON mistakes.
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_PYTHON_BOOL_RE = re.compile(r"\b(True|False|None)\b")
_SMART_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})
_BARE_KEY_RE = re.compile(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_ -]*?)(\s*:(?!\s*/))')


@dataclass
class LLMCallResult:
    """
    Result of a single LLM generation call.

    ok=True  → ``text`` contains the raw response; ``model_used`` is populated.
    ok=False → ``error_reason`` explains the failure; ``text`` is empty.
    """

    ok: bool
    text: str = ""
    model_used: str = ""
    error_reason: str = ""
    # Whether the llm-service fell back to a secondary model.
    used_fallback: bool = False


@dataclass
class ParsedLLMResult:
    """
    Outcome of attempting to parse a structured (JSON) LLM response.

    ok=True  → ``data`` is the parsed dict/list.
    ok=False → ``error_reason`` explains why; ``data`` is ``{}``.
    """

    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error_reason: str = ""
    raw_text: str = ""


def _cheap_repair(text: str) -> str:
    """Apply cheap, regex-only JSON repairs (same patterns as llm-service)."""
    text = text.translate(_SMART_QUOTES)
    text = _PYTHON_BOOL_RE.sub(
        lambda m: {"True": "true", "False": "false", "None": "null"}[m.group()], text
    )
    text = _TRAILING_COMMA_RE.sub(r"\1", text)
    text = _BARE_KEY_RE.sub(
        lambda m: f'{m.group(1)}"{m.group(2).strip()}"{m.group(3)}', text
    )
    return text


def _balanced_extract(text: str) -> str | None:
    """Return the first fully balanced JSON object from ``text``, or None."""
    start: int | None = None
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if in_string:
            if ch == "\\":
                escape_next = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start : i + 1].strip()
                if ":" in candidate:
                    return candidate
                start = None
    return None


def _extract_json(raw: str) -> str:
    """
    Extract the first balanced JSON object from raw LLM output.

    Tries strategies in order:
    1. Fenced JSON block (``` ... ```)
    2. Balanced extract on raw text
    3. Cheap repair then balanced extract
    4. Last resort: first { to last }
    """
    text = raw.strip() if isinstance(raw, str) else str(raw).strip()

    # Strategy 1: fenced block
    fence_re = re.compile(r"```(?:json|JSON)?\s*", re.IGNORECASE)
    close_re = re.compile(r"\s*```")
    pos = 0
    while True:
        m_open = fence_re.search(text, pos)
        if not m_open:
            break
        inner_start = m_open.end()
        m_close = close_re.search(text, inner_start)
        inner_end = m_close.start() if m_close else len(text)
        block = text[inner_start:inner_end].strip()
        for attempt in (block, _cheap_repair(block)):
            candidate = _balanced_extract(attempt)
            if candidate:
                return candidate
        pos = inner_end + 1
        if not m_close:
            break

    # Strategy 2: balanced extract
    candidate = _balanced_extract(text)
    if candidate:
        return candidate

    # Strategy 3: cheap repair then balanced extract
    candidate = _balanced_extract(_cheap_repair(text))
    if candidate:
        return candidate

    # Strategy 4: last resort
    start_obj = text.find("{")
    end_obj = text.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        return text[start_obj : end_obj + 1].strip()

    return text


def parse_json_response(result: LLMCallResult) -> ParsedLLMResult:
    """
    Attempt to parse a JSON dict from a successful ``LLMCallResult``.

    Returns ``ParsedLLMResult(ok=False, ...)`` if the result was not ok or the
    text cannot be parsed as a JSON object.
    """
    if not result.ok:
        return ParsedLLMResult(ok=False, error_reason=result.error_reason, raw_text="")

    raw = result.text
    json_str = _extract_json(raw)
    try:
        data = json.loads(json_str)
        if not isinstance(data, dict):
            return ParsedLLMResult(
                ok=False,
                error_reason=f"LLM returned non-dict JSON: {type(data).__name__}",
                raw_text=raw[:500],
            )
        return ParsedLLMResult(ok=True, data=data, raw_text=raw)
    except json.JSONDecodeError as exc:
        return ParsedLLMResult(
            ok=False,
            error_reason=f"JSON decode failed: {exc}",
            raw_text=raw[:500],
        )


async def call_llm(
    prompt: str,
    *,
    role: str = "student",
    temperature: float = 0.2,
    tool_name: str = "mcp_tool",
) -> LLMCallResult:
    """
    Send a single prompt to the llm-service ``/llm/generate`` endpoint.

    Parameters
    ----------
    prompt
        The full prompt string.  Truncated to ``_MAX_INPUT_CHARS`` before
        transmission — the LLM service also has its own cap.
    role
        ``"student"`` or ``"professor"``.  Passed to the llm-service for
        logging context; does not change model selection.
    temperature
        Generation temperature.  Lower = more deterministic.
    tool_name
        Included in log messages for traceability.

    Returns
    -------
    LLMCallResult
        Always returns — never raises.  Callers must check ``result.ok``.
    """
    llm_url = settings.llm_service_url.rstrip("/")
    llm_secret = settings.llm_service_secret

    if not llm_url:
        return LLMCallResult(
            ok=False,
            error_reason="LLM_SERVICE_URL is not configured",
        )
    if not llm_secret:
        return LLMCallResult(
            ok=False,
            error_reason="LLM_SERVICE_SECRET is not configured",
        )

    safe_prompt = prompt[:_MAX_INPUT_CHARS]

    payload: dict[str, Any] = {
        "prompt": safe_prompt,
        "role": role,
        "temperature": temperature,
    }

    headers = {
        "Content-Type": "application/json",
        "X-AI-SECRET": llm_secret,
    }

    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.post(
                f"{llm_url}/llm/generate",
                json=payload,
                headers=headers,
            )
    except httpx.TimeoutException:
        logger.warning("mcp.llm_client: timeout calling llm-service tool=%s", tool_name)
        return LLMCallResult(ok=False, error_reason="llm-service request timed out")
    except httpx.RequestError as exc:
        logger.warning(
            "mcp.llm_client: transport error tool=%s error=%s", tool_name, exc
        )
        return LLMCallResult(
            ok=False, error_reason=f"llm-service unreachable: {type(exc).__name__}"
        )

    if resp.status_code != 200:
        detail = ""
        try:
            body = resp.json()
            detail = str(body.get("detail") or body.get("message") or "").strip()
        except Exception:
            detail = (resp.text or "").strip()[:200]
        logger.warning(
            "mcp.llm_client: non-200 from llm-service tool=%s status=%s",
            tool_name,
            resp.status_code,
        )
        return LLMCallResult(
            ok=False,
            error_reason=(
                f"llm-service returned HTTP {resp.status_code}"
                + (f": {detail}" if detail else "")
            ),
        )

    try:
        body = resp.json()
    except Exception as exc:
        return LLMCallResult(
            ok=False, error_reason=f"llm-service response not JSON: {exc}"
        )

    raw_text = str(body.get("response") or "").strip()
    model_used = str(body.get("model_used") or "")
    used_fallback = resp.headers.get("x-llm-fallback", "").lower() == "true"

    if not raw_text:
        return LLMCallResult(
            ok=False,
            error_reason="llm-service returned empty response",
            model_used=model_used,
        )

    return LLMCallResult(
        ok=True,
        text=raw_text[:_MAX_OUTPUT_CHARS],
        model_used=model_used,
        used_fallback=used_fallback,
    )
