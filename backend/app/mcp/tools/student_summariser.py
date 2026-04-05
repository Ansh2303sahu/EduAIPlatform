"""
MCP Tool: student.summariser.v1

Produces a concise academic summary of a student submission text.

LLM path  : calls the existing llm-service ``/llm/generate`` via
            ``mcp.llm_client.call_llm``.  Returns key_points extracted from
            the LLM's structured JSON response.
Fallback   : extractive sentence-boundary summarisation + simple keyword
             extraction — deterministic, never fails.

Input schema  : ``SummariserInput``
Output schema : ``SummariserOutput``

The handler embeds a ``_meta`` dict in the output so the executor can build
``ExplainabilityMeta`` without coupling the tool to the envelope layer.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from app.mcp.enums import RiskLevel, ToolNamespace, ToolRole
from app.mcp.handler_result import HandlerResult
from app.mcp.llm_client import call_llm, parse_json_response
from app.mcp.models import ToolDefinition
from app.mcp.registry import register_tool
from app.mcp.schemas import ToolExecutionContext

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

_FOCUS_MODES = Literal["overview", "methodology", "findings", "conclusion"]


class SummariserInput(BaseModel):
    model_config = {"extra": "forbid"}

    text: str = Field(..., min_length=1, max_length=8_000)
    max_sentences: int = Field(default=3, ge=1, le=10)
    focus_mode: _FOCUS_MODES = "overview"
    preserve_key_terms: bool = True


class SummariserOutput(BaseModel):
    model_config = {"extra": "forbid"}

    summary: str
    sentence_count: int
    original_word_count: int
    key_points: list[str]
    warnings: list[str]


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
# Simple academic keyword stopwords — terms to exclude from key point list.
_STOP_WORDS = frozenset(
    "the a an and or but in on at to of for with by from is are was were has have "
    "had this that these those it its they them their we our you your i".split()
)


def _extractive_summary(text: str, max_sentences: int) -> tuple[str, list[str]]:
    """Split on sentence boundaries and return first ``max_sentences`` sentences."""
    cleaned = " ".join(text.split())
    sentences = [s.strip() for s in _SENTENCE_RE.split(cleaned) if s.strip()]
    selected = sentences[:max_sentences]
    summary = " ".join(selected)
    # Extract key terms: words > 5 chars not in stop words, de-duped.
    words = re.findall(r"\b[a-zA-Z]{6,}\b", summary.lower())
    seen: set[str] = set()
    key_points: list[str] = []
    for w in words:
        if w not in _STOP_WORDS and w not in seen:
            seen.add(w)
            key_points.append(w.capitalize())
        if len(key_points) >= 5:
            break
    return summary, key_points


def _build_warnings(text: str) -> list[str]:
    word_count = len(text.split())
    warnings: list[str] = []
    if word_count < 50:
        warnings.append("Submission is very short - summary may not be representative.")
    if word_count > 3_000:
        warnings.append(
            "Submission is long - summary covers only the first portion of the text."
        )
    return warnings


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
You are an academic writing assistant. Summarise the following student submission.

FOCUS: {focus_mode}
REQUESTED SENTENCES: {max_sentences}

Rules:
- Produce ONLY a JSON object with keys: "summary", "key_points".
- "summary" must be a string of at most {max_sentences} sentences.
- "key_points" must be a JSON array of 3-5 short strings (noun phrases).
- Do NOT invent content not present in the submission.
- Do NOT add grading or evaluation language.
- Start with {{ and end with }}. No prose outside the JSON.

SUBMISSION TEXT:
{text}
"""


def _build_prompt(input: SummariserInput) -> str:
    excerpt = input.text[:6_000]
    return _PROMPT_TEMPLATE.format(
        focus_mode=input.focus_mode,
        max_sentences=input.max_sentences,
        text=excerpt,
    )


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

async def _handle(input: SummariserInput, ctx: ToolExecutionContext) -> HandlerResult:
    from app.mcp.config import mcp_settings

    word_count = len(input.text.split())
    warnings = _build_warnings(input.text)

    if mcp_settings.llm_enabled:
        llm_result = await call_llm(
            _build_prompt(input),
            role="student",
            temperature=0.15,
            tool_name="student.summariser.v1",
        )
        parsed = parse_json_response(llm_result)

        if parsed.ok:
            summary = str(parsed.data.get("summary") or "").strip()
            raw_kp = parsed.data.get("key_points") or []
            key_points = [str(k).strip() for k in raw_kp if str(k).strip()][:5]

            if summary:
                return HandlerResult(
                    output=SummariserOutput(
                        summary=summary[:1_200],
                        sentence_count=len(_SENTENCE_RE.split(summary)) or 1,
                        original_word_count=word_count,
                        key_points=key_points,
                        warnings=warnings,
                    ),
                    llm_used=True,
                    llm_fallback_used=llm_result.used_fallback,
                    model_used=llm_result.model_used,
                    deterministic_fallback=False,
                    confidence_note=(
                        "Summary generated by language model. "
                        "Review for accuracy against the original text."
                    ),
                )
            warnings.append("LLM returned empty summary - using extractive fallback.")
        else:
            warnings.append(f"LLM unavailable ({parsed.error_reason}). Using extractive fallback.")

    # Deterministic fallback
    summary, key_points = _extractive_summary(input.text, input.max_sentences)
    return HandlerResult(
        output=SummariserOutput(
            summary=summary,
            sentence_count=len(_SENTENCE_RE.split(summary)) or 1 if summary else 0,
            original_word_count=word_count,
            key_points=key_points,
            warnings=warnings,
        ),
        llm_used=False,
        deterministic_fallback=True,
        confidence_note=(
            "Extractive summary - first sentences of the submission. "
            "No semantic analysis was performed."
        ),
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_tool(
    ToolDefinition(
        tool_name="student.summariser.v1",
        namespace=ToolNamespace.STUDENT,
        version="v1",
        description=(
            "Produces a concise academic summary of a student submission. "
            "Returns the summary, key points, and word-count statistics. "
            "Uses LLM when available; falls back to extractive summarisation."
        ),
        allowed_roles=frozenset({ToolRole.STUDENT, ToolRole.ADMIN}),
        risk_level=RiskLevel.LOW,
        enabled=True,
        timeout_seconds=35.0,
        supports_idempotency=True,
        safe_for_multi_step=True,
        input_model=SummariserInput,
        output_model=SummariserOutput,
        handler=_handle,
    )
)
