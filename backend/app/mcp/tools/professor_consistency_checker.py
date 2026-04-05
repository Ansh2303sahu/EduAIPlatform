"""
MCP Tool: professor.consistency_checker.v1

Evaluation consistency checker for professors.

Combines two passes:
1. Deterministic checks (always run):
   - Sentiment–score mismatch detection
   - Statistical outlier detection
   - Count mismatch
   - Final-summary vs. scores contradiction
2. Optional LLM semantic review (when llm-service is available):
   - Detects tone inconsistencies across multiple feedback items
   - Suggests specific fixes for identified contradictions

Must remain fully functional without the LLM pass.

Input  : ``ConsistencyInput``
Output : ``ConsistencyOutput``
"""

from __future__ import annotations

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


class ConsistencyInput(BaseModel):
    model_config = {"extra": "forbid"}

    feedback_items: list[str] = Field(..., min_length=1, max_length=50)
    scores: list[float] = Field(..., min_length=1, max_length=50)
    expected_band_labels: list[str] = Field(default_factory=list, max_length=10)
    final_summary: str | None = Field(default=None, max_length=2_000)


class ConsistencyOutput(BaseModel):
    model_config = {"extra": "forbid"}

    consistency_score: float = Field(..., ge=0.0, le=1.0)
    contradictions: list[str]
    warnings: list[str]
    suggested_fixes: list[str]
    confidence_note: str


# ---------------------------------------------------------------------------
# Deterministic detection helpers
# ---------------------------------------------------------------------------

_POSITIVE_WORDS = frozenset(
    "excellent strong good well outstanding clear thorough solid "
    "impressive comprehensive detailed accurate sophisticated".split()
)
_NEGATIVE_WORDS = frozenset(
    "poor weak lacking insufficient missing unclear inadequate "
    "superficial incomplete vague underdeveloped inconsistent".split()
)

_LOW_SCORE_THRESHOLD = 40.0
_HIGH_SCORE_THRESHOLD = 70.0
_OUTLIER_DEVIATION = 20.0


def _token_sentiment(text: str) -> str:
    """Return 'positive', 'negative', or 'neutral' based on keyword presence."""
    words = set(text.lower().split())
    pos = words & _POSITIVE_WORDS
    neg = words & _NEGATIVE_WORDS
    if pos and not neg:
        return "positive"
    if neg and not pos:
        return "negative"
    return "neutral"


def _det_contradictions(
    feedback_items: list[str],
    scores: list[float],
) -> tuple[list[str], list[str]]:
    """
    Return (contradictions, suggestions) from sentiment-score mismatch analysis.
    """
    contradictions: list[str] = []
    suggestions: list[str] = []
    n = min(len(feedback_items), len(scores))
    for i in range(n):
        sentiment = _token_sentiment(feedback_items[i])
        score = scores[i]
        if sentiment == "positive" and score < _LOW_SCORE_THRESHOLD:
            contradictions.append(
                f"Item {i + 1}: positive language but low score ({score:.1f}). "
                f"Check: \"{feedback_items[i][:80]}...\""
                if len(feedback_items[i]) > 80
                else f"Item {i + 1}: positive language but low score ({score:.1f})."
            )
            suggestions.append(
                f"Item {i + 1}: Revise score upward OR replace positive language "
                f"with language consistent with a score of {score:.1f}."
            )
        elif sentiment == "negative" and score >= _HIGH_SCORE_THRESHOLD:
            contradictions.append(
                f"Item {i + 1}: negative language but high score ({score:.1f})."
            )
            suggestions.append(
                f"Item {i + 1}: Revise score downward OR soften negative language "
                f"to match a score of {score:.1f}."
            )
    return contradictions, suggestions


def _det_outliers(scores: list[float]) -> list[str]:
    if len(scores) < 2:
        return []
    mean = sum(scores) / len(scores)
    return [
        f"Score at position {i + 1} ({s:.1f}) deviates "
        f"{abs(s - mean):.1f} points from mean ({mean:.1f})."
        for i, s in enumerate(scores)
        if abs(s - mean) > _OUTLIER_DEVIATION
    ]


def _det_summary_mismatch(
    final_summary: str | None,
    scores: list[float],
) -> list[str]:
    """Detect if final summary tone contradicts the numeric scores."""
    if not final_summary or not scores:
        return []
    mean_score = sum(scores) / len(scores)
    sentiment = _token_sentiment(final_summary)
    issues: list[str] = []
    if sentiment == "positive" and mean_score < _LOW_SCORE_THRESHOLD:
        issues.append(
            f"Final summary is positive but mean score is {mean_score:.1f} "
            f"(< {_LOW_SCORE_THRESHOLD:.0f})."
        )
    elif sentiment == "negative" and mean_score >= _HIGH_SCORE_THRESHOLD:
        issues.append(
            f"Final summary is negative but mean score is {mean_score:.1f} "
            f"(>= {_HIGH_SCORE_THRESHOLD:.0f})."
        )
    return issues


def _compute_score(
    contradictions: list[str],
    warnings: list[str],
) -> float:
    base = 1.0
    base -= min(len(contradictions) * 0.2, 0.6)
    if warnings:
        base = min(base, 0.85)
    return round(max(0.1, base), 2)


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
You are helping a professor check evaluation consistency.

FEEDBACK ITEMS:
{feedback_items}

SCORES: {scores}
{summary_section}

Rules:
- Return ONLY a JSON object with keys: "contradictions", "suggested_fixes", "warnings".
- Each value is a JSON array of strings.
- "contradictions": describe each tone/score mismatch you detect.
- "suggested_fixes": one actionable suggestion per contradiction.
- "warnings": other concerns (e.g. outlier scores, vague feedback).
- Be concise and specific. Maximum 2 sentences per item.
- Start with {{ and end with }}. No prose outside the JSON.
"""


def _build_prompt(input: ConsistencyInput) -> str:
    feedback_lines = "\n".join(
        f"{i + 1}. \"{item}\" (score: {input.scores[i]:.1f})"
        if i < len(input.scores) else f"{i + 1}. \"{item}\""
        for i, item in enumerate(input.feedback_items[:20])
    )
    summary_section = (
        f"\nFINAL SUMMARY:\n{input.final_summary[:500]}"
        if input.final_summary else ""
    )
    return _PROMPT_TEMPLATE.format(
        feedback_items=feedback_lines,
        scores=", ".join(f"{s:.1f}" for s in input.scores[:20]),
        summary_section=summary_section,
    )


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

async def _handle(input: ConsistencyInput, ctx: ToolExecutionContext) -> HandlerResult:
    from app.mcp.config import mcp_settings

    warnings: list[str] = []

    if len(input.feedback_items) != len(input.scores):
        warnings.append(
            f"Count mismatch: {len(input.feedback_items)} feedback item(s) "
            f"but {len(input.scores)} score(s)."
        )

    # Deterministic pass — always runs.
    det_contradictions, det_suggestions = _det_contradictions(
        input.feedback_items, input.scores
    )
    outlier_warnings = _det_outliers(input.scores)
    summary_issues = _det_summary_mismatch(input.final_summary, input.scores)
    warnings.extend(outlier_warnings)

    contradictions = det_contradictions + summary_issues
    suggested_fixes = det_suggestions[:]

    if mcp_settings.llm_enabled:
        llm_result = await call_llm(
            _build_prompt(input),
            role="professor",
            temperature=0.1,
            tool_name="professor.consistency_checker.v1",
        )
        parsed = parse_json_response(llm_result)

        if parsed.ok:
            llm_contradictions = [
                str(c).strip()
                for c in (parsed.data.get("contradictions") or [])
                if str(c).strip()
            ]
            llm_fixes = [
                str(f).strip()
                for f in (parsed.data.get("suggested_fixes") or [])
                if str(f).strip()
            ]
            llm_warnings = [
                str(w).strip()
                for w in (parsed.data.get("warnings") or [])
                if str(w).strip()
            ]

            # Merge LLM findings with deterministic findings; avoid duplicates.
            existing_text = " ".join(contradictions).lower()
            for c in llm_contradictions:
                if c.lower()[:40] not in existing_text:
                    contradictions.append(c)
            for f in llm_fixes:
                if f.lower()[:40] not in " ".join(suggested_fixes).lower():
                    suggested_fixes.append(f)
            for w in llm_warnings:
                if w.lower()[:40] not in " ".join(warnings).lower():
                    warnings.append(w)

            consistency_score = _compute_score(contradictions, warnings)
            return HandlerResult(
                output=ConsistencyOutput(
                    consistency_score=consistency_score,
                    contradictions=contradictions[:15],
                    warnings=warnings[:10],
                    suggested_fixes=suggested_fixes[:15],
                    confidence_note=(
                        "Consistency analysis combines deterministic checks and "
                        "language model semantic review."
                    ),
                ),
                llm_used=True,
                llm_fallback_used=llm_result.used_fallback,
                model_used=llm_result.model_used,
                deterministic_fallback=False,
                confidence_note="LLM-assisted consistency check.",
            )

    # Pure deterministic path
    consistency_score = _compute_score(contradictions, warnings)
    return HandlerResult(
        output=ConsistencyOutput(
            consistency_score=consistency_score,
            contradictions=contradictions[:15],
            warnings=warnings[:10],
            suggested_fixes=suggested_fixes[:15],
            confidence_note=(
                "Consistency analysis based on deterministic sentiment and "
                "score checks. LLM semantic review was not available."
            ),
        ),
        llm_used=False,
        deterministic_fallback=True,
        confidence_note="Deterministic consistency check.",
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_tool(
    ToolDefinition(
        tool_name="professor.consistency_checker.v1",
        namespace=ToolNamespace.PROFESSOR,
        version="v1",
        description=(
            "Checks a professor's feedback items and paired scores for tone–score "
            "contradictions, statistical outliers, and final-summary mismatches. "
            "Returns a consistency score, contradiction descriptions, and suggested fixes."
        ),
        allowed_roles=frozenset({ToolRole.PROFESSOR, ToolRole.ADMIN}),
        risk_level=RiskLevel.LOW,
        enabled=True,
        timeout_seconds=35.0,
        supports_idempotency=True,
        safe_for_multi_step=True,
        input_model=ConsistencyInput,
        output_model=ConsistencyOutput,
        handler=_handle,
    )
)
