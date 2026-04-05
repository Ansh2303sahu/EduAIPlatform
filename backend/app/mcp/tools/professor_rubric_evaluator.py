"""
MCP Tool: professor.rubric_evaluator.v1

Rubric-aware evaluation assistant for professors.

LLM path   : calls llm-service to generate criterion-linked justifications
             and evidence quotes grounded in the submission text.
Fallback   : heuristic band assignment based on word count + keyword presence.

This tool is ASSISTIVE — it produces provisional bands and justifications to
help professors, not to replace their academic judgement.

Input  : ``RubricInput``
Output : ``RubricOutput``
"""

from __future__ import annotations

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

_GRADING_SCALES = Literal["uk_honours", "us_letter", "percentage", "pass_fail"]
_STRICTNESS = Literal["lenient", "standard", "strict"]


class RubricInput(BaseModel):
    model_config = {"extra": "forbid"}

    submission_text: str = Field(..., min_length=1, max_length=8_000)
    rubric_criteria: list[str] = Field(..., min_length=1, max_length=20)
    grading_scale: _GRADING_SCALES = "uk_honours"
    strictness: _STRICTNESS = "standard"
    max_evidence_quotes: int = Field(default=2, ge=0, le=5)


class RubricEvaluation(BaseModel):
    model_config = {"extra": "forbid"}

    criterion: str
    provisional_band: str
    justification: str
    evidence_points: list[str]


class RubricOutput(BaseModel):
    model_config = {"extra": "forbid"}

    evaluations: list[RubricEvaluation]
    overall_assessment: str
    warnings: list[str]
    confidence_note: str


# ---------------------------------------------------------------------------
# Grading scale definitions
# ---------------------------------------------------------------------------

_SCALE_BANDS: dict[str, list[str]] = {
    "uk_honours": ["First (70%+)", "Upper Second (60-69%)", "Lower Second (50-59%)", "Third (40-49%)", "Fail (<40%)"],
    "us_letter": ["A (90-100%)", "B (80-89%)", "C (70-79%)", "D (60-69%)", "F (<60%)"],
    "percentage": ["80-100%", "60-79%", "40-59%", "20-39%", "0-19%"],
    "pass_fail": ["Pass", "Borderline Pass", "Fail"],
}

_STRICTNESS_THRESHOLD: dict[str, int] = {
    "lenient": 600,
    "standard": 1_000,
    "strict": 1_500,
}


def _heuristic_band(word_count: int, strictness: str, scale: str) -> str:
    """Assign a band based on word count and strictness as a heuristic proxy."""
    threshold = _STRICTNESS_THRESHOLD.get(strictness, 1_000)
    bands = _SCALE_BANDS.get(scale, _SCALE_BANDS["uk_honours"])
    if word_count >= threshold * 1.5:
        return bands[0]
    if word_count >= threshold:
        return bands[1]
    if word_count >= threshold * 0.5:
        return bands[2]
    if len(bands) > 3:
        return bands[3]
    return bands[-1]


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
You are an academic assessment assistant helping a professor evaluate a student submission.

GRADING SCALE: {grading_scale}
STRICTNESS: {strictness}
RUBRIC CRITERIA: {criteria}
MAX EVIDENCE QUOTES PER CRITERION: {max_evidence}

Rules:
- Return ONLY a JSON object with keys: "evaluations", "overall_assessment", "warnings".
- "evaluations" is an array. Each item has: "criterion" (string), "provisional_band" (string from the scale), "justification" (1-2 sentence string), "evidence_points" (array of short quote/reference strings from the submission).
- "overall_assessment" is a 1-2 sentence summary string.
- "warnings" is an array of strings (concerns, e.g. missing evidence, boundary cases). May be empty.
- ONLY cite evidence from the submission text. Do NOT invent quotes.
- Use the exact band labels from the grading scale.
- This is ASSISTIVE — do NOT make definitive final grade claims.
- Start with {{ and end with }}. No prose outside the JSON.

SUBMISSION TEXT:
{text}
"""


def _build_prompt(input: RubricInput) -> str:
    scale_bands = _SCALE_BANDS.get(input.grading_scale, _SCALE_BANDS["uk_honours"])
    return _PROMPT_TEMPLATE.format(
        grading_scale=f"{input.grading_scale} [{', '.join(scale_bands)}]",
        strictness=input.strictness,
        criteria=", ".join(f'"{c}"' for c in input.rubric_criteria[:10]),
        max_evidence=input.max_evidence_quotes,
        text=input.submission_text[:5_500],
    )


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

async def _handle(input: RubricInput, ctx: ToolExecutionContext) -> HandlerResult:
    from app.mcp.config import mcp_settings

    word_count = len(input.submission_text.split())
    warnings: list[str] = []

    if word_count < 100:
        warnings.append("Submission is very short - evaluation confidence is low.")

    if mcp_settings.llm_enabled:
        llm_result = await call_llm(
            _build_prompt(input),
            role="professor",
            temperature=0.1,
            tool_name="professor.rubric_evaluator.v1",
        )
        parsed = parse_json_response(llm_result)

        if parsed.ok:
            raw_evals = parsed.data.get("evaluations") or []
            evaluations: list[RubricEvaluation] = []
            for item in raw_evals[:10]:
                if not isinstance(item, dict):
                    continue
                ev_points = [
                    str(e).strip()
                    for e in (item.get("evidence_points") or [])
                    if str(e).strip()
                ][: input.max_evidence_quotes]
                evaluations.append(
                    RubricEvaluation(
                        criterion=str(item.get("criterion") or "")[:200],
                        provisional_band=str(item.get("provisional_band") or "Not assessed")[:80],
                        justification=str(item.get("justification") or "")[:800],
                        evidence_points=ev_points,
                    )
                )
            overall = str(parsed.data.get("overall_assessment") or "")[:600]
            llm_warnings = [str(w) for w in (parsed.data.get("warnings") or []) if w]
            warnings.extend(llm_warnings)

            if evaluations:
                return HandlerResult(
                    output=RubricOutput(
                        evaluations=evaluations,
                        overall_assessment=overall or "No overall assessment provided.",
                        warnings=warnings,
                        confidence_note=(
                            "Provisional evaluation assisted by language model. "
                            "Final academic judgement rests with the professor."
                        ),
                    ),
                    llm_used=True,
                    llm_fallback_used=llm_result.used_fallback,
                    model_used=llm_result.model_used,
                    deterministic_fallback=False,
                    confidence_note="LLM-assisted rubric evaluation.",
                )
            warnings.append("LLM returned no valid evaluations - using heuristic fallback.")
        else:
            warnings.append(f"LLM unavailable ({parsed.error_reason}). Using heuristic fallback.")

    # Deterministic fallback
    heuristic_band = _heuristic_band(word_count, input.strictness, input.grading_scale)
    evaluations = [
        RubricEvaluation(
            criterion=criterion,
            provisional_band=heuristic_band,
            justification=(
                f"Heuristic evaluation: submission contains approximately {word_count} words. "
                "LLM-grounded assessment was not available for this call."
            ),
            evidence_points=[],
        )
        for criterion in input.rubric_criteria[:10]
    ]

    return HandlerResult(
        output=RubricOutput(
            evaluations=evaluations,
            overall_assessment=(
                f"Heuristic assessment only - submission is ~{word_count} words. "
                "LLM grounding not available."
            ),
            warnings=warnings,
            confidence_note=(
                "Heuristic band assignment based on submission length. "
                "This is a placeholder - verify manually."
            ),
        ),
        llm_used=False,
        deterministic_fallback=True,
        confidence_note="Heuristic rubric evaluation.",
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_tool(
    ToolDefinition(
        tool_name="professor.rubric_evaluator.v1",
        namespace=ToolNamespace.PROFESSOR,
        version="v1",
        description=(
            "Assistive rubric evaluation for professors. Returns provisional band, "
            "criterion-linked justification, and evidence quotes grounded in the "
            "submission text. LLM-assisted when available; heuristic fallback otherwise."
        ),
        allowed_roles=frozenset({ToolRole.PROFESSOR, ToolRole.ADMIN}),
        risk_level=RiskLevel.MEDIUM,
        enabled=True,
        timeout_seconds=40.0,
        supports_idempotency=True,
        safe_for_multi_step=True,
        input_model=RubricInput,
        output_model=RubricOutput,
        handler=_handle,
    )
)
