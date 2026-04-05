"""
MCP Tool: student.structure_improver.v1

Analyses a student submission for structural gaps using a two-pass approach:
1. Deterministic heading/keyword detection (always runs).
2. Optional LLM refinement for semantic section identification.

Purpose: assistive feedback only — never grades, never rejects submissions.

Input  : ``StructureInput``
Output : ``StructureOutput``
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

_SUBMISSION_TYPES = Literal["essay", "report", "dissertation", "project_report"]


class StructureInput(BaseModel):
    model_config = {"extra": "forbid"}

    text: str = Field(..., min_length=1, max_length=8_000)
    submission_type: _SUBMISSION_TYPES = "essay"
    expected_sections: list[str] = Field(default_factory=list, max_length=20)


class StructureOutput(BaseModel):
    model_config = {"extra": "forbid"}

    detected_sections: list[str]
    missing_sections: list[str]
    suggested_order: list[str]
    improvement_notes: list[str]
    confidence_note: str


# ---------------------------------------------------------------------------
# Section definitions per submission type
# ---------------------------------------------------------------------------

_CANONICAL_ORDER: dict[str, list[str]] = {
    "essay": [
        "Introduction", "Background", "Thesis Statement", "Main Argument",
        "Evidence & Analysis", "Discussion", "Conclusion", "References",
    ],
    "report": [
        "Abstract", "Introduction", "Literature Review", "Methodology",
        "Results", "Discussion", "Conclusion", "Recommendations", "References",
    ],
    "dissertation": [
        "Abstract", "Acknowledgements", "Introduction", "Literature Review",
        "Methodology", "Results", "Discussion", "Conclusion", "Bibliography",
    ],
    "project_report": [
        "Introduction", "Requirements", "Architecture", "Implementation",
        "Testing", "Evaluation", "Conclusion", "References",
    ],
}

# Keyword → section name mapping for deterministic detection.
_SECTION_KEYWORDS: dict[str, list[str]] = {
    "Introduction": ["introduction", "background", "overview"],
    "Abstract": ["abstract", "executive summary"],
    "Literature Review": ["literature review", "related work", "prior work"],
    "Methodology": ["methodology", "method", "approach", "research design"],
    "Results": ["results", "findings", "data analysis"],
    "Discussion": ["discussion", "analysis", "interpretation"],
    "Conclusion": ["conclusion", "summary", "closing remarks"],
    "References": ["references", "bibliography", "works cited"],
    "Testing": ["testing", "test plan", "unit test", "acceptance test"],
    "Architecture": ["architecture", "system design", "design"],
    "Implementation": ["implementation", "features built", "development"],
    "Requirements": ["requirements", "user stories", "specification"],
    "Evaluation": ["evaluation", "limitations", "future work"],
    "Recommendations": ["recommendations", "suggestions", "future"],
    "Thesis Statement": ["thesis statement", "thesis:"],
    "Evidence & Analysis": ["evidence", "analysis", "critical analysis"],
    "Main Argument": ["argument", "main point"],
    "Acknowledgements": ["acknowledgements", "acknowledgments"],
}


def _detect_sections(text: str) -> list[str]:
    """Return section names whose keywords appear in the text (case-insensitive)."""
    text_lower = text.lower()
    detected: list[str] = []
    for section_name, keywords in _SECTION_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            detected.append(section_name)
    return detected


def _detect_sections_from_headings(text: str) -> list[str]:
    """
    Detect section headings by looking for lines with heading-like formatting
    (short lines, title case, or Markdown headers).
    """
    heading_re = re.compile(
        r"^(?:#{1,3}\s+)?([A-Z][A-Za-z &\-]{2,60})$",
        re.MULTILINE,
    )
    headings = [m.group(1).strip() for m in heading_re.finditer(text)]
    return list(dict.fromkeys(headings))[:15]  # de-dupe, cap at 15


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
You are an academic writing advisor. Analyse the following {submission_type} for structural completeness.

Expected sections for a {submission_type}: {expected_sections}

Rules:
- Return ONLY a JSON object with keys: "detected_sections", "missing_sections", "suggested_order", "improvement_notes".
- Each value is a JSON array of strings.
- "detected_sections": section names actually found in the text.
- "missing_sections": expected section names NOT found in the text.
- "suggested_order": all section names in the recommended reading order.
- "improvement_notes": 2-4 concise, actionable, non-grading suggestions.
- Do NOT invent sections unrelated to a {submission_type}.
- Start with {{ and end with }}. No prose outside the JSON.

TEXT:
{text}
"""


def _build_prompt(input: StructureInput) -> str:
    expected = (
        input.expected_sections
        if input.expected_sections
        else _CANONICAL_ORDER.get(input.submission_type, [])
    )
    return _PROMPT_TEMPLATE.format(
        submission_type=input.submission_type.replace("_", " "),
        expected_sections=", ".join(expected),
        text=input.text[:5_000],
    )


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

async def _handle(input: StructureInput, ctx: ToolExecutionContext) -> HandlerResult:
    from app.mcp.config import mcp_settings

    # Always run deterministic detection — it seeds the fallback and validates LLM output.
    canonical = (
        input.expected_sections
        if input.expected_sections
        else _CANONICAL_ORDER.get(input.submission_type, [])
    )
    det_by_keywords = _detect_sections(input.text)
    det_by_headings = _detect_sections_from_headings(input.text)
    det_combined = list(dict.fromkeys(det_by_keywords + det_by_headings))

    if mcp_settings.llm_enabled:
        llm_result = await call_llm(
            _build_prompt(input),
            role="student",
            temperature=0.2,
            tool_name="student.structure_improver.v1",
        )
        parsed = parse_json_response(llm_result)

        if parsed.ok:
            detected = [str(s).strip() for s in parsed.data.get("detected_sections", []) if s]
            missing = [str(s).strip() for s in parsed.data.get("missing_sections", []) if s]
            order = [str(s).strip() for s in parsed.data.get("suggested_order", []) if s]
            notes = [str(s).strip() for s in parsed.data.get("improvement_notes", []) if s]

            if detected or missing:
                return HandlerResult(
                    output=StructureOutput(
                        detected_sections=detected[:15],
                        missing_sections=missing[:10],
                        suggested_order=order[:15] or canonical,
                        improvement_notes=notes[:5],
                        confidence_note=(
                            "Structural analysis assisted by language model. "
                            "This is advisory guidance - not a grade."
                        ),
                    ),
                    llm_used=True,
                    llm_fallback_used=llm_result.used_fallback,
                    model_used=llm_result.model_used,
                    deterministic_fallback=False,
                    confidence_note="LLM-assisted structure analysis.",
                )

    # Deterministic fallback
    missing = [s for s in canonical if not any(
        kw.lower() in input.text.lower()
        for kw in _SECTION_KEYWORDS.get(s, [s.lower()])
    )]
    notes: list[str] = []
    for s in missing[:4]:
        notes.append(
            f"Consider adding a '{s}' section to strengthen your {input.submission_type} structure."
        )
    if not missing:
        notes.append(
            f"The submission covers the expected sections for a {input.submission_type}."
        )

    return HandlerResult(
        output=StructureOutput(
            detected_sections=det_combined,
            missing_sections=missing,
            suggested_order=canonical,
            improvement_notes=notes,
            confidence_note=(
                "Structural analysis based on keyword detection. "
                "LLM semantic refinement was not available."
            ),
        ),
        llm_used=False,
        deterministic_fallback=True,
        confidence_note="Keyword-based structure detection.",
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_tool(
    ToolDefinition(
        tool_name="student.structure_improver.v1",
        namespace=ToolNamespace.STUDENT,
        version="v1",
        description=(
            "Analyses a student submission for structural completeness against "
            "expected sections for the given submission type. Returns detected sections, "
            "missing sections, suggested ordering, and improvement notes."
        ),
        allowed_roles=frozenset({ToolRole.STUDENT, ToolRole.ADMIN}),
        risk_level=RiskLevel.LOW,
        enabled=True,
        timeout_seconds=35.0,
        supports_idempotency=True,
        safe_for_multi_step=True,
        input_model=StructureInput,
        output_model=StructureOutput,
        handler=_handle,
    )
)
