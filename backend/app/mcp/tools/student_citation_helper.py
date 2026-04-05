"""
MCP Tool: student.citation_helper.v1

Citation-needed guidance for student submissions.

Deterministic pass:
- flags likely citation-needed sentences and paragraphs
- detects visible citation markers to reduce false positives
- produces citation density and style-pattern warnings

Optional LLM refinement:
- re-prioritises deterministic candidates only
- never invents sources or bibliographic entries
- falls back cleanly to deterministic output on any failure

This tool is assistive only. It does not verify truth, detect plagiarism,
or generate authoritative references.
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

_CITATION_STYLE = Literal["apa", "harvard", "ieee", "generic"]
_SENSITIVITY = Literal["low", "medium", "high"]
_SEVERITY = Literal["low", "medium", "high"]

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_PAREN_CITATION_RE = re.compile(
    r"\(([A-Z][A-Za-z'`\-]+(?:\s+[A-Z][A-Za-z'`\-]+)*(?:\s+et al\.)?,\s*(?:19|20)\d{2}[a-z]?)"
)
_NARRATIVE_CITATION_RE = re.compile(
    r"\b[A-Z][A-Za-z'`\-]+(?:\s+et al\.)?\s*\((?:19|20)\d{2}[a-z]?\)"
)
_IEEE_CITATION_RE = re.compile(r"\[(?:\d{1,3})(?:\s*,\s*\d{1,3})*\]")
_NUMERIC_CLAIM_RE = re.compile(
    r"\b(?:\d{4}|\d+(?:\.\d+)?%|\d+(?:\.\d+)?\s*(?:percent|per cent|million|billion|thousand))\b",
    re.IGNORECASE,
)
_GENERAL_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_TRIGGER_PHRASES = (
    "research shows",
    "studies indicate",
    "studies show",
    "experts say",
    "it is widely known",
    "according to",
    "evidence suggests",
    "data shows",
    "reports indicate",
    "the literature suggests",
)
_SOURCE_LIKE_TERMS = (
    "developed",
    "introduced",
    "published",
    "reported",
    "demonstrated",
    "established",
    "recommended",
    "defined",
    "standard",
    "framework",
    "algorithm",
    "protocol",
    "historically",
    "government",
    "policy",
    "ranked",
)
_MAX_EXCERPT_CHARS = 220

_SENSITIVITY_PARAGRAPH_WORDS = {
    "low": 170,
    "medium": 130,
    "high": 95,
}

_SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}


class CitationHelperInput(BaseModel):
    model_config = {"extra": "forbid"}

    text: str = Field(..., min_length=1, max_length=10_000)
    citation_style: _CITATION_STYLE = "generic"
    max_flags: int = Field(default=5, ge=1, le=12)
    sensitivity: _SENSITIVITY = "medium"


class FlaggedSegment(BaseModel):
    model_config = {"extra": "forbid"}

    text_excerpt: str = Field(..., min_length=1, max_length=_MAX_EXCERPT_CHARS)
    reason: str = Field(..., min_length=1, max_length=400)
    severity: _SEVERITY
    suggested_action: str = Field(..., min_length=1, max_length=300)


class CitationHelperOutput(BaseModel):
    model_config = {"extra": "forbid"}

    flagged_segments: list[FlaggedSegment]
    citation_density_note: str = Field(..., min_length=1, max_length=300)
    style_warnings: list[str]
    warnings: list[str]
    confidence_note: str = Field(..., min_length=1, max_length=300)


def _clean_excerpt(text: str) -> str:
    compact = " ".join(text.split())
    if len(compact) <= _MAX_EXCERPT_CHARS:
        return compact
    return compact[: _MAX_EXCERPT_CHARS - 3].rstrip() + "..."


def _split_sentences(text: str) -> list[str]:
    compact = " ".join(text.split())
    return [s.strip() for s in _SENTENCE_RE.split(compact) if s.strip()]


def _split_paragraphs(text: str) -> list[str]:
    blocks = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if blocks:
        return blocks
    compact = " ".join(text.split())
    if not compact:
        return []
    return [compact]


def _contains_visible_citation(text: str) -> bool:
    return any(
        pattern.search(text)
        for pattern in (
            _PAREN_CITATION_RE,
            _NARRATIVE_CITATION_RE,
            _IEEE_CITATION_RE,
        )
    )


def _citation_counts(text: str) -> dict[str, int]:
    return {
        "author_year_parenthetical": len(_PAREN_CITATION_RE.findall(text)),
        "author_year_narrative": len(_NARRATIVE_CITATION_RE.findall(text)),
        "ieee": len(_IEEE_CITATION_RE.findall(text)),
    }


def _style_specific_action(citation_style: _CITATION_STYLE) -> str:
    if citation_style == "apa":
        return "Add an APA-style author-date citation for this claim."
    if citation_style == "harvard":
        return "Add a Harvard-style author-date citation for this claim."
    if citation_style == "ieee":
        return "Add an IEEE-style numbered citation for this claim."
    return "Add a supporting citation in your required referencing style."


def _raise_severity(current: _SEVERITY, candidate: _SEVERITY) -> _SEVERITY:
    return candidate if _SEVERITY_ORDER[candidate] > _SEVERITY_ORDER[current] else current


def _add_flag(
    store: dict[str, FlaggedSegment],
    *,
    excerpt: str,
    reason: str,
    severity: _SEVERITY,
    suggested_action: str,
) -> None:
    normalized_excerpt = _clean_excerpt(excerpt)
    key = normalized_excerpt.lower()
    existing = store.get(key)
    if existing is None:
        store[key] = FlaggedSegment(
            text_excerpt=normalized_excerpt,
            reason=reason,
            severity=severity,
            suggested_action=suggested_action,
        )
        return

    merged_reason = existing.reason
    if reason not in merged_reason:
        merged_reason = f"{merged_reason} {reason}"
    store[key] = FlaggedSegment(
        text_excerpt=existing.text_excerpt,
        reason=merged_reason[:400],
        severity=_raise_severity(existing.severity, severity),
        suggested_action=existing.suggested_action,
    )


def _looks_numeric_claim(sentence: str) -> bool:
    lower = sentence.lower()
    if _NUMERIC_CLAIM_RE.search(lower):
        return True
    if _GENERAL_NUMBER_RE.search(lower) and any(
        token in lower
        for token in ("rank", "survey", "increase", "decrease", "growth", "rate", "cases")
    ):
        return True
    return False


def _looks_source_like(sentence: str) -> bool:
    lower = sentence.lower()
    return any(term in lower for term in _SOURCE_LIKE_TERMS)


def _build_style_warnings(
    text: str, citation_style: _CITATION_STYLE, word_count: int
) -> list[str]:
    counts = _citation_counts(text)
    author_year_total = counts["author_year_parenthetical"] + counts["author_year_narrative"]
    ieee_total = counts["ieee"]
    warnings: list[str] = []

    if citation_style in {"apa", "harvard"} and ieee_total > 0:
        warnings.append(
            "Bracketed numeric citations were detected; check that this matches your selected citation style."
        )
    if citation_style == "ieee" and author_year_total > 0:
        warnings.append(
            "Author-year citation patterns were detected; IEEE usually expects bracketed numeric citations."
        )
    if citation_style == "generic" and author_year_total > 0 and ieee_total > 0:
        warnings.append(
            "Mixed citation patterns were detected. Use one citation style consistently."
        )
    if word_count >= 180 and author_year_total + ieee_total == 0:
        warnings.append(
            "No visible citation markers were detected in a substantial passage."
        )
    return warnings


def _citation_density_note(text: str, sensitivity: _SENSITIVITY) -> str:
    word_count = len(text.split())
    counts = _citation_counts(text)
    citation_total = sum(counts.values())
    if word_count < 80:
        return "The text is short, so citation-density guidance is limited."
    if citation_total == 0:
        return "No visible citation markers were detected across the analysed text."

    words_per_citation = round(word_count / citation_total, 1)
    if words_per_citation > 190 or (
        sensitivity == "high" and words_per_citation > 150
    ):
        return (
            "Visible citation density appears sparse for the length of the text "
            f"(about {words_per_citation} words per citation marker)."
        )
    return (
        "Visible citation density looks reasonable overall, but some individual "
        "claims may still need clearer attribution."
    )


def _heuristic_flags(input: CitationHelperInput) -> tuple[list[FlaggedSegment], list[str], str]:
    flags: dict[str, FlaggedSegment] = {}
    action = _style_specific_action(input.citation_style)

    for sentence in _split_sentences(input.text):
        if len(sentence.split()) < 7 or _contains_visible_citation(sentence):
            continue

        lower = sentence.lower()
        reasons: list[tuple[str, _SEVERITY]] = []
        if any(phrase in lower for phrase in _TRIGGER_PHRASES):
            reasons.append(
                (
                    "This sentence refers to research, evidence, or expert opinion without a visible citation.",
                    "high" if input.sensitivity == "high" else "medium",
                )
            )
        if _looks_numeric_claim(sentence):
            reasons.append(
                (
                    "This sentence contains a factual, numerical, or time-based claim without a visible citation.",
                    "medium",
                )
            )
        if _looks_source_like(sentence):
            reasons.append(
                (
                    "This sentence reads like a source-based technical or historical assertion but has no visible citation.",
                    "medium" if input.sensitivity == "low" else "high",
                )
            )

        if reasons:
            severity: _SEVERITY = "low"
            merged_reason_parts: list[str] = []
            for reason, reason_severity in reasons:
                merged_reason_parts.append(reason)
                severity = _raise_severity(severity, reason_severity)
            _add_flag(
                flags,
                excerpt=sentence,
                reason=" ".join(dict.fromkeys(merged_reason_parts)),
                severity=severity,
                suggested_action=action,
            )

    long_paragraph_threshold = _SENSITIVITY_PARAGRAPH_WORDS[input.sensitivity]
    for paragraph in _split_paragraphs(input.text):
        if _contains_visible_citation(paragraph):
            continue
        if len(paragraph.split()) < long_paragraph_threshold:
            continue
        _add_flag(
            flags,
            excerpt=paragraph,
            reason=(
                "This long paragraph does not show a visible citation pattern, so readers may struggle "
                "to tell which claims are supported by sources."
            ),
            severity="medium" if input.sensitivity != "low" else "low",
            suggested_action=(
                "Add citations to the source-dependent claims in this paragraph, especially where evidence, "
                "background facts, or historical context are introduced."
            ),
        )

    style_warnings = _build_style_warnings(
        input.text,
        input.citation_style,
        len(input.text.split()),
    )
    density_note = _citation_density_note(input.text, input.sensitivity)

    ordered = sorted(
        flags.values(),
        key=lambda item: (
            _SEVERITY_ORDER[item.severity],
            len(item.text_excerpt),
        ),
        reverse=True,
    )
    return ordered[: input.max_flags], style_warnings, density_note


_LLM_PROMPT = """\
You are assisting with citation-needed guidance for a student submission.

Important boundaries:
- Suggest where citations may be needed.
- Do NOT invent references, authors, titles, URLs, DOIs, or publication details.
- Do NOT claim factual verification.
- Do NOT claim plagiarism detection.
- Use ONLY the candidate excerpts provided below.

Citation style: {citation_style}
Max flags: {max_flags}
Sensitivity: {sensitivity}

Return ONLY a JSON object with keys:
- "flagged_segments": array of objects with keys "text_excerpt", "reason", "severity", "suggested_action"
- "citation_density_note": string
- "style_warnings": array of strings

Rules:
- Choose at most {max_flags} candidate excerpts.
- "text_excerpt" must exactly match one of the candidate excerpts.
- Keep reasons concise and grounded in citation-needed guidance only.
- Suggested actions may describe citation patterns, but must not generate references.
- Start with {{ and end with }}. No prose outside the JSON.

Candidate excerpts:
{candidate_lines}

Current citation density note:
{density_note}

Current style warnings:
{style_warnings}
"""


def _build_llm_prompt(
    input: CitationHelperInput,
    candidate_flags: list[FlaggedSegment],
    style_warnings: list[str],
    density_note: str,
) -> str:
    candidate_lines = "\n".join(
        (
            f"- excerpt: {flag.text_excerpt}\n"
            f"  heuristic_reason: {flag.reason}\n"
            f"  heuristic_severity: {flag.severity}\n"
            f"  suggested_action: {flag.suggested_action}"
        )
        for flag in candidate_flags[:8]
    )
    warnings_line = "; ".join(style_warnings) if style_warnings else "None"
    return _LLM_PROMPT.format(
        citation_style=input.citation_style,
        max_flags=input.max_flags,
        sensitivity=input.sensitivity,
        candidate_lines=candidate_lines or "- excerpt: None",
        density_note=density_note,
        style_warnings=warnings_line,
    )


def _llm_refined_output(
    input: CitationHelperInput,
    candidate_flags: list[FlaggedSegment],
    parsed_data: dict[str, object],
    fallback_density_note: str,
    fallback_style_warnings: list[str],
) -> CitationHelperOutput | None:
    allowed_by_excerpt = {flag.text_excerpt: flag for flag in candidate_flags}
    refined_segments: list[FlaggedSegment] = []
    raw_segments = parsed_data.get("flagged_segments")
    if isinstance(raw_segments, list):
        for item in raw_segments[: input.max_flags]:
            if not isinstance(item, dict):
                continue
            excerpt = str(item.get("text_excerpt") or "").strip()
            if excerpt not in allowed_by_excerpt:
                continue
            severity = str(item.get("severity") or allowed_by_excerpt[excerpt].severity).lower()
            if severity not in {"low", "medium", "high"}:
                severity = allowed_by_excerpt[excerpt].severity
            reason = str(item.get("reason") or allowed_by_excerpt[excerpt].reason).strip()
            suggested_action = str(
                item.get("suggested_action") or allowed_by_excerpt[excerpt].suggested_action
            ).strip()
            if not reason or not suggested_action:
                continue
            refined_segments.append(
                FlaggedSegment(
                    text_excerpt=excerpt,
                    reason=reason[:400],
                    severity=severity,  # type: ignore[arg-type]
                    suggested_action=suggested_action[:300],
                )
            )

    if not refined_segments:
        return None

    style_warnings = [
        str(item).strip()
        for item in (parsed_data.get("style_warnings") or [])
        if str(item).strip()
    ][:6]
    density_note = str(parsed_data.get("citation_density_note") or fallback_density_note).strip()
    if not density_note:
        density_note = fallback_density_note

    confidence_note = (
        "Citation guidance was prioritised with language-model assistance. "
        "It suggests where citations may be needed, but it does not verify truth "
        "or generate authoritative references."
    )

    return CitationHelperOutput(
        flagged_segments=refined_segments[: input.max_flags],
        citation_density_note=density_note[:300],
        style_warnings=style_warnings or fallback_style_warnings,
        warnings=[],
        confidence_note=confidence_note,
    )


async def _handle(input: CitationHelperInput, ctx: ToolExecutionContext) -> HandlerResult:
    from app.mcp.config import mcp_settings

    warnings = [
        "This tool suggests where citations may be needed; it does not verify factual truth.",
        "This tool does not generate authoritative references automatically or perform plagiarism detection.",
    ]

    heuristic_flags, style_warnings, density_note = _heuristic_flags(input)
    confidence_note = (
        "Heuristic citation review based on visible citation patterns and source-like claims. "
        "Suggestions are advisory only."
    )

    if mcp_settings.llm_enabled and heuristic_flags:
        llm_result = await call_llm(
            _build_llm_prompt(input, heuristic_flags, style_warnings, density_note),
            role="student",
            temperature=min(mcp_settings.llm_temperature, 0.15),
            tool_name="student.citation_helper.v1",
        )
        parsed = parse_json_response(llm_result)
        if parsed.ok:
            refined = _llm_refined_output(
                input,
                heuristic_flags,
                parsed.data,
                density_note,
                style_warnings,
            )
            if refined is not None:
                refined.warnings.extend(warnings)
                return HandlerResult(
                    output=refined,
                    llm_used=True,
                    llm_fallback_used=llm_result.used_fallback,
                    model_used=llm_result.model_used,
                    deterministic_fallback=False,
                    confidence_note=refined.confidence_note,
                )
            warnings.append("LLM refinement returned no usable prioritisation. Using heuristic output.")
        else:
            warnings.append(
                f"LLM refinement unavailable ({parsed.error_reason}). Using heuristic output."
            )

    return HandlerResult(
        output=CitationHelperOutput(
            flagged_segments=heuristic_flags,
            citation_density_note=density_note,
            style_warnings=style_warnings,
            warnings=warnings,
            confidence_note=confidence_note,
        ),
        llm_used=False,
        deterministic_fallback=True,
        confidence_note=confidence_note,
    )


register_tool(
    ToolDefinition(
        tool_name="student.citation_helper.v1",
        namespace=ToolNamespace.STUDENT,
        version="v1",
        description=(
            "Highlights submission segments that may need citations, reports citation-pattern issues, "
            "and suggests citation-focused next steps without generating references."
        ),
        allowed_roles=frozenset({ToolRole.STUDENT, ToolRole.ADMIN}),
        risk_level=RiskLevel.LOW,
        enabled=True,
        timeout_seconds=35.0,
        supports_idempotency=True,
        safe_for_multi_step=False,
        input_model=CitationHelperInput,
        output_model=CitationHelperOutput,
        handler=_handle,
    )
)
