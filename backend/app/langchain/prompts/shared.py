"""
Shared prompt helpers for the Phase 10 LangChain pipelines.

This module only contains low-level formatting and common guardrails. Student
and professor task instructions remain in their own modules so the two roles
stay fully separated.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence


SHARED_RULE_LINES = (
    "Use only the provided evidence from the submission, Phase 6 predictions, and approved retrieved context.",
    "Never invent citations, sources, rubric rules, policies, or missing evidence.",
    "Never claim certainty when the evidence is weak, conflicting, or incomplete.",
    "Return JSON only.",
    "Do not wrap the output in markdown fences.",
    "Do not include commentary outside the required schema.",
    "The response must start with { and end with }.",
    "Use double quotes for all JSON keys and string values.",
    "Do not add keys that are not in the schema.",
)

WEAK_EVIDENCE_LINES = (
    "If evidence is limited, say so inside the relevant JSON fields rather than guessing.",
    "If a section is not supported by the evidence, use a conservative fallback such as \"Not assessed.\" or \"Evidence is limited.\"",
    "Only use citation indices that appear in the retrieved context section.",
)


def _field(source: Any, key: str, default: Any = None) -> Any:
    """Safe attribute/key access for dicts and objects."""
    if isinstance(source, dict):
        return source.get(key, default)
    try:
        return getattr(source, key, default)
    except Exception:
        return default


def _clean_text(value: Any, limit: int) -> str:
    """Normalise arbitrary values into a compact prompt-safe text preview."""
    if value is None:
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _pretty_json(value: Any) -> str:
    """Serialise a value to indented JSON for prompt embedding."""
    try:
        return json.dumps(value, indent=2, sort_keys=True)
    except TypeError:
        return json.dumps(str(value), indent=2)


def render_section(title: str, body: str) -> str:
    """Render a named prompt section."""
    body = body.strip()
    if not body:
        return ""
    return f"{title}\n{body}"


def render_bullets(title: str, lines: Sequence[str]) -> str:
    """Render a bullet list section."""
    cleaned = [str(line).strip() for line in lines if str(line).strip()]
    if not cleaned:
        return ""
    body = "\n".join(f"- {line}" for line in cleaned)
    return render_section(title, body)


def join_sections(*sections: Optional[str]) -> str:
    """Join prompt sections with blank lines, dropping empties."""
    return "\n\n".join(section.strip() for section in sections if section and section.strip())


def build_shared_rules_section(*, extra_lines: Optional[Sequence[str]] = None) -> str:
    """Build the common non-hallucination and JSON-only rules block."""
    lines = list(SHARED_RULE_LINES)
    if extra_lines:
        lines.extend(str(line).strip() for line in extra_lines if str(line).strip())
    return render_bullets("SHARED RULES", lines)


def build_weak_evidence_section(*, extra_lines: Optional[Sequence[str]] = None) -> str:
    """Build the common conservative-behaviour block for uncertain inputs."""
    lines = list(WEAK_EVIDENCE_LINES)
    if extra_lines:
        lines.extend(str(line).strip() for line in extra_lines if str(line).strip())
    return render_bullets("WHEN EVIDENCE IS WEAK", lines)


def build_submission_summary_section(ingestion: Any, *, submission_kind: str) -> str:
    """Render a compact extracted-summary section from ingestion outputs."""
    text_preview = _clean_text(_field(ingestion, "text_content", ""), 360)
    ocr_preview = _clean_text(_field(ingestion, "ocr_text", ""), 220)
    transcript_preview = _clean_text(_field(ingestion, "audio_transcript", ""), 220)
    tables = _field(ingestion, "tables_json", None)

    available_sources: list[str] = []
    if text_preview:
        available_sources.append("text")
    if ocr_preview:
        available_sources.append("ocr")
    if transcript_preview:
        available_sources.append("transcript")
    if tables:
        available_sources.append("tables")

    lines = [
        f"Submission kind: {submission_kind or 'unknown'}",
        f"Available extracted sources: {', '.join(available_sources) if available_sources else 'none'}",
        f"Primary extracted summary: {text_preview or 'No primary extracted text was available.'}",
    ]
    if ocr_preview:
        lines.append(f"OCR summary: {ocr_preview}")
    if transcript_preview:
        lines.append(f"Transcript summary: {transcript_preview}")
    if tables:
        if isinstance(tables, Mapping):
            table_keys = ", ".join(list(tables.keys())[:6]) or "table data present"
            lines.append(f"Table summary: extracted table data present ({table_keys}).")
        else:
            lines.append("Table summary: extracted table data present.")

    return render_bullets("EXTRACTED SUBMISSION SUMMARY", lines)


def build_submission_evidence_section(
    ingestion: Any,
    *,
    text_limit: int = 8000,
    ocr_limit: int = 2500,
    transcript_limit: int = 2500,
    table_limit: int = 2500,
) -> str:
    """Render the raw extracted evidence block for the model to quote from."""
    blocks: list[str] = []

    text_content = str(_field(ingestion, "text_content", "") or "").strip()
    ocr_text = str(_field(ingestion, "ocr_text", "") or "").strip()
    transcript = str(_field(ingestion, "audio_transcript", "") or "").strip()
    tables = _field(ingestion, "tables_json", None)

    if text_content:
        blocks.append(f"[TEXT]\n{text_content[:text_limit]}")
    if ocr_text:
        blocks.append(f"[OCR]\n{ocr_text[:ocr_limit]}")
    if transcript:
        blocks.append(f"[TRANSCRIPT]\n{transcript[:transcript_limit]}")
    if tables:
        blocks.append(f"[TABLES]\n{_pretty_json(tables)[:table_limit]}")

    if not blocks:
        blocks.append("(No extracted submission evidence was available.)")

    return render_section("SUBMISSION EVIDENCE", "\n\n".join(blocks))


def build_ml_predictions_section(
    title: str,
    *,
    ml_context_text: str = "",
    ml_raw: Optional[Mapping[str, Any]] = None,
) -> str:
    """Render Phase 6 prediction signals for prompt injection."""
    parts: list[str] = []
    if ml_context_text.strip():
        parts.append(ml_context_text.strip())
    if ml_raw:
        parts.append("Raw prediction object:\n" + _pretty_json(dict(ml_raw)))
    if not parts:
        parts.append("No Phase 6 prediction signals were available.")
    return render_section(title, "\n\n".join(parts))


def build_rag_section(
    *,
    title: str,
    context: Optional[str],
    instruction: Optional[str],
    citations: Optional[Sequence[Any]] = None,
    retrieved_chunks: Optional[Sequence[Any]] = None,
    confidence_label: Optional[str] = None,
    confidence_score: Optional[float] = None,
    safe_review: Optional[bool] = None,
) -> str:
    """Render the approved retrieved context block."""
    if not context:
        return render_section(
            title,
            "\n".join(
                [
                    "No approved retrieved context was provided.",
                    "Do not invent external guidance, policies, citations, or rubric language.",
                    "Rely only on submission evidence and the available Phase 6 predictions.",
                ]
            ),
        )

    citation_lines = []
    for fallback_idx, item in enumerate(citations or [], start=1):
        idx = _field(item, "index", fallback_idx)
        source = _field(item, "title", _field(item, "source", "Untitled source"))
        section = _field(item, "section", "unknown section")
        citation_lines.append(f"[{idx}] {source} | section: {section}")
    if not citation_lines:
        citation_lines.append("No citation metadata was supplied.")

    chunk_lines = []
    for idx, chunk in enumerate((retrieved_chunks or [])[:4], start=1):
        source = _field(chunk, "document_title", _field(chunk, "source", "Untitled source"))
        section = _field(chunk, "section", "unknown section")
        score = _field(chunk, "score", "unknown")
        chunk_lines.append(f"[Chunk {idx}] {source} | section: {section} | score: {score}")
    if not chunk_lines:
        chunk_lines.append("No retrieved chunk metadata was supplied.")

    review_flag = bool(safe_review)
    confidence_value = confidence_score if confidence_score is not None else 0.0

    return render_section(
        title,
        "\n".join(
            [
                "Approved retrieved context:",
                context.strip(),
                "",
                "Allowed inline citations:",
                *citation_lines,
                "",
                "Retrieved chunk signals:",
                *chunk_lines,
                "",
                "Retrieval confidence:",
                f"- label: {confidence_label or 'unknown'}",
                f"- score: {confidence_value}",
                f"- safe_review: {review_flag}",
                "",
                "Retrieved-context instructions:",
                instruction.strip()
                if instruction and instruction.strip()
                else "Use the retrieved context conservatively and never cite unsupported material.",
                "Only cite indices that appear in the allowed citation list above.",
            ]
        ),
    )


def build_output_schema_section(
    title: str,
    schema_example: Mapping[str, Any],
    *,
    field_rules: Optional[Sequence[str]] = None,
) -> str:
    """Render the expected JSON schema contract section."""
    body_parts = [
        "Return exactly one JSON object that matches this shape:",
        _pretty_json(dict(schema_example)),
    ]
    if field_rules:
        body_parts.extend(
            [
                "",
                "Field rules:",
                *[f"- {line}" for line in field_rules if str(line).strip()],
            ]
        )
    return render_section(title, "\n".join(body_parts))
