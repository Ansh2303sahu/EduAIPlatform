"""
Shared prompt helpers for the Phase 10 LangChain pipelines.

This module only contains low-level formatting and common guardrails. Student
and professor task instructions remain in their own modules so the two roles
stay fully separated.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence

from app.langchain.config import phase10_settings


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


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(values: Any, *, limit: int, item_limit: int) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return []
    items: list[str] = []
    for item in values:
        text = _clean_text(item, item_limit)
        if text:
            items.append(text)
        if len(items) >= limit:
            break
    return items


def _joined_text(values: Sequence[str]) -> str:
    return " ".join(part for part in values if part).strip()


def _pretty_json(value: Any) -> str:
    """Serialise a value to indented JSON for prompt embedding."""
    try:
        return json.dumps(value, indent=2, sort_keys=True)
    except TypeError:
        return json.dumps(str(value), indent=2)


def _compact_json(value: Any) -> str:
    """Serialise a value to compact JSON for prompt embedding."""
    try:
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    except TypeError:
        return json.dumps(str(value), separators=(",", ":"))


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


def _paragraphs(text: str) -> list[str]:
    raw_parts = [
        _clean_text(part, 900)
        for part in str(text or "").replace("\r", "\n").split("\n")
    ]
    parts = [part for part in raw_parts if part]
    if parts:
        return parts

    sentences = [
        _clean_text(part, 900)
        for part in str(text or "").split(". ")
    ]
    return [part for part in sentences if part]


def _best_excerpt(paragraphs: Sequence[str], *, index: int) -> str:
    if not paragraphs:
        return ""
    bounded = max(0, min(len(paragraphs) - 1, index))
    return _clean_text(paragraphs[bounded], min(420, phase10_settings.prompt_submission_text_chars // 4))


def _evidence_excerpt(paragraphs: Sequence[str]) -> str:
    evidence_terms = ("because", "for example", "evidence", "data", "reference", "test", "result", "finding")
    ranked = sorted(
        paragraphs,
        key=lambda part: (sum(term in part.lower() for term in evidence_terms), len(part)),
        reverse=True,
    )
    return _best_excerpt(ranked, index=0)


def build_assignment_context_section(
    *,
    file_metadata: Mapping[str, Any] | None,
    submission_kind: str,
    ingestion: Any,
) -> str:
    metadata = dict(file_metadata or {})
    title_hint = _clean_text(
        metadata.get("assignment_title")
        or metadata.get("title")
        or metadata.get("file_name")
        or _best_excerpt(_paragraphs(_field(ingestion, "text_content", "")), index=0),
        180,
    )
    module_hint = _clean_text(metadata.get("module") or metadata.get("course"), 120)
    level_hint = _clean_text(metadata.get("academic_level") or metadata.get("level"), 80)
    word_target = _clean_text(metadata.get("word_count_target") or metadata.get("word_limit"), 60)
    rubric_hint = _clean_text(metadata.get("rubric_summary") or metadata.get("rubric_text"), 320)

    lines = [
        f"Assignment title: {title_hint or 'Not available'}",
        f"Assignment type: {submission_kind or 'unknown'}",
    ]
    if module_hint:
        lines.append(f"Module or course: {module_hint}")
    if level_hint:
        lines.append(f"Academic level: {level_hint}")
    if word_target:
        lines.append(f"Word target: {word_target}")
    if rubric_hint:
        lines.append(f"Rubric summary: {rubric_hint}")
    return render_bullets("ASSIGNMENT CONTEXT", lines)


def build_submission_digest_section(ingestion: Any, *, submission_kind: str) -> str:
    paragraphs = _paragraphs(_field(ingestion, "text_content", ""))
    digest = _best_excerpt(paragraphs, index=0)
    if len(paragraphs) >= 3:
        digest = _joined_text([_best_excerpt(paragraphs, index=0), _best_excerpt(paragraphs, index=len(paragraphs) // 2)])

    lines = [
        f"Submission type profile: {submission_kind or 'unknown'}",
        f"Submission digest: {digest or 'No compact digest could be derived from the extracted text.'}",
        f"Extracted text length: {len(str(_field(ingestion, 'text_content', '') or ''))} characters",
    ]
    return render_bullets("SUBMISSION DIGEST", lines)


def build_representative_excerpts_section(ingestion: Any) -> str:
    paragraphs = _paragraphs(_field(ingestion, "text_content", ""))
    if not paragraphs:
        return render_section(
            "REPRESENTATIVE EXCERPTS",
            "No representative text excerpts were available from the extracted submission.",
        )

    intro = _best_excerpt(paragraphs, index=0)
    middle = _best_excerpt(paragraphs, index=len(paragraphs) // 2)
    conclusion = _best_excerpt(paragraphs, index=len(paragraphs) - 1)
    evidence = _evidence_excerpt(paragraphs)

    blocks = [
        f"[INTRO OR OPENING]\n{intro}",
    ]
    if middle and middle != intro:
        blocks.append(f"[MIDDLE OR CORE SECTION]\n{middle}")
    if evidence and evidence not in {intro, middle}:
        blocks.append(f"[EVIDENCE-RICH EXCERPT]\n{evidence}")
    if conclusion and conclusion not in {intro, middle, evidence}:
        blocks.append(f"[CONCLUSION OR FINAL SECTION]\n{conclusion}")

    return render_section("REPRESENTATIVE EXCERPTS", "\n\n".join(blocks))


def build_submission_summary_section(ingestion: Any, *, submission_kind: str) -> str:
    """Render a compact extracted-summary section from ingestion outputs."""
    text_preview = _clean_text(
        _field(ingestion, "text_content", ""),
        phase10_settings.prompt_submission_summary_chars,
    )
    ocr_preview = _clean_text(_field(ingestion, "ocr_text", ""), min(220, phase10_settings.prompt_ocr_chars))
    transcript_preview = _clean_text(
        _field(ingestion, "audio_transcript", ""),
        min(220, phase10_settings.prompt_transcript_chars),
    )
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
    text_limit: int | None = None,
    ocr_limit: int | None = None,
    transcript_limit: int | None = None,
    table_limit: int | None = None,
) -> str:
    """Render the raw extracted evidence block for the model to quote from."""
    blocks: list[str] = []
    text_limit = text_limit or phase10_settings.prompt_submission_text_chars
    ocr_limit = ocr_limit or phase10_settings.prompt_ocr_chars
    transcript_limit = transcript_limit or phase10_settings.prompt_transcript_chars
    table_limit = table_limit or phase10_settings.prompt_table_chars

    text_content = str(_field(ingestion, "text_content", "") or "").strip()
    ocr_text = str(_field(ingestion, "ocr_text", "") or "").strip()
    transcript = str(_field(ingestion, "audio_transcript", "") or "").strip()
    tables = _field(ingestion, "tables_json", None)

    if text_content:
        blocks.append(f"[TEXT EXCERPT]\n{_clean_text(text_content, text_limit)}")
    if ocr_text:
        blocks.append(f"[OCR EXCERPT]\n{_clean_text(ocr_text, ocr_limit)}")
    if transcript:
        blocks.append(f"[TRANSCRIPT EXCERPT]\n{_clean_text(transcript, transcript_limit)}")
    if tables:
        blocks.append(f"[TABLE SIGNALS]\n{_clean_text(_pretty_json(tables), table_limit)}")

    if not blocks:
        blocks.append("(No extracted submission evidence was available.)")

    return render_section("SUBMISSION EVIDENCE", "\n\n".join(blocks))


def build_ml_predictions_section(
    title: str,
    *,
    ml_context_text: str = "",
    ml_raw: Optional[Mapping[str, Any]] = None,
    include_raw: bool = False,
) -> str:
    """Render Phase 6 prediction signals for prompt injection."""
    parts: list[str] = []
    if ml_context_text.strip():
        parts.append(ml_context_text.strip())
    if include_raw and ml_raw:
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
    trace: Optional[Mapping[str, Any]] = None,
    context_limit: int | None = None,
    citation_limit: int | None = None,
    chunk_preview_limit: int | None = None,
    chunk_preview_chars: int | None = None,
) -> str:
    """Render the approved retrieved context block."""
    trace_data = dict(trace or {})
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
    citation_limit = max(1, citation_limit or phase10_settings.prompt_rag_citation_limit)
    for fallback_idx, item in enumerate((citations or [])[:citation_limit], start=1):
        idx = _field(item, "index", fallback_idx)
        source = _field(item, "title", _field(item, "source", "Untitled source"))
        section = _field(item, "section", "unknown section")
        citation_lines.append(f"[{idx}] {source} | section: {section}")
    if not citation_lines:
        citation_lines.append("No citation metadata was supplied.")

    chunk_lines = []
    chunk_preview_limit = max(1, chunk_preview_limit or phase10_settings.prompt_rag_chunk_preview_limit)
    chunk_preview_chars = max(80, chunk_preview_chars or phase10_settings.prompt_rag_chunk_preview_chars)
    for idx, chunk in enumerate((retrieved_chunks or [])[:chunk_preview_limit], start=1):
        source = _field(chunk, "document_title", _field(chunk, "source", "Untitled source"))
        section = _field(chunk, "section", "unknown section")
        category = _field(chunk, "category", "unknown")
        score = _field(chunk, "score", "unknown")
        chunk_lines.append(
            f"[Chunk {idx}] {source} | section: {section} | category: {category} | score: {score}"
        )
    if not chunk_lines:
        chunk_lines.append("No retrieved chunk metadata was supplied.")

    review_flag = bool(safe_review)
    confidence_value = confidence_score if confidence_score is not None else 0.0
    trace_lines: list[str] = []
    if trace_data.get("mode"):
        trace_lines.append(f"- mode: {_clean_text(trace_data.get('mode'), 80)}")
    keywords = _string_list(trace_data.get("keywords_used"), limit=8, item_limit=60)
    if keywords:
        trace_lines.append(f"- keywords_used: {', '.join(keywords)}")
    selected_titles = _string_list(trace_data.get("selected_titles"), limit=5, item_limit=100)
    if selected_titles:
        trace_lines.append(f"- selected_titles: {', '.join(selected_titles)}")
    final_categories = _string_list(trace_data.get("final_categories"), limit=6, item_limit=60)
    if final_categories:
        trace_lines.append(f"- final_categories: {', '.join(final_categories)}")
    if trace_data.get("degraded_input"):
        trace_lines.append("- degraded_input: true")

    context_block = _clean_text(context.strip(), context_limit or phase10_settings.prompt_rag_context_chars)

    return render_section(
        title,
        "\n".join(
            [
                "Approved retrieved context:",
                context_block,
                "",
                "Allowed inline citations:",
                *citation_lines,
                "",
                "Retrieved chunk signals:",
                *chunk_lines,
                *(
                    ["", "Retrieval trace summary:", *trace_lines]
                    if trace_lines
                    else []
                ),
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
        _compact_json(dict(schema_example)),
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
