from __future__ import annotations

import re
from typing import Any, Mapping


_GENERIC_PHRASES = (
    "this report combines submission evidence",
    "mixed signals",
    "useful next improvements",
    "some parts need sharper support",
    "automated review generated with limited confidence",
    "manual review recommended",
)

_PLACEHOLDER_PHRASES = (
    "not assessed.",
    "evidence is limited.",
    "detailed feedback explanation unavailable.",
)

_ACTION_MARKERS = (
    "revise",
    "clarify",
    "add",
    "explain",
    "support",
    "strengthen",
    "tighten",
    "restructure",
    "test",
    "justify",
    "compare",
    "cite",
    "reference",
    "document",
)

_SECTION_MARKERS = (
    "introduction",
    "body",
    "conclusion",
    "paragraph",
    "section",
    "argument",
    "analysis",
    "evidence",
    "reference",
    "reflection",
    "method",
    "discussion",
    "code",
    "function",
    "module",
    "class",
    "component",
    "test",
)

_SYSTEM_CENTRIC_MARKERS = (
    "the system",
    "the platform",
    "automated",
    "model",
    "llm",
    "generation",
    "pipeline",
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_SPACE_RE = re.compile(r"\s+")


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
    else:
        text = str(value).strip()
    return _SPACE_RE.sub(" ", text)


def _joined_text(values: list[Any]) -> str:
    return " ".join(part for part in (_text(value) for value in values) if part).strip()


def _sentence_fingerprints(text: str) -> list[str]:
    cleaned = _text(text).lower()
    if not cleaned:
        return []
    parts = []
    for sentence in _SENTENCE_SPLIT_RE.split(cleaned):
        fingerprint = re.sub(r"[^a-z0-9 ]+", "", sentence).strip()
        if fingerprint:
            parts.append(fingerprint)
    return parts


def _count_phrase_hits(text: str, phrases: tuple[str, ...]) -> int:
    lowered = _text(text).lower()
    return sum(1 for phrase in phrases if phrase in lowered)


def _list_of_mappings(value: Any) -> list[dict[str, Any]]:
    return [_as_dict(item) for item in _as_list(value) if _as_dict(item)]


def _object_text(*values: Any) -> str:
    return _joined_text(list(values))


def extract_priority_issue(report: Mapping[str, Any] | None) -> dict[str, Any]:
    data = _as_dict(report)
    priority_issue = _as_dict(data.get("priority_issue"))
    if priority_issue:
        return {
            "title": _text(priority_issue.get("title")),
            "why_it_matters": _text(priority_issue.get("why_it_matters")),
            "how_to_fix_it": _text(priority_issue.get("how_to_fix_it")),
        }

    issues = _list_of_mappings(data.get("issues")) or _list_of_mappings(data.get("weaknesses")) or _list_of_mappings(data.get("concerns"))
    if not issues:
        return {}

    ranked = sorted(
        issues,
        key=lambda item: (
            {"high": 2, "med": 1, "medium": 1, "low": 0}.get(_text(item.get("severity")).lower(), 1),
            len(_object_text(item.get("evidence"), item.get("detail"), item.get("description"))),
        ),
        reverse=True,
    )
    top = ranked[0]
    return {
        "title": _text(top.get("title") or top.get("risk") or top.get("issue")),
        "why_it_matters": _text(top.get("evidence") or top.get("detail") or top.get("description")),
        "how_to_fix_it": "",
    }


def extract_best_summary(role: str, report: Mapping[str, Any] | None) -> str:
    data = _as_dict(report)
    role_name = _text(role).lower()
    candidates = []
    if role_name == "student":
        candidates.extend(
            [
                data.get("summary"),
                data.get("overall_judgment"),
                _as_dict(data.get("priority_issue")).get("title"),
                data.get("confidence_explanation"),
            ]
        )
    else:
        candidates.extend(
            [
                data.get("summary"),
                data.get("evaluator_overview"),
                data.get("feedback_explanation"),
                data.get("moderation_notes"),
            ]
        )

    for candidate in candidates:
        text = _text(candidate)
        if not text:
            continue
        if _count_phrase_hits(text, _GENERIC_PHRASES) >= 1 and len(text) < 120:
            continue
        return text
    return ""


def build_report_preview(role: str, report: Mapping[str, Any] | None) -> dict[str, Any]:
    data = _as_dict(report)
    strengths = _as_list(data.get("strengths"))
    weaknesses = _as_list(data.get("weaknesses")) or _as_list(data.get("issues")) or _as_list(data.get("concerns"))
    checklist = _as_list(data.get("checklist"))
    preview_items: list[str] = []
    for item in checklist[:3]:
        text = _text(_as_dict(item).get("item") if isinstance(item, Mapping) else item)
        if text:
            preview_items.append(text[:140])
    return {
        "summary": extract_best_summary(role, data),
        "priority_issue": extract_priority_issue(data),
        "strengths_count": len(strengths),
        "weaknesses_count": len(weaknesses),
        "checklist_preview": preview_items,
    }


def report_richness_score(role: str, report: Mapping[str, Any] | None) -> int:
    data = _as_dict(report)
    score = 0
    summary = extract_best_summary(role, data)
    if len(summary) >= 80:
        score += 2

    strengths = _as_list(data.get("strengths"))
    weaknesses = _as_list(data.get("weaknesses")) or _as_list(data.get("issues")) or _as_list(data.get("concerns"))
    section_feedback = _as_list(data.get("section_feedback")) or _as_list(data.get("section_observations"))
    actions = _as_list(data.get("improvement_plan")) or _as_list(data.get("action_recommendations"))
    checklist = _as_list(data.get("checklist"))

    score += min(len(strengths), 3)
    score += min(len(weaknesses), 3)
    score += min(len(section_feedback), 3)
    score += min(len(actions), 3)
    score += 1 if checklist else 0
    score += 1 if _text(data.get("confidence_explanation") or data.get("evidence_coverage") or data.get("grounding_summary")) else 0

    if role == "professor":
        score += 1 if _as_list(data.get("rubric_alignment")) else 0
        score += 1 if _as_list(data.get("marking_considerations")) else 0
    else:
        score += 1 if _text(data.get("overall_judgment")) else 0
        score += 1 if _as_dict(data.get("priority_issue")) else 0
    return score


def report_low_content_quality(role: str, report: Mapping[str, Any] | None) -> bool:
    data = _as_dict(report)
    if not data:
        return False

    summary = extract_best_summary(role, data)
    flattened = _joined_text(
        [
            summary,
            data.get("feedback_explanation"),
            data.get("overall_judgment"),
            data.get("evaluator_overview"),
            data.get("confidence_explanation"),
            data.get("evidence_coverage"),
            data.get("grounding_summary"),
            data.get("moderation_notes"),
        ]
    )
    repeated_sentences = _sentence_fingerprints(flattened)
    duplicate_sentence_ratio = 0.0
    if repeated_sentences:
        duplicate_sentence_ratio = 1.0 - (len(set(repeated_sentences)) / max(1, len(repeated_sentences)))

    generic_hits = _count_phrase_hits(flattened, _GENERIC_PHRASES)
    placeholder_hits = _count_phrase_hits(flattened, _PLACEHOLDER_PHRASES)
    system_hits = _count_phrase_hits(flattened, _SYSTEM_CENTRIC_MARKERS)
    action_hits = _count_phrase_hits(flattened, _ACTION_MARKERS)
    section_hits = _count_phrase_hits(flattened, _SECTION_MARKERS)

    issue_like = _list_of_mappings(data.get("issues")) or _list_of_mappings(data.get("weaknesses")) or _list_of_mappings(data.get("concerns"))
    detail_rows = 0
    for item in issue_like[:4]:
        if _text(item.get("evidence") or item.get("detail") or item.get("description")):
            detail_rows += 1

    actionable_items = _as_list(data.get("improvement_plan")) or _as_list(data.get("action_recommendations")) or _as_list(data.get("checklist"))
    richness = report_richness_score(role, data)

    if len(summary) < 50 and not actionable_items and detail_rows == 0:
        return True
    if generic_hits >= 2 and detail_rows == 0:
        return True
    if duplicate_sentence_ratio >= 0.45 and richness < 6:
        return True
    if action_hits == 0 and not actionable_items and richness < 6:
        return True
    if section_hits == 0 and len(summary) < 140 and richness < 5:
        return True
    if system_hits >= 2 and placeholder_hits >= 1 and richness < 6:
        return True
    return False
