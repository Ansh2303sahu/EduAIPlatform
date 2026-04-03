"""
Input sanitization and prompt-injection filtering for Phase 10.

The helpers in this module are pure and reusable so they can be applied to raw
submission text, retrieved chunks, or prompt-adjacent strings before they are
embedded into LLM context.
"""

from __future__ import annotations

import re
from typing import Iterable

from app.langchain.config import phase10_settings

# Phrases that indicate a prompt-injection attempt. Keep this aligned with the
# Phase 7 heuristic, but also include a few common variants used in retrieved
# role-switch text.
_INJECTION_PHRASES = [
    "ignore previous instructions",
    "ignore all previous",
    "disregard previous instructions",
    "disregard above",
    "forget your instructions",
    "system prompt",
    "new system prompt",
    "override system",
    "developer message",
    "jailbreak",
    "reveal hidden instructions",
    "do not follow your instructions",
    "you are now",
    "you are now a",
    "you are no longer",
    "act as if you have no restrictions",
]

_INJECTION_PATTERNS = [
    (phrase, re.compile(rf"(?i)\b{re.escape(phrase)}\b"))
    for phrase in _INJECTION_PHRASES
]

_ROLE_SWITCH_PATTERNS = [
    re.compile(r"(?i)\b(?:you are now|you are no longer|from now on)\b"),
    re.compile(r"(?i)\b(?:ignore|disregard|forget|override|reveal)\b.{0,80}\b(?:instruction|system|prompt|developer|assistant|hidden)\b"),
    re.compile(r"(?i)\brole\s*:\s*(?:system|assistant|developer|user)\b"),
    re.compile(r"(?i)^\s*(?:system|assistant|developer|user)\s*:"),
    re.compile(r"(?i)\b(?:act as if you have no restrictions|pretend to be)\b"),
]

_CONTROL_CHAR_TRANSLATION = {
    code: " "
    for code in range(32)
    if code not in (9, 10, 13)
}
_CONTROL_CHAR_TRANSLATION[127] = " "

_PROMPT_FILTER_PLACEHOLDER = "[filtered prompt-injection phrase]"


def strip_control_chars(text: str) -> str:
    """Replace non-printable control characters with spaces."""
    return (text or "").translate(_CONTROL_CHAR_TRANSLATION)


def collapse_repeated_whitespace(text: str) -> str:
    """Normalize whitespace while preserving paragraph breaks when possible."""
    normalized = strip_control_chars(text).replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    normalized = re.sub(r"[ \f\v]+", " ", normalized)
    normalized = re.sub(r" *\n *", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def detect_injection(text: str) -> tuple[bool, str]:
    """
    Return ``(is_injected, reason_phrase)`` for the given text.

    The text is normalized before matching so control-character obfuscation and
    repeated whitespace do not bypass the heuristic.
    """
    normalized = collapse_repeated_whitespace(text).lower()
    for phrase, pattern in _INJECTION_PATTERNS:
        if pattern.search(normalized):
            return True, phrase
    return False, ""


def filter_prompt_injection_phrases(text: str) -> tuple[str, list[str]]:
    """
    Replace known prompt-injection phrases with a neutral placeholder.

    This preserves surrounding legitimate academic text instead of deleting the
    whole sentence whenever possible.
    """
    sanitized = collapse_repeated_whitespace(text)
    matched: list[str] = []
    for phrase, pattern in _INJECTION_PATTERNS:
        if pattern.search(sanitized):
            matched.append(phrase)
            sanitized = pattern.sub(_PROMPT_FILTER_PLACEHOLDER, sanitized)
    sanitized = collapse_repeated_whitespace(sanitized)
    return sanitized, matched


def sanitize_plain_text(
    text: str,
    *,
    max_chars: int | None = None,
    filter_injection_phrases: bool = False,
) -> str:
    """Sanitize a plain-text field while keeping legitimate content intact."""
    limit = max_chars or phase10_settings.max_input_chars
    sanitized = collapse_repeated_whitespace(text)
    if filter_injection_phrases:
        sanitized, _ = filter_prompt_injection_phrases(sanitized)
    return sanitized[:limit].rstrip()


def sanitize_text_list(
    texts: Iterable[str],
    *,
    max_chars: int | None = None,
    filter_injection_phrases: bool = False,
    drop_empty: bool = True,
) -> list[str]:
    """Sanitize a list of plain-text strings."""
    cleaned = [
        sanitize_plain_text(
            text,
            max_chars=max_chars,
            filter_injection_phrases=filter_injection_phrases,
        )
        for text in texts
    ]
    if drop_empty:
        return [item for item in cleaned if item]
    return cleaned


def sanitize_retrieved_text(text: str, *, max_chars: int | None = None) -> str:
    """
    Sanitize retrieved context text.

    This applies phrase filtering and removes entire lines that look like
    role-switching or prompt-instruction content.
    """
    limit = max_chars or phase10_settings.max_input_chars
    normalized = strip_control_chars(text).replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines: list[str] = []
    for raw_line in normalized.split("\n"):
        line = collapse_repeated_whitespace(raw_line)
        if not line:
            continue
        if any(pattern.search(line) for pattern in _ROLE_SWITCH_PATTERNS):
            continue
        line, _ = filter_prompt_injection_phrases(line)
        if line and line != _PROMPT_FILTER_PLACEHOLDER:
            cleaned_lines.append(line)

    sanitized = collapse_repeated_whitespace("\n".join(cleaned_lines))
    return sanitized[:limit].rstrip()


def sanitize_retrieved_list(
    texts: Iterable[str],
    *,
    max_chars: int | None = None,
    drop_empty: bool = True,
) -> list[str]:
    """Sanitize a list of retrieved-text strings."""
    cleaned = [sanitize_retrieved_text(text, max_chars=max_chars) for text in texts]
    if drop_empty:
        return [item for item in cleaned if item]
    return cleaned


def sanitize_input(text: str, max_chars: int | None = None) -> tuple[str, bool, str]:
    """
    Sanitize a user-controlled input string and detect prompt injection.

    Returns ``(sanitized_text, is_injected, injection_reason)``.
    """
    limit = max_chars or phase10_settings.max_input_chars
    normalized = collapse_repeated_whitespace(text)[:limit]
    injected, reason = detect_injection(normalized)
    sanitized = sanitize_plain_text(
        normalized,
        max_chars=limit,
        filter_injection_phrases=True,
    )
    return sanitized, injected, reason
