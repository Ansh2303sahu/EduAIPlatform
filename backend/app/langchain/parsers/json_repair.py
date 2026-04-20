"""
JSON extraction and repair utilities for Phase 10 parser stages.

This keeps the cheap extraction logic from ``llm-service/main.py`` but adds a
local best-effort repair pass so the backend can recover from common LLM JSON
mistakes before asking a fallback model to repair the output remotely.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any


_FENCED_BLOCK_RE = re.compile(
    r"```(?:json|javascript|js)?\s*(.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)
_TRAILING_COMMA_RE = re.compile(r",(?=\s*[}\]])")
_BARE_KEY_RE = re.compile(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_\- ]*?)(\s*:)')
_BARE_VALUE_RE = re.compile(r'(:\s*)([A-Za-z_][A-Za-z0-9_\- /]*?)(\s*[,}\]])')
_SMART_QUOTES = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2032": "'",
        "\u2033": '"',
    }
)


def _as_text(raw: Any) -> str:
    return raw if isinstance(raw, str) else str(raw)


def _strip_outer_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json|javascript|js)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _iter_search_spaces(text: str) -> list[str]:
    stripped = _strip_outer_fence(text)
    candidates: list[str] = []

    for match in _FENCED_BLOCK_RE.finditer(text):
        block = match.group(1).strip()
        if block:
            candidates.append(block)

    if stripped:
        candidates.append(stripped)
    if text.strip() and text.strip() != stripped:
        candidates.append(text.strip())

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique


def _looks_like_json_object(candidate: str) -> bool:
    compact = candidate.strip()
    return compact.startswith("{") and ":" in compact


def _find_first_likely_json_object(text: str) -> str | None:
    start: int | None = None
    depth = 0
    quote_char: str | None = None
    escaped = False

    for idx, ch in enumerate(text):
        if quote_char is not None:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote_char:
                quote_char = None
            continue

        if ch in {'"', "'"}:
            quote_char = ch
            continue

        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
            continue

        if ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start : idx + 1].strip()
                if _looks_like_json_object(candidate):
                    return candidate
                start = None

    if start is not None:
        candidate = text[start:].strip()
        if _looks_like_json_object(candidate):
            return candidate

    return None


def _normalize_quotes(text: str) -> str:
    return text.translate(_SMART_QUOTES).lstrip("\ufeff").strip()


def _remove_trailing_commas(text: str) -> str:
    return _TRAILING_COMMA_RE.sub("", text)


def _balance_json_delimiters(text: str) -> str:
    stack: list[str] = []
    quote_char: str | None = None
    escaped = False

    for ch in text:
        if quote_char is not None:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote_char:
                quote_char = None
            continue

        if ch in {'"', "'"}:
            quote_char = ch
            continue

        if ch in "{[":
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()

    closers = {"{": "}", "[": "]"}
    return text + "".join(closers[ch] for ch in reversed(stack))


def _quote_bare_keys(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(2).strip()
        return f'{match.group(1)}{json.dumps(key)}{match.group(3)}'

    return _BARE_KEY_RE.sub(repl, text)


def _quote_bare_word_values(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        raw_value = match.group(2).strip()
        lowered = raw_value.lower()
        if lowered in {"true", "false", "null", "none"}:
            return f"{match.group(1)}{raw_value}{match.group(3)}"
        return f"{match.group(1)}{json.dumps(raw_value)}{match.group(3)}"

    return _BARE_VALUE_RE.sub(repl, text)


def _replace_tokens_outside_strings(text: str, replacements: dict[str, str]) -> str:
    result: list[str] = []
    token: list[str] = []
    quote_char: str | None = None
    escaped = False

    def flush() -> None:
        if not token:
            return
        word = "".join(token)
        replacement = replacements.get(word)
        if replacement is None:
            replacement = replacements.get(word.lower(), word)
        result.append(replacement)
        token.clear()

    for ch in text:
        if quote_char is not None:
            flush()
            result.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote_char:
                quote_char = None
            continue

        if ch in {'"', "'"}:
            flush()
            quote_char = ch
            result.append(ch)
            continue

        if ch.isalpha():
            token.append(ch)
            continue

        flush()
        result.append(ch)

    flush()
    return "".join(result)


def _next_significant_char(text: str, start: int) -> str:
    idx = max(start, 0)
    while idx < len(text):
        ch = text[idx]
        if not ch.isspace():
            return ch
        idx += 1
    return ""


def _next_token_looks_like_bare_key(text: str, start: int) -> bool:
    idx = max(start, 0)
    while idx < len(text) and text[idx].isspace():
        idx += 1
    if idx >= len(text):
        return False
    if not (text[idx].isalpha() or text[idx] == "_"):
        return False

    idx += 1
    while idx < len(text) and (text[idx].isalnum() or text[idx] in {"_", "-", " "}):
        idx += 1
    while idx < len(text) and text[idx].isspace():
        idx += 1
    return idx < len(text) and text[idx] == ":"


def _next_token_looks_like_quoted_key(text: str, start: int) -> bool:
    idx = max(start, 0)
    while idx < len(text) and text[idx].isspace():
        idx += 1
    if idx >= len(text) or text[idx] != '"':
        return False

    idx += 1
    escaped = False
    while idx < len(text):
        ch = text[idx]
        if escaped:
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == '"':
            idx += 1
            break
        idx += 1

    while idx < len(text) and text[idx].isspace():
        idx += 1
    return idx < len(text) and text[idx] == ":"


def _escape_inner_quotes(text: str) -> str:
    """Escape quote characters that look like literal content inside JSON strings."""

    result: list[str] = []
    in_string = False
    escaped = False

    for idx, ch in enumerate(text):
        if not in_string:
            result.append(ch)
            if ch == '"':
                in_string = True
            continue

        if escaped:
            result.append(ch)
            escaped = False
            continue

        if ch == "\\":
            result.append(ch)
            escaped = True
            continue

        if ch == '"':
            next_char = _next_significant_char(text, idx + 1)
            if (
                next_char in {":", ",", "}", "]"}
                or not next_char
                or _next_token_looks_like_bare_key(text, idx + 1)
                or _next_token_looks_like_quoted_key(text, idx + 1)
            ):
                result.append(ch)
                in_string = False
            else:
                result.append('\\"')
            continue

        result.append(ch)

    return "".join(result)


def _parse_json_dict(text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, str(exc)
    if not isinstance(parsed, dict):
        return None, f"Expected JSON object, got {type(parsed).__name__}"
    return parsed, None


def _parse_python_like_dict(text: str) -> tuple[dict[str, Any] | None, str | None]:
    python_like = _replace_tokens_outside_strings(
        text,
        {
            "true": "True",
            "false": "False",
            "null": "None",
        },
    )
    try:
        parsed = ast.literal_eval(python_like)
    except (ValueError, SyntaxError) as exc:
        return None, str(exc)
    if not isinstance(parsed, dict):
        return None, f"Expected object-like payload, got {type(parsed).__name__}"
    return parsed, None


def extract_json_text(raw: str) -> str:
    """
    Extract the first likely JSON object from raw LLM output.

    Handles prose before/after the object and prefers fenced JSON blocks when
    present. Raises ``ValueError`` when no object-like content is found.
    """
    text = _as_text(raw).strip()
    for candidate in _iter_search_spaces(text):
        extracted = _find_first_likely_json_object(candidate)
        if extracted:
            return extracted
    raise ValueError(f"No JSON object found in LLM output: {text[:200]!r}")


def repair_json_text(raw: str) -> str:
    """
    Repair common malformed JSON patterns and return a JSON string.

    The repair path targets common LLM issues: fenced code blocks, smart
    quotes, Python-style dicts, bare keys, trailing commas, and truncated
    closing delimiters.
    """
    text = _as_text(raw)
    try:
        candidate = extract_json_text(text)
    except ValueError:
        candidate = _strip_outer_fence(text)

    base = _normalize_quotes(candidate)
    variants: list[str] = []

    def add_variant(value: str) -> None:
        cleaned = value.strip()
        if cleaned and cleaned not in variants:
            variants.append(cleaned)

    add_variant(base)
    add_variant(_remove_trailing_commas(base))
    balanced = _balance_json_delimiters(_remove_trailing_commas(base))
    add_variant(balanced)
    escaped = _escape_inner_quotes(balanced)
    add_variant(escaped)
    add_variant(_quote_bare_keys(escaped))
    add_variant(_quote_bare_word_values(_quote_bare_keys(escaped)))

    last_error = "Unknown JSON repair failure."
    for variant in variants:
        parsed, err = _parse_json_dict(variant)
        if parsed is not None:
            return json.dumps(parsed, ensure_ascii=False)
        if err:
            last_error = err

        parsed, err = _parse_python_like_dict(variant)
        if parsed is not None:
            return json.dumps(parsed, ensure_ascii=False)
        if err:
            last_error = err

    if variants:
        return variants[-1]
    raise ValueError(last_error)


def build_repair_prompt(bad_output: str, target: str = "student") -> str:
    """
    Build a follow-up prompt asking the LLM to fix broken JSON.

    ``target`` should be ``"student"``, ``"student_project_review"``,
    or ``"professor"``.
    """
    return (
        f"The previous JSON output was invalid.\n\n"
        f"BAD OUTPUT:\n{bad_output[:3000]}\n\n"
        f"Fix the JSON so it is valid and matches the expected {target} report schema.\n"
        f"Return ONLY the corrected JSON object.\n"
        f"The first character must be {{.\n"
        f"The last character must be }}.\n"
        f"Do not include markdown fences or any prose.\n"
    )


def try_parse_json(text: str) -> tuple[dict[str, Any], None] | tuple[None, str]:
    """
    Parse raw LLM output into a JSON object.

    Runs cheap extraction first, then a local repair pass for common malformed
    JSON issues. Returns ``(parsed_dict, None)`` on success or ``(None, err)``
    on failure.
    """
    errors: list[str] = []

    try:
        extracted = extract_json_text(text)
        parsed, err = _parse_json_dict(extracted)
        if parsed is not None:
            return parsed, None
        if err:
            errors.append(err)
    except ValueError as exc:
        errors.append(str(exc))

    try:
        repaired = repair_json_text(text)
    except ValueError as exc:
        errors.append(str(exc))
        return None, "; ".join(errors)

    parsed, err = _parse_json_dict(repaired)
    if parsed is not None:
        return parsed, None
    if err:
        errors.append(err)

    parsed, err = _parse_python_like_dict(repaired)
    if parsed is not None:
        return parsed, None
    if err:
        errors.append(err)

    return None, "; ".join(errors)
