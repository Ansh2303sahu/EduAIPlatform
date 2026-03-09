import re
from typing import Tuple

# Only detect instruction-style prompt injection.
# Avoid flagging normal document vocabulary like "password", "token", etc.
_PATTERNS = [
    r"\bignore\s+(all\s+|the\s+)?(previous|prior)\s+instructions\b",
    r"\bignore\s+above\b",
    r"\bforget\s+(all\s+|the\s+)?previous\b",
    r"\b(system prompt|developer message|hidden instructions)\b",
    r"\breveal\s+(the\s+)?(system prompt|developer message|hidden instructions)\b",
    r"\bshow\s+(the\s+)?(system prompt|developer message|hidden instructions)\b",
    r"\bprint\s+(the\s+)?(system prompt|developer message|hidden instructions)\b",
    r"\bdisplay\s+(the\s+)?(system prompt|developer message|hidden instructions)\b",
    r"\bexfiltrate\b",
    r"\bbypass\b.{0,40}\b(safety|guardrails|filters|restrictions)\b",
    r"\bdo not follow\b.{0,40}\binstructions\b",
    r"\bact as\b.{0,40}\b(system|developer|admin)\b",
    r"\bpretend to be\b.{0,40}\b(system|developer|admin)\b",
    r"\bcat\s+\.env\b",
    r"\bprintenv\b",
]

_REPEAT_RE = re.compile(r"(.)\1{20,}", re.DOTALL)


def sanitize_input(text: str, max_chars: int) -> Tuple[str, bool, str]:
    if not text:
        return "", False, ""

    text = text[:max_chars]
    text = _REPEAT_RE.sub(lambda m: m.group(1) * 5, text)

    lowered = text.lower()
    for pat in _PATTERNS:
        if re.search(pat, lowered, flags=re.IGNORECASE | re.DOTALL):
            return text, True, f"prompt_injection:{pat}"

    return text, False, ""