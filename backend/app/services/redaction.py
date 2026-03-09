from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Tuple


EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4}\b")
# You can add your own student-id pattern here if you have a known format.
STUDENT_ID_RE = re.compile(r"\b(?:student\s*id[:\s]*)?(\d{6,12})\b", re.IGNORECASE)


@dataclass
class RedactionResult:
    redacted_text: str
    summary: Dict[str, int]


def redact_pii(text: str) -> RedactionResult:
    summary: Dict[str, int] = {"emails": 0, "phones": 0, "student_ids": 0}

    def _sub_count(pattern: re.Pattern, repl: str, key: str, s: str) -> str:
        matches = list(pattern.finditer(s))
        summary[key] += len(matches)
        return pattern.sub(repl, s)

    out = text
    out = _sub_count(EMAIL_RE, "[REDACTED_EMAIL]", "emails", out)
    out = _sub_count(PHONE_RE, "[REDACTED_PHONE]", "phones", out)
    out = _sub_count(STUDENT_ID_RE, "[REDACTED_ID]", "student_ids", out)

    return RedactionResult(redacted_text=out, summary=summary)
