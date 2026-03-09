from __future__ import annotations
import re
import html

_ws = re.compile(r"\s+")
_tags = re.compile(r"<[^>]+>")

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = html.unescape(s)
    s = _tags.sub(" ", s)           # remove HTML tags
    s = s.replace("\u00a0", " ")    # NBSP
    s = _ws.sub(" ", s).strip()     # normalize whitespace
    return s
