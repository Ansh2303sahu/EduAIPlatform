from __future__ import annotations

import httpx
from app.core.config import settings


def _parser_base() -> str:
    if not settings.parser_url:
        raise RuntimeError("PARSER_URL is not configured")
    return settings.parser_url.rstrip("/")


async def parse_text_via_parser(*, filename: str, file_bytes: bytes) -> dict:
    """
    Calls parser sandbox to extract text.
    IMPORTANT: parser endpoint is /parse/text (NOT /)
    """
    url = f"{_parser_base()}/parse/text"

    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(
            url,
            headers={"X-PARSER-SECRET": settings.parser_secret},
            files={"file": (filename, file_bytes)},
        )

    if resp.status_code >= 300:
        raise RuntimeError(f"Parser text extraction failed: {resp.status_code} {resp.text}")

    return resp.json() if resp.content else {}
