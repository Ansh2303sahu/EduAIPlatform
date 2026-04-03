import anthropic
from typing import Any, Dict, Mapping, Optional, Sequence, Union

from anthropic.types import ContentBlock, TextBlock
from app.config import settings

ANTHROPIC_API_KEY = str(settings.anthropic_api_key or "")
ANTHROPIC_PRIMARY_MODEL = str(settings.anthropic_primary_model or "claude-sonnet-4-6")
ANTHROPIC_FALLBACK_MODEL = str(settings.anthropic_fallback_model or "claude-haiku-4-5-20251001")
ANTHROPIC_MAX_TOKENS = int(settings.anthropic_max_tokens or 4096)


def _err(e: Exception) -> str:
    s = str(e)
    return f"{type(e).__name__}: {s if s else repr(e)}"


def _extract_prompt(payload: Union[Dict[str, Any], str, Mapping[str, Any]]) -> str:
    """Extract the prompt string from an Ollama-style payload or plain string."""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, Mapping):
        return str(payload.get("prompt", ""))
    raise TypeError(f"payload must be a dict/mapping or str, got {type(payload).__name__}")


def _normalize_result(text: str, model_used: str) -> Dict[str, Any]:
    """Return the same shape as ollama_client._normalize_result."""
    return {
        "model_used": model_used,
        "response": text,
        "done": True,
        "done_reason": "stop",
        "raw": {"content": [{"text": text}], "model": model_used},
    }


def _extract_text_blocks(content: Sequence[ContentBlock]) -> str:
    """Join text blocks from an Anthropic message content array."""
    text_parts: list[str] = []
    for block in content:
        if isinstance(block, TextBlock):
            text_parts.append(block.text)

    text = "".join(text_parts).strip()
    if text:
        return text
    raise RuntimeError("Anthropic response did not contain any text blocks")


async def anthropic_generate_json(model: str, prompt: str) -> str:
    """Call the Anthropic Messages API and return the response text."""
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    message = await client.messages.create(
        model=model,
        max_tokens=ANTHROPIC_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return _extract_text_blocks(message.content)


async def generate_with_fallback(payload: Union[Dict[str, Any], str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Try primary Claude model, fall back to secondary on failure.

    Drop-in replacement for ollama_client.generate_with_fallback — same
    input/output contract so main.py needs no logic changes.
    """
    prompt = _extract_prompt(payload)

    e1: Optional[Exception] = None
    e2: Optional[Exception] = None

    try:
        text = await anthropic_generate_json(ANTHROPIC_PRIMARY_MODEL, prompt)
        return _normalize_result(text, ANTHROPIC_PRIMARY_MODEL)
    except Exception as e:
        e1 = e

    try:
        text = await anthropic_generate_json(ANTHROPIC_FALLBACK_MODEL, prompt)
        return _normalize_result(text, ANTHROPIC_FALLBACK_MODEL)
    except Exception as e:
        e2 = e

    raise RuntimeError(
        f"anthropic primary ({ANTHROPIC_PRIMARY_MODEL}) failed: {_err(e1)} | "
        f"fallback ({ANTHROPIC_FALLBACK_MODEL}) failed: {_err(e2)}"
    )
