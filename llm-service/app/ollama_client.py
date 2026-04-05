import json
import logging
import httpx
from typing import Any, Dict, Mapping, Optional, Union

from app.config import settings

logger = logging.getLogger("llm_service.ollama_client")

OLLAMA_URL = str(settings.ollama_base_url or "http://host.docker.internal:11434").rstrip("/")
OLLAMA_TIMEOUT_S = float(settings.timeout_seconds or 180)
OLLAMA_PRIMARY_MODEL = str(settings.primary_model or "mistral:latest")
OLLAMA_FALLBACK_MODEL = str(settings.fallback_model or OLLAMA_PRIMARY_MODEL)
OLLAMA_OPTIONS_JSON = str(settings.ollama_options_json or "").strip()

logger.info(
    "ollama client configured primary=%s fallback=%s base_url=%s",
    OLLAMA_PRIMARY_MODEL,
    OLLAMA_FALLBACK_MODEL,
    OLLAMA_URL,
)


def _err(e: Exception) -> str:
    s = str(e)
    return f"{type(e).__name__}: {s if s else repr(e)}"


def _as_mapping_payload(payload: Union[Dict[str, Any], str, Mapping[str, Any]]) -> Dict[str, Any]:
    if isinstance(payload, Mapping):
        return dict(payload)

    if isinstance(payload, str):
        s = payload.strip()

        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return parsed
                return {"prompt": payload, "stream": False}
            except json.JSONDecodeError:
                return {"prompt": payload, "stream": False}

        return {"prompt": payload, "stream": False}

    raise TypeError(f"payload must be a dict/mapping or str, got {type(payload).__name__}: {payload!r}")


_STABLE_OUTPUT_DEFAULTS: Dict[str, Any] = {
    # Low temperature for deterministic structured output
    "temperature": 0.1,
    # Enough tokens for a full student/professor report
    "num_predict": 4096,
    # Controlled sampling — reduces hallucinated prose around JSON
    "top_p": 0.9,
    "top_k": 40,
    # Mild repetition penalty to discourage trailing duplicate content
    "repeat_penalty": 1.1,
}


def _apply_defaults(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload.setdefault("stream", False)
    payload.setdefault("format", "json")

    # Start from stable structured-output defaults, let OLLAMA_OPTIONS_JSON override,
    # then let the caller's own "options" key take final precedence.
    merged_options: Dict[str, Any] = dict(_STABLE_OUTPUT_DEFAULTS)

    if OLLAMA_OPTIONS_JSON:
        try:
            opts = json.loads(OLLAMA_OPTIONS_JSON)
            if isinstance(opts, dict):
                merged_options.update(opts)
        except Exception:
            pass

    caller_options = payload.get("options") or {}
    if isinstance(caller_options, dict):
        merged_options.update(caller_options)

    payload["options"] = merged_options
    return payload


async def ollama_generate_json(payload: Union[Dict[str, Any], str, Mapping[str, Any]]) -> Dict[str, Any]:
    url = f"{OLLAMA_URL}/api/generate"
    req = _apply_defaults(_as_mapping_payload(payload))

    timeout = httpx.Timeout(
        connect=10.0,
        read=OLLAMA_TIMEOUT_S,
        write=30.0,
        pool=30.0,
    )

    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json=req)
        r.raise_for_status()
        return r.json()


def _normalize_result(raw: Dict[str, Any], model_used: str) -> Dict[str, Any]:
    return {
        "model_used": model_used,
        "response": raw.get("response", ""),
        "done": raw.get("done", False),
        "done_reason": raw.get("done_reason"),
        "raw": raw,
    }


async def generate_with_fallback(payload: Union[Dict[str, Any], str, Mapping[str, Any]]) -> Dict[str, Any]:
    base = _apply_defaults(_as_mapping_payload(payload))

    e1: Optional[Exception] = None
    e2: Optional[Exception] = None

    try:
        raw = await ollama_generate_json({**base, "model": OLLAMA_PRIMARY_MODEL})
        return _normalize_result(raw, OLLAMA_PRIMARY_MODEL)
    except Exception as e:
        e1 = e

    try:
        raw = await ollama_generate_json({**base, "model": OLLAMA_FALLBACK_MODEL})
        return _normalize_result(raw, OLLAMA_FALLBACK_MODEL)
    except Exception as e:
        e2 = e

    raise RuntimeError(f"ollama primary failed: {_err(e1)} | fallback failed: {_err(e2)}")


async def generate_with_specific_model(
    model: str,
    payload: Union[Dict[str, Any], str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Call a specific named model without any primary/fallback logic.

    Used for JSON-repair retries where we want to explicitly route to the
    fallback model (mistral:latest) after the primary model (gemma3:latest)
    returned unparseable output.  A JSON-parse failure is counted as a model
    failure here, so we should not retry the same model.
    """
    base = _apply_defaults(_as_mapping_payload(payload))
    raw = await ollama_generate_json({**base, "model": model})
    return _normalize_result(raw, model)
