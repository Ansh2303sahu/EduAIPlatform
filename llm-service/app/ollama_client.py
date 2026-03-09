import os
import json
import httpx
from typing import Any, Dict, Mapping, Optional, Union

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434").rstrip("/")
OLLAMA_TIMEOUT_S = float(os.getenv("OLLAMA_TIMEOUT_S", "180"))
OLLAMA_PRIMARY_MODEL = os.getenv("OLLAMA_PRIMARY_MODEL", "mistral")
OLLAMA_FALLBACK_MODEL = os.getenv("OLLAMA_FALLBACK_MODEL", "phi3")
OLLAMA_OPTIONS_JSON = os.getenv("OLLAMA_OPTIONS_JSON", "").strip()


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


def _apply_defaults(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload.setdefault("stream", False)

    if OLLAMA_OPTIONS_JSON:
        try:
            opts = json.loads(OLLAMA_OPTIONS_JSON)
            if isinstance(opts, dict):
                payload_options = payload.get("options") or {}
                if not isinstance(payload_options, dict):
                    payload_options = {}
                payload["options"] = {**payload_options, **opts}
        except Exception:
            pass

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