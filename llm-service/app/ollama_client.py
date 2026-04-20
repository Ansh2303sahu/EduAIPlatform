import json
import logging
import httpx
from typing import Any, Dict, Mapping, Optional, Union

from app.config import settings

logger = logging.getLogger("llm_service.ollama_client")

OLLAMA_URL = str(settings.ollama_base_url or "http://host.docker.internal:11434").rstrip("/")
OLLAMA_TIMEOUT_S = float(settings.timeout_seconds or 180)
OLLAMA_PRIMARY_MODEL = str(settings.primary_model or "gemma3:4b")
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


def _timeout() -> httpx.Timeout:
    return httpx.Timeout(
        connect=10.0,
        read=OLLAMA_TIMEOUT_S,
        write=30.0,
        pool=30.0,
    )


def _transport_runtime_error(exc: Exception, *, model: str = "") -> RuntimeError:
    model_suffix = f" Requested model: {model}." if model else ""
    if isinstance(exc, httpx.ConnectError):
        return RuntimeError(
            f"Could not reach Ollama at {OLLAMA_URL}. "
            "Start the Ollama service and ensure the configured OLLAMA_BASE_URL "
            "is reachable from the llm-service container. "
            f"If Ollama runs elsewhere, update OLLAMA_BASE_URL.{model_suffix}"
        )
    if isinstance(exc, httpx.TimeoutException):
        return RuntimeError(
            f"Ollama at {OLLAMA_URL} did not respond before the timeout expired. "
            "Verify the Ollama service is healthy, or increase LLM_TIMEOUT_SECONDS "
            f"if the model is still loading.{model_suffix}"
        )
    return RuntimeError(_err(exc))


def _extract_response_error(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except Exception:
        payload = None

    if isinstance(payload, dict):
        for key in ("error", "message", "detail"):
            value = payload.get(key)
            if value:
                return str(value).strip()

    try:
        text = response.text
    except Exception:
        text = ""
    return str(text or "").strip()


async def ollama_list_models() -> list[str]:
    url = f"{OLLAMA_URL}/api/tags"
    async with httpx.AsyncClient(timeout=_timeout()) as client:
        try:
            response = await client.get(url)
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise _transport_runtime_error(exc) from exc
        response.raise_for_status()
        payload = response.json()

    models = payload.get("models") if isinstance(payload, dict) else []
    names: list[str] = []
    if isinstance(models, list):
        for item in models:
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("model") or item.get("name") or "").strip()
            if name:
                names.append(name)
    return names


async def ollama_provider_status() -> Dict[str, Any]:
    configured_models: list[str] = []
    for model in (OLLAMA_PRIMARY_MODEL, OLLAMA_FALLBACK_MODEL):
        model_name = str(model or "").strip()
        if model_name and model_name not in configured_models:
            configured_models.append(model_name)

    status: Dict[str, Any] = {
        "provider": "ollama",
        "base_url": OLLAMA_URL,
        "primary_model": OLLAMA_PRIMARY_MODEL,
        "fallback_model": OLLAMA_FALLBACK_MODEL,
        "configured_models": configured_models,
        "installed_models": [],
        "missing_models": list(configured_models),
        "ready": False,
    }

    try:
        installed_models = await ollama_list_models()
    except Exception as exc:
        status["error"] = _err(exc)
        return status

    missing_models = [model for model in configured_models if model not in installed_models]
    status["installed_models"] = installed_models
    status["missing_models"] = missing_models
    status["ready"] = len(missing_models) == 0
    return status


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
    # Slightly warmer but still stable settings for richer structured feedback.
    "temperature": float(settings.ollama_temperature or 0.15),
    "num_ctx": int(settings.ollama_num_ctx or 8192),
    "num_predict": int(settings.ollama_num_predict or 1200),
    "num_batch": int(settings.ollama_num_batch or 16),
    "top_p": float(settings.ollama_top_p or 0.9),
    "top_k": int(settings.ollama_top_k or 40),
    "repeat_penalty": float(settings.ollama_repeat_penalty or 1.12),
}

_PRIMARY_MAX_PREDICT = int(settings.ollama_max_num_predict or settings.ollama_num_predict or 1400)
_PRIMARY_MAX_CTX = int(settings.ollama_max_num_ctx or settings.ollama_num_ctx or 8192)
_FALLBACK_MAX_PREDICT = int(settings.ollama_fallback_num_predict or 512)
_FALLBACK_NUM_CTX = int(settings.ollama_fallback_num_ctx or 3072)
_FALLBACK_NUM_BATCH = int(settings.ollama_fallback_num_batch or 16)
_PRIMARY_RETRY_NUM_CTX = max(2048, min(_PRIMARY_MAX_CTX, 6144))
_PRIMARY_RETRY_NUM_PREDICT = max(512, min(_PRIMARY_MAX_PREDICT, 1000))
_PRIMARY_RETRY_NUM_BATCH = max(1, min(_FALLBACK_NUM_BATCH, 8))


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

    num_predict = _coerce_positive_int(merged_options.get("num_predict"))
    if _PRIMARY_MAX_PREDICT > 0:
        merged_options["num_predict"] = min(num_predict or _PRIMARY_MAX_PREDICT, _PRIMARY_MAX_PREDICT)

    num_ctx = _coerce_positive_int(merged_options.get("num_ctx"))
    if _PRIMARY_MAX_CTX > 0:
        merged_options["num_ctx"] = min(num_ctx or _PRIMARY_MAX_CTX, _PRIMARY_MAX_CTX)

    payload["options"] = merged_options
    return payload


def _coerce_positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _apply_fallback_relief(payload: Dict[str, Any]) -> Dict[str, Any]:
    fallback_payload = dict(payload)
    options = dict(fallback_payload.get("options") or {})

    num_predict = _coerce_positive_int(options.get("num_predict"))
    options["num_predict"] = min(num_predict or _FALLBACK_MAX_PREDICT, _FALLBACK_MAX_PREDICT)

    num_ctx = _coerce_positive_int(options.get("num_ctx"))
    options["num_ctx"] = min(num_ctx or _FALLBACK_NUM_CTX, _FALLBACK_NUM_CTX)

    num_batch = _coerce_positive_int(options.get("num_batch"))
    options["num_batch"] = min(num_batch or _FALLBACK_NUM_BATCH, _FALLBACK_NUM_BATCH)

    fallback_payload["options"] = options
    return fallback_payload


def _apply_primary_retry_relief(payload: Dict[str, Any]) -> Dict[str, Any]:
    retry_payload = dict(payload)
    options = dict(retry_payload.get("options") or {})

    num_predict = _coerce_positive_int(options.get("num_predict"))
    options["num_predict"] = min(num_predict or _PRIMARY_RETRY_NUM_PREDICT, _PRIMARY_RETRY_NUM_PREDICT)

    num_ctx = _coerce_positive_int(options.get("num_ctx"))
    options["num_ctx"] = min(num_ctx or _PRIMARY_RETRY_NUM_CTX, _PRIMARY_RETRY_NUM_CTX)

    num_batch = _coerce_positive_int(options.get("num_batch"))
    options["num_batch"] = min(num_batch or _PRIMARY_RETRY_NUM_BATCH, _PRIMARY_RETRY_NUM_BATCH)

    retry_payload["options"] = options
    return retry_payload


async def ollama_generate_json(payload: Union[Dict[str, Any], str, Mapping[str, Any]]) -> Dict[str, Any]:
    url = f"{OLLAMA_URL}/api/generate"
    req = _apply_defaults(_as_mapping_payload(payload))
    model = str(req.get("model") or "").strip()

    async with httpx.AsyncClient(timeout=_timeout()) as client:
        try:
            r = await client.post(url, json=req)
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise _transport_runtime_error(exc, model=model) from exc
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = _extract_response_error(exc.response)
            if (
                exc.response.status_code == 404
                and model
                and detail
                and "not found" in detail.lower()
            ):
                installed_models: list[str] = []
                try:
                    installed_models = await ollama_list_models()
                except Exception:
                    pass

                installed_summary = ", ".join(installed_models) if installed_models else "none"
                raise RuntimeError(
                    f"Ollama model '{model}' is not installed at {OLLAMA_URL}. "
                    f"Installed models: {installed_summary}. "
                    f"Run `ollama pull {model}` on the host, or update OLLAMA_PRIMARY_MODEL/OLLAMA_FALLBACK_MODEL."
                ) from exc

            detail_text = detail or _err(exc)
            raise RuntimeError(
                f"Ollama request to {url} failed with HTTP {exc.response.status_code}: {detail_text}"
            ) from exc

        return r.json()


def _normalize_result(raw: Dict[str, Any], model_used: str, *, stage: str = "primary") -> Dict[str, Any]:
    return {
        "model_used": model_used,
        "response": raw.get("response", ""),
        "done": raw.get("done", False),
        "done_reason": raw.get("done_reason"),
        "fallback_used": stage == "fallback",
        "fallback_stage": stage,
        "raw": raw,
    }


def _empty_response(raw: Dict[str, Any]) -> bool:
    return not str(raw.get("response") or "").strip()


async def generate_with_fallback(payload: Union[Dict[str, Any], str, Mapping[str, Any]]) -> Dict[str, Any]:
    base = _apply_defaults(_as_mapping_payload(payload))
    attempts: list[tuple[str, Dict[str, Any], str]] = [
        (OLLAMA_PRIMARY_MODEL, {**base, "model": OLLAMA_PRIMARY_MODEL}, "primary"),
        (
            OLLAMA_PRIMARY_MODEL,
            _apply_primary_retry_relief({**base, "model": OLLAMA_PRIMARY_MODEL}),
            "primary_relief",
        ),
    ]
    if OLLAMA_FALLBACK_MODEL.strip() and OLLAMA_FALLBACK_MODEL != OLLAMA_PRIMARY_MODEL:
        attempts.append(
            (
                OLLAMA_FALLBACK_MODEL,
                _apply_primary_retry_relief(
                    _apply_fallback_relief({**base, "model": OLLAMA_FALLBACK_MODEL})
                ),
                "fallback",
            )
        )

    errors: list[str] = []
    for model_name, request_payload, stage in attempts:
        try:
            if stage != "primary":
                logger.warning(
                    "ollama generation retry stage=%s model=%s options=%s",
                    stage,
                    model_name,
                    request_payload.get("options"),
                )
            raw = await ollama_generate_json(request_payload)
            if _empty_response(raw):
                raise RuntimeError(f"{stage} returned an empty response")
            return _normalize_result(raw, model_name, stage=stage)
        except Exception as exc:
            errors.append(f"{stage}={_err(exc)}")

    raise RuntimeError(" | ".join(errors) or "ollama generation failed")


async def generate_with_specific_model(
    model: str,
    payload: Union[Dict[str, Any], str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Call a specific named model without any primary/fallback logic.

    Used for JSON-repair retries where we want to explicitly route to the
    fallback model (phi3:mini) after the primary model (gemma3:4b)
    returned unparseable output.  A JSON-parse failure is counted as a model
    failure here, so we should not retry the same model.
    """
    base = _apply_defaults(_as_mapping_payload(payload))
    raw = await ollama_generate_json({**base, "model": model})
    return _normalize_result(raw, model)

