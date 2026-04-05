"""
Role-aware LangChain model and chain factory for Phase 10.

Architecture choice
-------------------
Phase 7 already treats ``llm-service`` as the protected backend boundary that
talks to Ollama. To avoid bypassing that boundary from the backend directly,
this factory builds LangChain-compatible runnables that proxy prompt execution
through ``llm-service`` instead of calling Ollama from the backend process.

That keeps:
- the existing ``x-ai-secret`` security boundary
- centralized provider/model switching in ``llm-service``
- the Phase 10 prompt-building flow inside the backend LangChain pipelines
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import httpx
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableLambda

from app.core.config import settings
from app.langchain.config import phase10_settings
from app.langchain.enums import ChainRole

if TYPE_CHECKING:
    from app.langchain.services.execution_logger import Phase10ExecutionLogger

logger = logging.getLogger("phase10.chain_factory")


@dataclass(frozen=True)
class _LLMServiceModelSpec:
    role: str
    base_url: str
    model_name: str
    timeout_seconds: float
    temperature: float
    primary: bool

    @property
    def endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/llm/generate"

    @property
    def run_name(self) -> str:
        slot = "primary" if self.primary else "fallback"
        return f"phase10_{self.role}_{slot}_proxy_model"


def _llm_service_base_url(base_url: str | None = None) -> str:
    resolved = str(base_url or settings.llm_service_url or "").strip().rstrip("/")
    if not resolved:
        raise RuntimeError("LLM_SERVICE_URL not set; Phase 10 chain factory expects llm-service proxy access.")
    return resolved


def _llm_service_headers() -> dict[str, str]:
    if not settings.llm_service_secret:
        raise RuntimeError("LLM_SERVICE_SECRET not set; cannot call llm-service securely.")
    return {"x-ai-secret": settings.llm_service_secret}


def _coerce_timeout(timeout: float | int | None) -> float:
    if timeout is None:
        return float(phase10_settings.llm_timeout_seconds)
    try:
        return max(1.0, float(timeout))
    except (TypeError, ValueError):
        return float(phase10_settings.llm_timeout_seconds)


def _describe_http_error(exc: BaseException) -> str:
    text = str(exc).strip()
    return text or repr(exc)


def _coerce_role(role: str | ChainRole) -> str:
    return role.value if isinstance(role, ChainRole) else str(role).strip().lower()


def _resolve_model_name(*, primary: bool, model_name: str | None = None) -> str:
    if model_name:
        return str(model_name)
    return phase10_settings.primary_model if primary else phase10_settings.fallback_model


def _prompt_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, PromptValue):
        return value.to_string()
    if isinstance(value, BaseMessage):
        return str(value.content)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parts: list[str] = []
        for item in value:
            if isinstance(item, BaseMessage):
                parts.append(str(item.content))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(value)


async def _call_llm_service(spec: _LLMServiceModelSpec, prompt_text: str) -> AIMessage:
    payload = {
        "prompt": prompt_text,
        "role": spec.role,
        "temperature": spec.temperature,
        "requested_model": spec.model_name,
        "options": {
            "num_predict": phase10_settings.output_tokens_for(spec.role),
        },
    }
    timeout = httpx.Timeout(
        connect=min(spec.timeout_seconds, 15.0),
        read=spec.timeout_seconds,
        write=30.0,
        pool=30.0,
    )

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                spec.endpoint,
                json=payload,
                headers=_llm_service_headers(),
            )
    except httpx.TimeoutException as exc:
        raise RuntimeError(
            f"llm-service generate timeout role={spec.role} model={spec.model_name} "
            f"timeout_s={spec.timeout_seconds}: {_describe_http_error(exc)}"
        ) from exc
    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"llm-service generate connect_error role={spec.role} endpoint={spec.endpoint}: "
            f"{_describe_http_error(exc)}"
        ) from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"llm-service generate transport_error role={spec.role} endpoint={spec.endpoint}: "
            f"{_describe_http_error(exc)}"
        ) from exc

    if response.status_code >= 400:
        detail = response.text[:400]
        raise RuntimeError(
            f"llm-service generate failed role={spec.role} status={response.status_code}: {detail}"
        )

    data = response.json() or {}
    model_used = str(
        data.get("model_used")
        or response.headers.get("x-llm-model-used")
        or spec.model_name
        or ""
    )
    content = str(data.get("response") or "")
    return AIMessage(
        content=content,
        response_metadata={
            "model_used": model_used,
            "role": spec.role,
            "temperature": spec.temperature,
            "endpoint": "llm-service",
        },
    )


def _build_proxy_model(spec: _LLMServiceModelSpec) -> Runnable[Any, AIMessage]:
    async def _ainvoke(prompt_input: Any) -> AIMessage:
        prompt_text = _prompt_to_text(prompt_input)
        logger.debug(
            "chain_factory: proxying role=%s model=%s timeout=%s temperature=%s chars=%d",
            spec.role,
            spec.model_name,
            spec.timeout_seconds,
            spec.temperature,
            len(prompt_text),
        )
        return await _call_llm_service(spec, prompt_text)

    return RunnableLambda(_ainvoke, name=spec.run_name)


def _build_model_spec(
    *,
    role: str | ChainRole,
    primary: bool,
    base_url: str | None = None,
    model_name: str | None = None,
    timeout: float | int | None = None,
    temperature: float | None = None,
) -> _LLMServiceModelSpec:
    role_name = _coerce_role(role)
    return _LLMServiceModelSpec(
        role=role_name,
        base_url=_llm_service_base_url(base_url),
        model_name=_resolve_model_name(primary=primary, model_name=model_name),
        timeout_seconds=(
            _coerce_timeout(timeout)
            if timeout is not None
            else float(phase10_settings.timeout_for(role_name))
        ),
        temperature=float(phase10_settings.temperature_for(role_name) if temperature is None else temperature),
        primary=primary,
    )


def build_student_model(
    *,
    primary: bool = True,
    base_url: str | None = None,
    model_name: str | None = None,
    timeout: float | int | None = None,
    temperature: float | None = None,
) -> Runnable[Any, AIMessage]:
    """Build the student generation model runnable."""
    spec = _build_model_spec(
        role=ChainRole.STUDENT,
        primary=primary,
        base_url=base_url,
        model_name=model_name,
        timeout=timeout,
        temperature=temperature,
    )
    return _build_proxy_model(spec)


def build_professor_model(
    *,
    primary: bool = True,
    base_url: str | None = None,
    model_name: str | None = None,
    timeout: float | int | None = None,
    temperature: float | None = None,
) -> Runnable[Any, AIMessage]:
    """Build the professor generation model runnable."""
    spec = _build_model_spec(
        role=ChainRole.PROFESSOR,
        primary=primary,
        base_url=base_url,
        model_name=model_name,
        timeout=timeout,
        temperature=temperature,
    )
    return _build_proxy_model(spec)


def get_primary_model(*, role: str | ChainRole = ChainRole.STUDENT) -> Runnable[Any, AIMessage]:
    """Backward-compatible primary-model accessor with role awareness."""
    if _coerce_role(role) == ChainRole.PROFESSOR.value:
        return build_professor_model(primary=True)
    return build_student_model(primary=True)


def get_fallback_model(*, role: str | ChainRole = ChainRole.STUDENT) -> Runnable[Any, AIMessage]:
    """Backward-compatible fallback-model accessor with role awareness."""
    if _coerce_role(role) == ChainRole.PROFESSOR.value:
        return build_professor_model(primary=False)
    return build_student_model(primary=False)


def _extract_prompt_text(inputs: Any) -> str:
    if isinstance(inputs, dict):
        value = inputs.get("prompt_text", "")
    else:
        value = inputs
    return str(value or "")


def _extract_repair_prompt(inputs: Any) -> str:
    if isinstance(inputs, dict):
        value = inputs.get("repair_prompt", "")
    else:
        value = inputs
    return str(value or "")


def build_generation_chain(model: Runnable[Any, Any]) -> Runnable[Any, str]:
    """
    Build ``extract prompt_text | model | StrOutputParser``.

    The backend prompt builders already return a fully rendered prompt string,
    so we keep the chain lean and avoid wrapping that prompt inside another
    synthetic chat template.
    """
    return RunnableLambda(_extract_prompt_text, name="phase10_extract_prompt_text") | model | StrOutputParser()


def build_repair_chain(model: Runnable[Any, Any]) -> Runnable[Any, str]:
    """
    Build ``extract repair_prompt | model | StrOutputParser``.
    """
    return RunnableLambda(_extract_repair_prompt, name="phase10_extract_repair_prompt") | model | StrOutputParser()


def build_student_generation_chain(*, primary: bool = True) -> Runnable[Any, str]:
    """Convenience builder for the full student generation chain."""
    return build_generation_chain(build_student_model(primary=primary))


def build_professor_generation_chain(*, primary: bool = True) -> Runnable[Any, str]:
    """Convenience builder for the full professor generation chain."""
    return build_generation_chain(build_professor_model(primary=primary))


def build_chain_execution_config(
    *,
    role: str | ChainRole,
    callbacks: Sequence[Phase10ExecutionLogger] | None = None,
    tags: Sequence[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a small LangChain config payload for structured execution.
    """
    role_name = _coerce_role(role)
    config: dict[str, Any] = {
        "callbacks": list(callbacks or []),
        "tags": list(tags or [f"phase10:{role_name}"]),
        "metadata": {
            "role": role_name,
            "chain_version": phase10_settings.chain_version,
            "schema_version": phase10_settings.schema_version,
            "prompt_version": (
                phase10_settings.professor_prompt_version
                if role_name == ChainRole.PROFESSOR.value
                else phase10_settings.student_prompt_version
            ),
            **dict(metadata or {}),
        },
    }
    return config
