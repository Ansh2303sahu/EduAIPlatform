"""
Tests for the Phase 10 LangChain chain factory.
"""

from __future__ import annotations

from typing import Any

import pytest
import httpx

from app.langchain.config import phase10_settings
from app.langchain.services.chain_factory import (
    build_chain_execution_config,
    build_generation_chain,
    build_professor_model,
    build_student_model,
)


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], *, status_code: int = 200, headers: dict[str, str] | None = None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.text = str(payload)

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeAsyncClient:
    def __init__(self, sink: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        sink["init_kwargs"] = kwargs
        self._sink = sink

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, url: str, *, json: dict[str, Any], headers: dict[str, str]) -> _FakeResponse:
        self._sink["url"] = url
        self._sink["json"] = json
        self._sink["headers"] = headers
        return _FakeResponse(
            {"response": "{\"ok\":true}", "model_used": json.get("requested_model", "")},
            headers={"x-llm-model-used": json.get("requested_model", "")},
        )


@pytest.mark.asyncio
async def test_student_generation_chain_uses_llm_service_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    sink: dict[str, Any] = {}

    monkeypatch.setattr("app.langchain.services.chain_factory.settings.llm_service_secret", "test-secret")
    monkeypatch.setattr(
        "app.langchain.services.chain_factory.httpx.AsyncClient",
        lambda *args, **kwargs: _FakeAsyncClient(sink, *args, **kwargs),
    )

    chain = build_generation_chain(build_student_model())
    result = await chain.ainvoke({"prompt_text": "student prompt"})

    assert result == '{"ok":true}'
    assert sink["url"].endswith("/llm/generate")
    assert sink["json"]["role"] == "student"
    assert sink["json"]["temperature"] == phase10_settings.student_temperature
    assert sink["json"]["requested_model"] == phase10_settings.primary_model
    assert sink["json"]["options"]["num_predict"] == phase10_settings.output_tokens_for(
        "student",
        submission_chars=len("student prompt"),
    )
    assert sink["json"]["options"]["num_ctx"] == phase10_settings.ollama_num_ctx
    assert sink["json"]["options"]["top_p"] == phase10_settings.ollama_top_p
    assert sink["json"]["options"]["top_k"] == phase10_settings.ollama_top_k
    assert sink["json"]["options"]["repeat_penalty"] == phase10_settings.ollama_repeat_penalty
    assert sink["headers"]["x-ai-secret"]


@pytest.mark.asyncio
async def test_professor_generation_chain_applies_professor_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    sink: dict[str, Any] = {}

    monkeypatch.setattr("app.langchain.services.chain_factory.settings.llm_service_secret", "test-secret")
    monkeypatch.setattr(
        "app.langchain.services.chain_factory.httpx.AsyncClient",
        lambda *args, **kwargs: _FakeAsyncClient(sink, *args, **kwargs),
    )

    chain = build_generation_chain(build_professor_model(primary=False))
    await chain.ainvoke({"prompt_text": "professor prompt"})

    assert sink["json"]["role"] == "professor"
    assert sink["json"]["temperature"] == phase10_settings.professor_temperature
    assert sink["json"]["requested_model"] == phase10_settings.fallback_model
    assert sink["json"]["options"]["num_predict"] == phase10_settings.output_tokens_for(
        "professor",
        submission_chars=len("professor prompt"),
    )


@pytest.mark.asyncio
async def test_generation_chain_scales_output_budget_from_submission_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink: dict[str, Any] = {}

    monkeypatch.setattr("app.langchain.services.chain_factory.settings.llm_service_secret", "test-secret")
    monkeypatch.setattr(
        "app.langchain.services.chain_factory.httpx.AsyncClient",
        lambda *args, **kwargs: _FakeAsyncClient(sink, *args, **kwargs),
    )

    chain = build_generation_chain(build_student_model())
    await chain.ainvoke({"prompt_text": "student prompt", "submission_chars": 1500})
    short_budget = sink["json"]["options"]["num_predict"]

    await chain.ainvoke({"prompt_text": "student prompt", "submission_chars": 9000})
    long_budget = sink["json"]["options"]["num_predict"]

    assert short_budget == phase10_settings.output_tokens_for("student", submission_chars=1500)
    assert long_budget == phase10_settings.output_tokens_for("student", submission_chars=9000)


class _TimeoutAsyncClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def __aenter__(self) -> "_TimeoutAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, url: str, *, json: dict[str, Any], headers: dict[str, str]):
        raise httpx.ReadTimeout("", request=httpx.Request("POST", url))


@pytest.mark.asyncio
async def test_generation_chain_surfaces_descriptive_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.langchain.services.chain_factory.settings.llm_service_secret", "test-secret")
    monkeypatch.setattr(
        "app.langchain.services.chain_factory.httpx.AsyncClient",
        lambda *args, **kwargs: _TimeoutAsyncClient(*args, **kwargs),
    )

    chain = build_generation_chain(build_student_model())

    with pytest.raises(RuntimeError) as excinfo:
        await chain.ainvoke({"prompt_text": "student prompt"})

    message = str(excinfo.value)
    assert "llm-service generate timeout" in message
    assert "role=student" in message


def test_build_chain_execution_config_includes_versions() -> None:
    config = build_chain_execution_config(role="student")

    assert config["metadata"]["role"] == "student"
    assert config["metadata"]["chain_version"] == phase10_settings.chain_version
    assert config["metadata"]["schema_version"] == phase10_settings.schema_version
    assert config["metadata"]["prompt_version"] == phase10_settings.student_prompt_version
