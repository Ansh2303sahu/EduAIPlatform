from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
import httpx

from app.main import app


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, object], url: str) -> None:
        self.status_code = status_code
        self._payload = payload
        self._text = str(payload)
        self.request = httpx.Request("POST", url)

    def json(self) -> dict[str, object]:
        return self._payload

    @property
    def text(self) -> str:
        return self._text


@pytest.mark.asyncio
async def test_llm_student_report_returns_clean_502_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.api.llm_reports as llm_reports

    monkeypatch.setattr(llm_reports.settings, "rag_enabled", False)
    monkeypatch.setattr(llm_reports.settings, "llm_service_url", "http://llm-service:8030")
    monkeypatch.setattr(llm_reports.settings, "llm_service_secret", "dev_llm_secret")

    class _TimeoutAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self) -> "_TimeoutAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, json: dict[str, object], headers: dict[str, str]):
            raise httpx.ConnectTimeout("connection timed out", request=httpx.Request("POST", url))

    monkeypatch.setattr(llm_reports.httpx, "AsyncClient", _TimeoutAsyncClient)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/api/llm/student/report",
            headers={"Authorization": "Bearer token-student-a"},
            json={"submission_id": "sub-1"},
        )

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "llm-service timed out calling /llm/student/report" in detail
    assert "ConnectTimeout" in detail


@pytest.mark.asyncio
async def test_llm_professor_report_surfaces_downstream_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.api.llm_reports as llm_reports

    monkeypatch.setattr(llm_reports.settings, "rag_enabled", False)
    monkeypatch.setattr(llm_reports.settings, "llm_service_url", "http://llm-service:8030")
    monkeypatch.setattr(llm_reports.settings, "llm_service_secret", "dev_llm_secret")

    class _ErrorAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self) -> "_ErrorAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, json: dict[str, object], headers: dict[str, str]) -> _FakeResponse:
            return _FakeResponse(
                502,
                {
                    "detail": (
                        "local generation failed requested_model=auto: RuntimeError: "
                        "Could not reach Ollama at http://host.docker.internal:11434."
                    )
                },
                url,
            )

    monkeypatch.setattr(llm_reports.httpx, "AsyncClient", _ErrorAsyncClient)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/api/llm/professor/report",
            headers={"Authorization": "Bearer token-admin"},
            json={"submission_id": "sub-2"},
        )

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "llm-service failed calling /llm/professor/report: HTTP 502" in detail
    assert "Could not reach Ollama" in detail
