from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import app.mcp  # noqa: F401


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, json_body=None, text: str = ""):
        self.status_code = status_code
        self._json_body = json_body
        self.text = text
        self.headers: dict[str, str] = {}

    def json(self):
        return self._json_body


def _fake_async_client_factory(handler):
    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, **kwargs):
            return handler(url, kwargs)

    return _FakeAsyncClient


class TestAdminMCPProxyTools:
    @pytest.mark.asyncio
    async def test_admin_can_execute_github_tool(self, async_client, monkeypatch):
        monkeypatch.setattr(
            "app.mcp.tools.admin_github_repo_status.get_repo_status",
            AsyncMock(
                return_value={
                    "repo_name": "EduAIPlatform",
                    "owner": "Ansh2303sahu",
                    "default_branch": "main",
                    "latest_commit_sha_short": "abc12345",
                    "latest_commit_timestamp": "2026-04-05T12:00:00+00:00",
                    "open_issues_count": 3,
                    "open_pull_requests_count": 1,
                    "actions_summary": {
                        "total_runs": 4,
                        "latest_status": "completed",
                        "latest_conclusion": "success",
                        "latest_run_created_at": "2026-04-05T11:00:00+00:00",
                        "recent_conclusion_counts": {"success": 3, "failure": 1},
                    },
                    "warnings": [],
                    "checked_at": "2026-04-05T12:05:00+00:00",
                }
            ),
        )

        async with async_client as ac:
            resp = await ac.post(
                "/api/mcp/execute",
                json={
                    "tool_name": "admin.github_repo_status.v1",
                    "payload": {"repo_name": "EduAIPlatform", "owner": "Ansh2303sahu"},
                },
                headers={"Authorization": "Bearer token-admin"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["tool_name"] == "admin.github_repo_status.v1"
        assert body["result"]["repo_name"] == "EduAIPlatform"
        assert set(body["result"]) >= {
            "repo_name",
            "owner",
            "default_branch",
            "latest_commit_sha_short",
            "latest_commit_timestamp",
            "open_issues_count",
            "open_pull_requests_count",
            "warnings",
            "checked_at",
        }

    @pytest.mark.asyncio
    async def test_non_admin_denied_github_tool(self, async_client):
        async with async_client as ac:
            resp = await ac.post(
                "/api/mcp/execute",
                json={
                    "tool_name": "admin.github_repo_status.v1",
                    "payload": {"repo_name": "EduAIPlatform"},
                },
                headers={"Authorization": "Bearer token-student-a"},
            )

        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_admin_can_execute_docker_tool(self, async_client, monkeypatch):
        monkeypatch.setattr(
            "app.mcp.tools.admin_docker_service_health.get_service_health",
            AsyncMock(
                return_value={
                    "services": [
                        {
                            "service_name": "backend",
                            "state": "running",
                            "health_status": "healthy",
                            "restart_count": 0,
                            "image_name": "eduaiplatform/backend",
                            "image_tag": "latest",
                            "checked_at": "2026-04-05T12:05:00+00:00",
                        }
                    ],
                    "warnings": [],
                    "checked_at": "2026-04-05T12:05:00+00:00",
                }
            ),
        )

        async with async_client as ac:
            resp = await ac.post(
                "/api/mcp/execute",
                json={
                    "tool_name": "admin.docker_service_health.v1",
                    "payload": {"service_names": ["backend"], "include_image_info": True},
                },
                headers={"Authorization": "Bearer token-admin"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["result"]["services"][0]["service_name"] == "backend"

    @pytest.mark.asyncio
    async def test_non_admin_denied_docker_tool(self, async_client):
        async with async_client as ac:
            resp = await ac.post(
                "/api/mcp/execute",
                json={
                    "tool_name": "admin.docker_service_health.v1",
                    "payload": {},
                },
                headers={"Authorization": "Bearer token-student-a"},
            )

        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_github_client_success_path(self, monkeypatch):
        from app.mcp import github_client
        from app.mcp.config import mcp_settings

        mcp_settings.github_tool_enabled = True
        mcp_settings.github_api_url = "https://api.github.example"
        mcp_settings.github_default_owner = "Ansh2303sahu"
        mcp_settings.github_allowed_owners = "Ansh2303sahu"
        mcp_settings.github_allowed_repos = "Ansh2303sahu/EduAIPlatform"

        def handler(url, kwargs):
            if url.endswith("/repos/Ansh2303sahu/EduAIPlatform"):
                return _FakeResponse(
                    json_body={
                        "default_branch": "main",
                        "open_issues_count": 7,
                    }
                )
            if url.endswith("/repos/Ansh2303sahu/EduAIPlatform/commits/main"):
                return _FakeResponse(
                    json_body={
                        "sha": "abcdef1234567890",
                        "commit": {"author": {"date": "2026-04-05T10:00:00Z"}},
                    }
                )
            if "/search/issues" in url:
                return _FakeResponse(json_body={"total_count": 2})
            if url.endswith("/repos/Ansh2303sahu/EduAIPlatform/actions/runs"):
                return _FakeResponse(
                    json_body={
                        "total_count": 3,
                        "workflow_runs": [
                            {
                                "status": "completed",
                                "conclusion": "success",
                                "created_at": "2026-04-05T09:00:00Z",
                            },
                            {
                                "status": "completed",
                                "conclusion": "failure",
                                "created_at": "2026-04-05T08:00:00Z",
                            },
                        ],
                    }
                )
            raise AssertionError(f"Unexpected URL: {url}")

        monkeypatch.setattr(
            github_client.httpx,
            "AsyncClient",
            _fake_async_client_factory(handler),
        )

        result = await github_client.get_repo_status(
            repo_name="EduAIPlatform",
            owner="Ansh2303sahu",
            include_actions_summary=True,
        )

        assert result["repo_name"] == "EduAIPlatform"
        assert result["owner"] == "Ansh2303sahu"
        assert result["default_branch"] == "main"
        assert result["latest_commit_sha_short"] == "abcdef12"
        assert result["open_pull_requests_count"] == 2
        assert result["open_issues_count"] == 5
        assert result["actions_summary"]["total_runs"] == 3

    @pytest.mark.asyncio
    async def test_github_client_service_error_path(self, monkeypatch):
        from app.mcp import github_client
        from app.mcp.config import mcp_settings
        from app.mcp.errors import ExternalServiceError

        mcp_settings.github_tool_enabled = True
        mcp_settings.github_api_url = "https://api.github.example"
        mcp_settings.github_default_owner = "Ansh2303sahu"
        mcp_settings.github_allowed_owners = "Ansh2303sahu"
        mcp_settings.github_allowed_repos = "Ansh2303sahu/EduAIPlatform"
        mcp_settings.github_token = "ghp_secret_123"

        def handler(url, kwargs):
            return _FakeResponse(
                status_code=503,
                json_body={"message": "bad upstream ghp_secret_123"},
            )

        monkeypatch.setattr(
            github_client.httpx,
            "AsyncClient",
            _fake_async_client_factory(handler),
        )

        with pytest.raises(ExternalServiceError) as exc_info:
            await github_client.get_repo_status(
                repo_name="EduAIPlatform",
                owner="Ansh2303sahu",
            )

        assert "ghp_secret_123" not in str(exc_info.value)
        assert "[redacted]" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_docker_client_success_path(self, monkeypatch):
        from app.mcp import docker_client
        from app.mcp.config import mcp_settings

        mcp_settings.docker_tool_enabled = True
        mcp_settings.docker_api_url = "http://docker-proxy"
        mcp_settings.docker_allowed_services = "backend,llm-service"

        def handler(url, kwargs):
            if url.endswith("/containers/json?all=true"):
                return _FakeResponse(
                    json_body=[
                        {
                            "Id": "c1",
                            "Image": "eduaiplatform/backend:latest",
                            "State": "running",
                            "Names": ["/backend-1"],
                            "Labels": {"com.docker.compose.service": "backend"},
                        },
                        {
                            "Id": "c2",
                            "Image": "eduaiplatform/llm-service:latest",
                            "State": "running",
                            "Names": ["/llm-service-1"],
                            "Labels": {"com.docker.compose.service": "llm-service"},
                        },
                    ]
                )
            if url.endswith("/containers/c1/json"):
                return _FakeResponse(
                    json_body={
                        "Config": {"Image": "eduaiplatform/backend:latest"},
                        "State": {"Status": "running", "Health": {"Status": "healthy"}},
                        "RestartCount": 1,
                    }
                )
            if url.endswith("/containers/c2/json"):
                return _FakeResponse(
                    json_body={
                        "Config": {"Image": "eduaiplatform/llm-service:latest"},
                        "State": {"Status": "running", "Health": {"Status": "healthy"}},
                        "RestartCount": 0,
                    }
                )
            raise AssertionError(f"Unexpected URL: {url}")

        monkeypatch.setattr(
            docker_client.httpx,
            "AsyncClient",
            _fake_async_client_factory(handler),
        )

        result = await docker_client.get_service_health(
            include_image_info=True,
        )

        assert len(result["services"]) == 2
        assert result["services"][0]["service_name"] in {"backend", "llm-service"}
        assert all("checked_at" in service for service in result["services"])

    @pytest.mark.asyncio
    async def test_docker_client_filtered_output_path(self, monkeypatch):
        from app.mcp import docker_client
        from app.mcp.config import mcp_settings

        mcp_settings.docker_tool_enabled = True
        mcp_settings.docker_api_url = "http://docker-proxy"
        mcp_settings.docker_allowed_services = "backend,llm-service"

        def handler(url, kwargs):
            if url.endswith("/containers/json?all=true"):
                return _FakeResponse(
                    json_body=[
                        {
                            "Id": "c1",
                            "Image": "eduaiplatform/backend:latest",
                            "State": "running",
                            "Names": ["/backend-1"],
                            "Labels": {"com.docker.compose.service": "backend"},
                        }
                    ]
                )
            if url.endswith("/containers/c1/json"):
                return _FakeResponse(
                    json_body={
                        "Config": {"Image": "eduaiplatform/backend:latest"},
                        "State": {"Status": "running", "Health": {"Status": "healthy"}},
                        "RestartCount": 0,
                    }
                )
            raise AssertionError(f"Unexpected URL: {url}")

        monkeypatch.setattr(
            docker_client.httpx,
            "AsyncClient",
            _fake_async_client_factory(handler),
        )

        result = await docker_client.get_service_health(
            service_names=["backend"],
            include_image_info=False,
        )

        assert [service["service_name"] for service in result["services"]] == ["backend"]
        assert result["services"][0]["image_name"] is None
        assert result["services"][0]["image_tag"] is None

    def test_no_orchestration_exposure_for_admin_tools(self):
        from app.mcp.planner import list_visible_workflows
        from app.mcp.registry import resolve_tool

        github_tool = resolve_tool("admin.github_repo_status.v1")
        docker_tool = resolve_tool("admin.docker_service_health.v1")

        assert github_tool.safe_for_multi_step is False
        assert docker_tool.safe_for_multi_step is False
        assert all(
            "admin." not in step.tool_name
            for workflow in list_visible_workflows("admin")
            for step in workflow.steps
        )

    @pytest.mark.asyncio
    async def test_no_secret_leakage_in_failure_envelope(self, async_client, monkeypatch):
        from app.mcp.cache import invalidate
        from app.mcp.errors import ExternalServiceError

        invalidate("admin.github_repo_status.v1")
        monkeypatch.setattr(
            "app.mcp.tools.admin_github_repo_status.get_repo_status",
            AsyncMock(
                side_effect=ExternalServiceError(
                    "GitHub API returned HTTP 503. [redacted]"
                )
            ),
        )

        async with async_client as ac:
            resp = await ac.post(
                "/api/mcp/execute",
                json={
                    "tool_name": "admin.github_repo_status.v1",
                    "payload": {"repo_name": "EduAIPlatform", "owner": "Ansh2303sahu"},
                },
                headers={"Authorization": "Bearer token-admin"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is False
        assert "ghp_" not in body["message"]

    @pytest.mark.asyncio
    async def test_admin_tool_list_shows_stable_schema(self, async_client):
        async with async_client as ac:
            resp = await ac.get(
                "/api/mcp/tools",
                headers={"Authorization": "Bearer token-admin"},
            )

        assert resp.status_code == 200
        body = resp.json()
        tool_names = {tool["tool_name"] for tool in body["tools"]}
        assert "admin.github_repo_status.v1" in tool_names
        assert "admin.docker_service_health.v1" in tool_names
