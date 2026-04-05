"""
Phase 11 / 11.2 MCP settings.

Uses the same ``BaseSettings`` / ``pydantic-settings`` import pattern as
``app.core.config`` and ``app.langchain.config`` for consistency.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import cast

from pydantic import ConfigDict, Field

try:
    _pydantic_settings = import_module("pydantic_settings")
    _BaseSettings = _pydantic_settings.BaseSettings
except ModuleNotFoundError:  # pragma: no cover
    from pydantic import BaseModel as _BaseSettings  # type: ignore[assignment]

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILES = tuple(
    str(path)
    for path in (_BACKEND_ROOT / ".env",)
    if path.exists()
)


class MCPSettings(_BaseSettings):  # type: ignore[misc]
    """Configuration values for the Phase 11 MCP tool layer."""

    model_config = cast(
        ConfigDict,
        {
            "case_sensitive": False,
            "env_file": _ENV_FILES,
            "env_file_encoding": "utf-8",
            "extra": "ignore",
        },
    )

    # Feature flag — set MCP_ENABLED=false to disable the entire MCP layer.
    enabled: bool = Field(default=True, alias="MCP_ENABLED")

    # Default timeout applied when a tool does not specify its own.
    default_timeout_seconds: float = Field(
        default=30.0,
        alias="MCP_DEFAULT_TIMEOUT_SECONDS",
    )

    # Default per-tool sliding-window rate limit.
    default_rate_limit_calls: int = Field(
        default=20,
        alias="MCP_DEFAULT_RATE_LIMIT_CALLS",
    )
    default_rate_limit_window_seconds: float = Field(
        default=60.0,
        alias="MCP_DEFAULT_RATE_LIMIT_WINDOW_SECONDS",
    )

    # Audit toggle — set to False to skip audit_log calls in tests if needed.
    audit_enabled: bool = Field(default=True, alias="MCP_AUDIT_ENABLED")

    # ── Phase 11.2 additions ─────────────────────────────────────────────────

    # Result cache TTL in seconds.  Set to 0 to disable caching.
    cache_ttl_seconds: float = Field(default=300.0, alias="MCP_CACHE_TTL_SECONDS")

    # LLM integration toggle.  Set to False to force deterministic fallback for
    # all tools (useful in CI / offline environments).
    llm_enabled: bool = Field(default=True, alias="MCP_LLM_ENABLED")

    # LLM temperature for MCP tool calls.  Lower = more deterministic.
    llm_temperature: float = Field(default=0.2, alias="MCP_LLM_TEMPERATURE")

    # Ownership check toggle.  When False the ownership check always passes
    # (useful in tests / local dev where Supabase is not available).
    ownership_check_enabled: bool = Field(
        default=True, alias="MCP_OWNERSHIP_CHECK_ENABLED"
    )

    # Metrics toggle.
    metrics_enabled: bool = Field(default=True, alias="MCP_METRICS_ENABLED")

    # Phase 11.3 bounded orchestration toggle.
    orchestration_enabled: bool = Field(
        default=True, alias="MCP_ORCHESTRATION_ENABLED"
    )

    # Hard upper bound on the number of steps a workflow may execute.
    orchestration_max_steps: int = Field(
        default=4, alias="MCP_ORCHESTRATION_MAX_STEPS"
    )

    # Default behavior for continuing after a non-critical step failure.
    orchestration_continue_on_non_critical_failure: bool = Field(
        default=False,
        alias="MCP_ORCHESTRATION_CONTINUE_ON_NON_CRITICAL_FAILURE",
    )

    # Phase 11 extra-credit admin proxy tools.
    github_tool_enabled: bool = Field(
        default=True, alias="MCP_GITHUB_TOOL_ENABLED"
    )
    github_api_url: str = Field(
        default="https://api.github.com",
        alias="MCP_GITHUB_API_URL",
    )
    github_token: str = Field(default="", alias="MCP_GITHUB_TOKEN")
    github_default_owner: str = Field(
        default="",
        alias="MCP_GITHUB_DEFAULT_OWNER",
    )
    github_allowed_owners: str = Field(
        default="",
        alias="MCP_GITHUB_ALLOWED_OWNERS",
    )
    github_allowed_repos: str = Field(
        default="",
        alias="MCP_GITHUB_ALLOWED_REPOS",
    )
    github_actions_summary_limit: int = Field(
        default=10,
        alias="MCP_GITHUB_ACTIONS_SUMMARY_LIMIT",
    )

    docker_tool_enabled: bool = Field(
        default=True, alias="MCP_DOCKER_TOOL_ENABLED"
    )
    docker_api_url: str = Field(default="", alias="MCP_DOCKER_API_URL")
    docker_allowed_services: str = Field(
        default="backend,llm-service,parser,ai-service,clamav",
        alias="MCP_DOCKER_ALLOWED_SERVICES",
    )
    proxy_timeout_seconds: float = Field(
        default=10.0,
        alias="MCP_PROXY_TIMEOUT_SECONDS",
    )

    @staticmethod
    def _csv_to_set(value: str) -> set[str]:
        return {item.strip() for item in value.split(",") if item.strip()}

    @property
    def github_allowed_owners_set(self) -> set[str]:
        return self._csv_to_set(self.github_allowed_owners)

    @property
    def github_allowed_repos_set(self) -> set[str]:
        return self._csv_to_set(self.github_allowed_repos)

    @property
    def docker_allowed_services_set(self) -> set[str]:
        return self._csv_to_set(self.docker_allowed_services)


# Module-level singleton — imported by executor.py and api/mcp.py.
mcp_settings = MCPSettings()
