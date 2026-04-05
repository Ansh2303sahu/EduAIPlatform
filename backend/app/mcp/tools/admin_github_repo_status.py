"""
MCP Tool: admin.github_repo_status.v1

Admin-only read-only GitHub repository status proxy.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.mcp.config import mcp_settings
from app.mcp.enums import RiskLevel, ToolNamespace, ToolRole
from app.mcp.github_client import get_repo_status
from app.mcp.handler_result import HandlerResult
from app.mcp.models import ToolDefinition
from app.mcp.registry import register_tool
from app.mcp.schemas import ToolExecutionContext


class GitHubRepoStatusInput(BaseModel):
    model_config = {"extra": "forbid"}

    repo_name: str = Field(..., min_length=1, max_length=200)
    owner: str | None = Field(default=None, max_length=200)
    include_actions_summary: bool = False


class GitHubActionsSummary(BaseModel):
    model_config = {"extra": "forbid"}

    total_runs: int = 0
    latest_status: str = ""
    latest_conclusion: str = ""
    latest_run_created_at: str = ""
    recent_conclusion_counts: dict[str, int] = Field(default_factory=dict)


class GitHubRepoStatusOutput(BaseModel):
    model_config = {"extra": "forbid"}

    repo_name: str
    owner: str
    default_branch: str
    latest_commit_sha_short: str
    latest_commit_timestamp: str
    open_issues_count: int = 0
    open_pull_requests_count: int = 0
    actions_summary: GitHubActionsSummary | None = None
    warnings: list[str] = Field(default_factory=list)
    checked_at: str


async def _handle(
    input: GitHubRepoStatusInput,
    ctx: ToolExecutionContext,
) -> HandlerResult:
    del ctx
    result = await get_repo_status(
        repo_name=input.repo_name,
        owner=input.owner,
        include_actions_summary=input.include_actions_summary,
    )
    return HandlerResult(
        output=GitHubRepoStatusOutput.model_validate(result),
        llm_used=False,
        deterministic_fallback=False,
        confidence_note="Live repository metadata fetched through the GitHub proxy.",
    )


register_tool(
    ToolDefinition(
        tool_name="admin.github_repo_status.v1",
        namespace=ToolNamespace.ADMIN,
        version="v1",
        description=(
            "Admin-only read-only GitHub repository status proxy. Returns a small "
            "repository metadata summary without exposing code, diffs, or tokens."
        ),
        allowed_roles=frozenset({ToolRole.ADMIN}),
        risk_level=RiskLevel.LOW,
        enabled=mcp_settings.github_tool_enabled,
        timeout_seconds=15.0,
        supports_idempotency=True,
        safe_for_multi_step=False,
        input_model=GitHubRepoStatusInput,
        output_model=GitHubRepoStatusOutput,
        handler=_handle,
    )
)
