"""
Safe GitHub API proxy for admin MCP tools.

This client exposes a narrow, read-only repository status surface. It does not
return file trees, code blobs, PR diffs, or any token-bearing request details.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from app.mcp.config import mcp_settings
from app.mcp.errors import ExternalServiceError, PolicyDeniedError


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_detail(detail: str) -> str:
    text = (detail or "").strip()
    token = mcp_settings.github_token.strip()
    if token:
        text = text.replace(token, "[redacted]")
    return text


def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = mcp_settings.github_token.strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _resolve_repo(repo_name: str, owner: str | None) -> tuple[str, str]:
    value = repo_name.strip()
    if "/" in value and not owner:
        parsed_owner, parsed_repo = value.split("/", 1)
        return parsed_owner.strip(), parsed_repo.strip()

    resolved_owner = (owner or mcp_settings.github_default_owner).strip()
    if not resolved_owner:
        raise PolicyDeniedError(
            "GitHub owner is required for this tool.",
            reason="github_owner_required",
        )
    return resolved_owner, value


def _enforce_allowlist(owner: str, repo: str) -> None:
    full_name = f"{owner}/{repo}"
    allowed_owners = mcp_settings.github_allowed_owners_set
    allowed_repos = mcp_settings.github_allowed_repos_set

    if allowed_owners and owner not in allowed_owners:
        raise PolicyDeniedError(
            f"GitHub owner {owner!r} is not allowed for this tool.",
            reason="github_owner_not_allowed",
        )
    if allowed_repos and full_name not in allowed_repos and repo not in allowed_repos:
        raise PolicyDeniedError(
            f"GitHub repository {full_name!r} is not allowed for this tool.",
            reason="github_repo_not_allowed",
        )


async def _get_json(
    path: str,
    *,
    params: dict[str, str] | None = None,
) -> Any:
    base = mcp_settings.github_api_url.rstrip("/")
    url = f"{base}{path}"
    try:
        async with httpx.AsyncClient(timeout=mcp_settings.proxy_timeout_seconds) as client:
            resp = await client.get(url, headers=_headers(), params=params)
    except httpx.TimeoutException as exc:
        raise ExternalServiceError("GitHub API request timed out.") from exc
    except httpx.HTTPError as exc:
        raise ExternalServiceError("GitHub API request failed.") from exc

    if resp.status_code >= 300:
        detail = ""
        try:
            payload = resp.json()
            if isinstance(payload, dict):
                detail = str(payload.get("message") or payload.get("detail") or "").strip()
        except Exception:
            detail = resp.text.strip()
        detail = _sanitize_detail(detail)
        message = f"GitHub API returned HTTP {resp.status_code}."
        if detail:
            message = f"{message} {detail}"
        raise ExternalServiceError(message)

    try:
        return resp.json()
    except Exception as exc:
        raise ExternalServiceError("GitHub API returned invalid JSON.") from exc


async def get_repo_status(
    *,
    repo_name: str,
    owner: str | None = None,
    include_actions_summary: bool = False,
) -> dict[str, Any]:
    if not mcp_settings.github_tool_enabled:
        raise PolicyDeniedError(
            "GitHub status tool is disabled.",
            reason="github_tool_disabled",
        )

    resolved_owner, resolved_repo = _resolve_repo(repo_name, owner)
    _enforce_allowlist(resolved_owner, resolved_repo)

    repo_data = await _get_json(f"/repos/{resolved_owner}/{resolved_repo}")
    default_branch = str(repo_data.get("default_branch") or "main")
    latest_commit = await _get_json(
        f"/repos/{resolved_owner}/{resolved_repo}/commits/{default_branch}"
    )

    latest_commit_sha = str(latest_commit.get("sha") or "")[:8]
    commit_info = latest_commit.get("commit") or {}
    author_info = commit_info.get("author") or {}
    committer_info = commit_info.get("committer") or {}
    latest_commit_timestamp = str(
        author_info.get("date") or committer_info.get("date") or ""
    )

    pull_search = await _get_json(
        "/search/issues",
        params={
            "q": f"repo:{resolved_owner}/{resolved_repo} type:pr state:open",
            "per_page": "1",
        },
    )
    open_pull_requests_count = int(pull_search.get("total_count") or 0)
    open_issues_total = int(repo_data.get("open_issues_count") or 0)
    open_issues_count = max(0, open_issues_total - open_pull_requests_count)

    warnings: list[str] = []
    actions_summary: dict[str, Any] | None = None
    if include_actions_summary:
        try:
            actions_data = await _get_json(
                f"/repos/{resolved_owner}/{resolved_repo}/actions/runs",
                params={"per_page": str(max(1, min(20, mcp_settings.github_actions_summary_limit)))},
            )
            workflow_runs = actions_data.get("workflow_runs") or []
            latest_run = workflow_runs[0] if workflow_runs else {}
            conclusion_counts: dict[str, int] = {}
            for run in workflow_runs:
                conclusion = str(run.get("conclusion") or run.get("status") or "unknown")
                conclusion_counts[conclusion] = conclusion_counts.get(conclusion, 0) + 1
            actions_summary = {
                "total_runs": int(actions_data.get("total_count") or len(workflow_runs)),
                "latest_status": str(latest_run.get("status") or ""),
                "latest_conclusion": str(latest_run.get("conclusion") or ""),
                "latest_run_created_at": str(latest_run.get("created_at") or ""),
                "recent_conclusion_counts": conclusion_counts,
            }
        except ExternalServiceError as exc:
            warnings.append(str(exc))

    return {
        "repo_name": resolved_repo,
        "owner": resolved_owner,
        "default_branch": default_branch,
        "latest_commit_sha_short": latest_commit_sha,
        "latest_commit_timestamp": latest_commit_timestamp,
        "open_issues_count": open_issues_count,
        "open_pull_requests_count": open_pull_requests_count,
        "actions_summary": actions_summary,
        "warnings": warnings,
        "checked_at": _now_iso(),
    }
