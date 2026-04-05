"""
Safe Docker status proxy for admin MCP tools.

This client reads a tightly filtered subset of container status data from a
configured Docker API endpoint. It never returns logs, env vars, mounts, or
arbitrary inspection output.
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
    return (detail or "").strip()


def _parse_image(image_ref: str) -> tuple[str, str]:
    if ":" not in image_ref:
        return image_ref, ""
    name, tag = image_ref.rsplit(":", 1)
    return name, tag


def _allowed_services() -> list[str]:
    services = sorted(mcp_settings.docker_allowed_services_set)
    if not services:
        raise PolicyDeniedError(
            "No Docker services are configured for this tool.",
            reason="docker_services_not_configured",
        )
    return services


def _resolve_service_names(service_names: list[str] | None) -> list[str]:
    allowed = set(_allowed_services())
    if not service_names:
        return sorted(allowed)

    resolved = [name.strip() for name in service_names if name and name.strip()]
    invalid = [name for name in resolved if name not in allowed]
    if invalid:
        raise PolicyDeniedError(
            f"Docker services not allowed for this tool: {invalid!r}",
            reason="docker_service_not_allowed",
        )
    return resolved


async def _get_json(path: str) -> Any:
    base = mcp_settings.docker_api_url.rstrip("/")
    if not base:
        raise ExternalServiceError("Docker status proxy is not configured.")

    url = f"{base}{path}"
    try:
        async with httpx.AsyncClient(timeout=mcp_settings.proxy_timeout_seconds) as client:
            resp = await client.get(url)
    except httpx.TimeoutException as exc:
        raise ExternalServiceError("Docker status request timed out.") from exc
    except httpx.HTTPError as exc:
        raise ExternalServiceError("Docker status request failed.") from exc

    if resp.status_code >= 300:
        detail = ""
        try:
            payload = resp.json()
            if isinstance(payload, dict):
                detail = str(payload.get("message") or payload.get("detail") or "").strip()
        except Exception:
            detail = resp.text.strip()
        detail = _sanitize_detail(detail)
        message = f"Docker status API returned HTTP {resp.status_code}."
        if detail:
            message = f"{message} {detail}"
        raise ExternalServiceError(message)

    try:
        return resp.json()
    except Exception as exc:
        raise ExternalServiceError("Docker status API returned invalid JSON.") from exc


def _service_name_from_container(container: dict[str, Any]) -> str:
    labels = container.get("Labels") or {}
    service_name = str(labels.get("com.docker.compose.service") or "").strip()
    if service_name:
        return service_name
    names = container.get("Names") or []
    if names:
        return str(names[0]).lstrip("/")
    return str(container.get("Id") or "")


async def get_service_health(
    *,
    service_names: list[str] | None = None,
    include_image_info: bool = False,
) -> dict[str, Any]:
    if not mcp_settings.docker_tool_enabled:
        raise PolicyDeniedError(
            "Docker service health tool is disabled.",
            reason="docker_tool_disabled",
        )

    requested_names = _resolve_service_names(service_names)
    checked_at = _now_iso()
    warnings: list[str] = []

    containers = await _get_json("/containers/json?all=true")
    if not isinstance(containers, list):
        raise ExternalServiceError("Docker status API returned unexpected container data.")

    by_service: dict[str, dict[str, Any]] = {}
    for container in containers:
        if not isinstance(container, dict):
            continue
        service_name = _service_name_from_container(container)
        by_service[service_name] = container

    services: list[dict[str, Any]] = []
    for service_name in requested_names:
        container = by_service.get(service_name)
        if container is None:
            warnings.append(f"Service {service_name!r} was not found in Docker status data.")
            services.append(
                {
                    "service_name": service_name,
                    "state": "not_found",
                    "health_status": "unknown",
                    "restart_count": 0,
                    "image_name": None,
                    "image_tag": None,
                    "checked_at": checked_at,
                }
            )
            continue

        inspect = await _get_json(f"/containers/{container.get('Id')}/json")
        state = inspect.get("State") or {}
        image_ref = str((inspect.get("Config") or {}).get("Image") or container.get("Image") or "")
        image_name, image_tag = _parse_image(image_ref)

        service_row = {
            "service_name": service_name,
            "state": str(state.get("Status") or container.get("State") or ""),
            "health_status": str((state.get("Health") or {}).get("Status") or "none"),
            "restart_count": int(inspect.get("RestartCount") or 0),
            "checked_at": checked_at,
        }
        if include_image_info:
            service_row["image_name"] = image_name
            service_row["image_tag"] = image_tag
        else:
            service_row["image_name"] = None
            service_row["image_tag"] = None
        services.append(service_row)

    return {
        "services": services,
        "warnings": warnings,
        "checked_at": checked_at,
    }
