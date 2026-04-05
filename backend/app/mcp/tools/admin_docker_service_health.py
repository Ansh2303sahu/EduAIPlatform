"""
MCP Tool: admin.docker_service_health.v1

Admin-only read-only Docker service health proxy.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.mcp.config import mcp_settings
from app.mcp.docker_client import get_service_health
from app.mcp.enums import RiskLevel, ToolNamespace, ToolRole
from app.mcp.handler_result import HandlerResult
from app.mcp.models import ToolDefinition
from app.mcp.registry import register_tool
from app.mcp.schemas import ToolExecutionContext


class DockerServiceHealthInput(BaseModel):
    model_config = {"extra": "forbid"}

    service_names: list[str] = Field(default_factory=list, max_length=20)
    include_image_info: bool = False


class DockerServiceStatus(BaseModel):
    model_config = {"extra": "forbid"}

    service_name: str
    state: str
    health_status: str
    restart_count: int = 0
    image_name: str | None = None
    image_tag: str | None = None
    checked_at: str


class DockerServiceHealthOutput(BaseModel):
    model_config = {"extra": "forbid"}

    services: list[DockerServiceStatus] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    checked_at: str


async def _handle(
    input: DockerServiceHealthInput,
    ctx: ToolExecutionContext,
) -> HandlerResult:
    del ctx
    result = await get_service_health(
        service_names=input.service_names or None,
        include_image_info=input.include_image_info,
    )
    return HandlerResult(
        output=DockerServiceHealthOutput.model_validate(result),
        llm_used=False,
        deterministic_fallback=False,
        confidence_note="Live Docker service metadata fetched through the Docker proxy.",
    )


register_tool(
    ToolDefinition(
        tool_name="admin.docker_service_health.v1",
        namespace=ToolNamespace.ADMIN,
        version="v1",
        description=(
            "Admin-only read-only Docker service health proxy. Returns filtered "
            "service state and health metadata without logs, mounts, or environment data."
        ),
        allowed_roles=frozenset({ToolRole.ADMIN}),
        risk_level=RiskLevel.LOW,
        enabled=mcp_settings.docker_tool_enabled,
        timeout_seconds=15.0,
        supports_idempotency=True,
        safe_for_multi_step=False,
        input_model=DockerServiceHealthInput,
        output_model=DockerServiceHealthOutput,
        handler=_handle,
    )
)
