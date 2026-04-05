"""
Phase 11.3 deterministic MCP workflow planner.

Planning is rule-based only. A workflow name maps to a fixed sequence after the
planner validates role access, max-step bounds, and tool multi-step safety.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.mcp.config import mcp_settings
from app.mcp.enums import ToolRole
from app.mcp.errors import PolicyDeniedError
from app.mcp.models import ToolDefinition
from app.mcp.registry import resolve_tool
from app.mcp.workflow_rules import WorkflowRule, get_workflow_rule, list_workflow_rules


@dataclass(frozen=True)
class PlannedStep:
    index: int
    step_name: str
    tool_name: str
    critical: bool
    definition: ToolDefinition


@dataclass(frozen=True)
class WorkflowPlan:
    workflow_name: str
    description: str
    steps: tuple[PlannedStep, ...]
    effective_max_steps: int
    continue_on_non_critical_failure: bool


def _parse_role(role: str) -> ToolRole:
    try:
        return ToolRole(role)
    except ValueError as exc:
        raise PolicyDeniedError(
            f"Unsupported caller role for workflow orchestration: {role!r}",
            reason="unsupported_role",
        ) from exc


def _validate_workflow_role(rule: WorkflowRule, role: ToolRole) -> None:
    if role not in rule.allowed_roles:
        raise PolicyDeniedError(
            f"Role {role.value!r} is not permitted for workflow {rule.workflow_name!r}.",
            reason="workflow_role_mismatch",
        )


def _validate_plannable_tool(tool_name: str, role: ToolRole) -> ToolDefinition:
    defn = resolve_tool(tool_name)
    if not defn.safe_for_multi_step:
        raise PolicyDeniedError(
            f"Tool {tool_name!r} is not approved for multi-step orchestration.",
            reason="tool_not_safe_for_multi_step",
        )
    if role not in defn.allowed_roles:
        raise PolicyDeniedError(
            f"Role {role.value!r} is not permitted for tool {tool_name!r}.",
            reason="tool_role_mismatch",
        )
    return defn


def build_workflow_plan(
    workflow_name: str,
    *,
    role: str,
    requested_max_steps: int | None = None,
    continue_on_non_critical_failure: bool | None = None,
) -> WorkflowPlan:
    rule = get_workflow_rule(workflow_name)
    if rule is None:
        raise PolicyDeniedError(
            f"Unknown workflow {workflow_name!r}.",
            reason="unknown_workflow",
        )

    role_enum = _parse_role(role)
    _validate_workflow_role(rule, role_enum)

    configured_max = max(1, int(mcp_settings.orchestration_max_steps))
    if requested_max_steps is None:
        effective_max = min(configured_max, len(rule.steps))
    else:
        effective_max = min(configured_max, requested_max_steps, len(rule.steps))

    can_continue = rule.continue_on_non_critical_failure
    if continue_on_non_critical_failure is False:
        can_continue = False
    elif continue_on_non_critical_failure is True:
        can_continue = can_continue and True

    steps = tuple(
        PlannedStep(
            index=i,
            step_name=step.step_name,
            tool_name=step.tool_name,
            critical=step.critical,
            definition=_validate_plannable_tool(step.tool_name, role_enum),
        )
        for i, step in enumerate(rule.steps, start=1)
    )

    return WorkflowPlan(
        workflow_name=rule.workflow_name,
        description=rule.description,
        steps=steps,
        effective_max_steps=effective_max,
        continue_on_non_critical_failure=can_continue,
    )


def list_visible_workflows(role: str) -> list[WorkflowRule]:
    role_enum = _parse_role(role)
    return [rule for rule in list_workflow_rules() if role_enum in rule.allowed_roles]
