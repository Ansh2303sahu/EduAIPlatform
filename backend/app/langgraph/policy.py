"""Policy and execution-constraint primitives for Phase 12 LangGraph.

These helpers govern graph behaviour without replacing existing Phase 10
business logic. They are intentionally deterministic, role-aware, JSON-safe,
and suitable for later use inside node modules and the internal engine.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.langchain.enums import ChainRole
from app.mcp.enums import ToolNamespace, ToolRole
from app.mcp.errors import DisabledToolError, UnknownToolError
from app.mcp.registry import resolve_tool

from .config import phase12_settings
from .state import GraphValidationStatus, Phase12GraphState


class Phase12FailureCode(str, Enum):
    """Failure classifications used by the Phase 12 orchestration layer."""

    INPUT_INVALID = "input_invalid"
    ROLE_BLOCKED = "role_blocked"
    POLICY_BLOCKED = "policy_blocked"
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"
    ML_TIMEOUT = "ml_timeout"
    ML_LOW_CONFIDENCE = "ml_low_confidence"
    RETRIEVAL_WEAK = "retrieval_weak"
    RETRIEVAL_REPAIR_FAILED = "retrieval_repair_failed"
    TOOL_BLOCKED = "tool_blocked"
    TOOL_VALIDATION_FAILED = "tool_validation_failed"
    GENERATION_FAILED = "generation_failed"
    VALIDATION_FAILED = "validation_failed"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    LOOP_PREVENTED = "loop_prevented"


class Phase12ExecutionLimits(BaseModel):
    """Boundaries that keep graph execution deterministic and finite."""

    max_total_graph_steps: int = Field(ge=1)
    max_retries_per_node: int = Field(ge=0)
    max_retrieval_retries: int = Field(ge=0)
    max_repair_loops: int = Field(ge=0)
    max_tool_calls: int = Field(ge=0)
    max_repeated_state_transitions: int = Field(ge=1)


class Phase12TimeoutBudgets(BaseModel):
    """Timeout budgets for later runtime wrappers, stored as policy only for now."""

    ml_timeout_budget_seconds: float = Field(ge=1.0)
    retrieval_timeout_budget_seconds: float = Field(ge=1.0)
    generation_timeout_budget_seconds: float = Field(ge=1.0)
    validation_timeout_budget_seconds: float = Field(ge=1.0)
    tool_timeout_budget_seconds: float = Field(ge=1.0)


class Phase12RolePolicy(BaseModel):
    """Role-specific policy contract for a Phase 12 graph run."""

    role: ChainRole
    allowed_graphs: list[str] = Field(default_factory=list)
    retrieval_audience: str
    allowed_tool_namespace: ToolNamespace
    allow_optional_mcp_tools: bool = False
    allow_safe_mode_downgrade: bool = True
    safe_mode_ml_confidence_threshold: float = Field(ge=0.0, le=1.0)
    safe_mode_failure_codes: list[Phase12FailureCode] = Field(default_factory=list)


class Phase12PolicyDecision(BaseModel):
    """Compact allow/deny result for graph policy checks."""

    allowed: bool
    reason: str = ""
    failure_code: Phase12FailureCode | None = None
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def allow(
        cls,
        *,
        reason: str = "",
        warnings: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "Phase12PolicyDecision":
        return cls(
            allowed=True,
            reason=reason,
            warnings=warnings or [],
            metadata=metadata or {},
        )

    @classmethod
    def deny(
        cls,
        *,
        reason: str,
        failure_code: Phase12FailureCode,
        warnings: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "Phase12PolicyDecision":
        return cls(
            allowed=False,
            reason=reason,
            failure_code=failure_code,
            warnings=warnings or [],
            metadata=metadata or {},
        )


class Phase12PolicyBundle(BaseModel):
    """Bundle the role policy, execution limits, and timeout budgets together."""

    role_policy: Phase12RolePolicy
    limits: Phase12ExecutionLimits
    timeouts: Phase12TimeoutBudgets


def _normalize_role(role: str | ChainRole) -> ChainRole:
    return role if isinstance(role, ChainRole) else ChainRole(str(role))


def build_execution_limits() -> Phase12ExecutionLimits:
    """Build the current graph execution limits from settings."""

    return Phase12ExecutionLimits(
        max_total_graph_steps=phase12_settings.max_graph_steps,
        max_retries_per_node=phase12_settings.max_node_retries,
        max_retrieval_retries=phase12_settings.max_retrieval_retries,
        max_repair_loops=phase12_settings.max_repair_loops,
        max_tool_calls=phase12_settings.max_tool_calls,
        max_repeated_state_transitions=phase12_settings.max_repeated_state_transitions,
    )


def build_timeout_budgets() -> Phase12TimeoutBudgets:
    """Build timeout budgets from settings without applying them at runtime yet."""

    return Phase12TimeoutBudgets(
        ml_timeout_budget_seconds=phase12_settings.ml_timeout_budget_seconds,
        retrieval_timeout_budget_seconds=phase12_settings.retrieval_timeout_budget_seconds,
        generation_timeout_budget_seconds=phase12_settings.generation_timeout_budget_seconds,
        validation_timeout_budget_seconds=phase12_settings.validation_timeout_budget_seconds,
        tool_timeout_budget_seconds=phase12_settings.tool_timeout_budget_seconds,
    )


def build_role_policy(role: str | ChainRole) -> Phase12RolePolicy:
    """Build the role-specific policy contract for a Phase 12 graph."""

    chain_role = _normalize_role(role)
    if chain_role == ChainRole.STUDENT:
        return Phase12RolePolicy(
            role=chain_role,
            allowed_graphs=[phase12_settings.student_graph_name],
            retrieval_audience="student",
            allowed_tool_namespace=ToolNamespace.STUDENT,
            allow_optional_mcp_tools=(
                phase12_settings.allow_mcp_tools and phase12_settings.allow_student_mcp_tools
            ),
            safe_mode_ml_confidence_threshold=phase12_settings.student_safe_mode_ml_confidence_threshold,
            safe_mode_failure_codes=[
                Phase12FailureCode.ML_TIMEOUT,
                Phase12FailureCode.ML_LOW_CONFIDENCE,
                Phase12FailureCode.RETRIEVAL_WEAK,
                Phase12FailureCode.RETRIEVAL_REPAIR_FAILED,
                Phase12FailureCode.GENERATION_FAILED,
                Phase12FailureCode.VALIDATION_FAILED,
                Phase12FailureCode.EVIDENCE_INSUFFICIENT,
            ],
        )
    return Phase12RolePolicy(
        role=chain_role,
        allowed_graphs=[phase12_settings.professor_graph_name],
        retrieval_audience="professor",
        allowed_tool_namespace=ToolNamespace.PROFESSOR,
        allow_optional_mcp_tools=(
            phase12_settings.allow_mcp_tools and phase12_settings.allow_professor_mcp_tools
        ),
        safe_mode_ml_confidence_threshold=phase12_settings.professor_safe_mode_ml_confidence_threshold,
        safe_mode_failure_codes=[
            Phase12FailureCode.ML_TIMEOUT,
            Phase12FailureCode.ML_LOW_CONFIDENCE,
            Phase12FailureCode.RETRIEVAL_WEAK,
            Phase12FailureCode.RETRIEVAL_REPAIR_FAILED,
            Phase12FailureCode.GENERATION_FAILED,
            Phase12FailureCode.VALIDATION_FAILED,
            Phase12FailureCode.EVIDENCE_INSUFFICIENT,
        ],
    )


def build_policy_bundle(role: str | ChainRole) -> Phase12PolicyBundle:
    """Build the full Phase 12 policy bundle for a role."""

    return Phase12PolicyBundle(
        role_policy=build_role_policy(role),
        limits=build_execution_limits(),
        timeouts=build_timeout_budgets(),
    )


def validate_graph_for_role(*, graph_name: str, role: str | ChainRole) -> Phase12PolicyDecision:
    """Ensure a role may only select its own internal graph."""

    role_policy = build_role_policy(role)
    if graph_name in role_policy.allowed_graphs:
        return Phase12PolicyDecision.allow(
            reason="graph_allowed",
            metadata={"graph_name": graph_name, "role": role_policy.role.value},
        )
    return Phase12PolicyDecision.deny(
        reason=f"Graph {graph_name!r} is not allowed for role {role_policy.role.value!r}.",
        failure_code=Phase12FailureCode.ROLE_BLOCKED,
        metadata={
            "graph_name": graph_name,
            "role": role_policy.role.value,
            "allowed_graphs": role_policy.allowed_graphs,
        },
    )


def validate_retrieval_access(
    *,
    role: str | ChainRole,
    requested_audience: str,
) -> Phase12PolicyDecision:
    """Keep student and professor retrieval access strictly separated."""

    role_policy = build_role_policy(role)
    requested = str(requested_audience).strip().lower()
    if requested == role_policy.retrieval_audience:
        return Phase12PolicyDecision.allow(
            reason="retrieval_allowed",
            metadata={"requested_audience": requested},
        )
    return Phase12PolicyDecision.deny(
        reason=(
            f"Retrieval audience {requested!r} does not match role "
            f"{role_policy.role.value!r}."
        ),
        failure_code=Phase12FailureCode.ROLE_BLOCKED,
        metadata={
            "requested_audience": requested,
            "expected_audience": role_policy.retrieval_audience,
        },
    )


def _load_tool_definition(tool_name: str):
    from app.mcp import tools as _registered_tools  # noqa: F401

    return resolve_tool(tool_name)


def validate_tool_access(
    *,
    role: str | ChainRole,
    tool_name: str,
) -> Phase12PolicyDecision:
    """Validate that a role may use a specific MCP tool inside a graph."""

    role_policy = build_role_policy(role)
    try:
        defn = _load_tool_definition(tool_name)
    except (UnknownToolError, DisabledToolError) as exc:
        return Phase12PolicyDecision.deny(
            reason=str(exc),
            failure_code=Phase12FailureCode.TOOL_BLOCKED,
            metadata={"tool_name": tool_name},
        )

    expected_role = ToolRole(role_policy.role.value)
    if defn.namespace != role_policy.allowed_tool_namespace:
        return Phase12PolicyDecision.deny(
            reason=(
                f"Tool {tool_name!r} belongs to namespace {defn.namespace.value!r}, "
                f"not {role_policy.allowed_tool_namespace.value!r}."
            ),
            failure_code=Phase12FailureCode.ROLE_BLOCKED,
            metadata={
                "tool_name": tool_name,
                "namespace": defn.namespace.value,
                "expected_namespace": role_policy.allowed_tool_namespace.value,
            },
        )
    if expected_role not in defn.allowed_roles:
        return Phase12PolicyDecision.deny(
            reason=f"Role {role_policy.role.value!r} is not allowed to call {tool_name!r}.",
            failure_code=Phase12FailureCode.TOOL_BLOCKED,
            metadata={"tool_name": tool_name},
        )
    if not defn.safe_for_multi_step:
        return Phase12PolicyDecision.deny(
            reason=f"Tool {tool_name!r} is not approved for bounded graph usage.",
            failure_code=Phase12FailureCode.TOOL_BLOCKED,
            metadata={"tool_name": tool_name, "safe_for_multi_step": False},
        )
    return Phase12PolicyDecision.allow(
        reason="tool_allowed",
        metadata={
            "tool_name": tool_name,
            "namespace": defn.namespace.value,
            "safe_for_multi_step": defn.safe_for_multi_step,
        },
    )


def validate_optional_mcp_usage(
    *,
    role: str | ChainRole,
    selected_tools: list[str] | None = None,
) -> Phase12PolicyDecision:
    """Validate whether optional MCP usage is enabled for a role and tool set."""

    role_policy = build_role_policy(role)
    tools = selected_tools or []
    if not role_policy.allow_optional_mcp_tools:
        return Phase12PolicyDecision.deny(
            reason=f"Optional MCP usage is disabled for role {role_policy.role.value!r}.",
            failure_code=Phase12FailureCode.POLICY_BLOCKED,
            metadata={"selected_tools": tools},
        )
    for tool_name in tools:
        decision = validate_tool_access(role=role_policy.role, tool_name=tool_name)
        if not decision.allowed:
            return decision
    return Phase12PolicyDecision.allow(
        reason="optional_mcp_allowed",
        metadata={"selected_tools": tools},
    )


def validate_safe_mode_downgrade(
    *,
    state: Phase12GraphState,
    failure_code: Phase12FailureCode | None = None,
) -> Phase12PolicyDecision:
    """Decide whether the graph may safely downgrade into restricted execution."""

    role_policy = build_role_policy(state.user_role)
    if not role_policy.allow_safe_mode_downgrade:
        return Phase12PolicyDecision.deny(
            reason="Safe-mode downgrade is disabled by policy.",
            failure_code=Phase12FailureCode.POLICY_BLOCKED,
        )
    if state.safe_mode:
        return Phase12PolicyDecision.deny(
            reason="Graph is already operating in safe mode.",
            failure_code=Phase12FailureCode.POLICY_BLOCKED,
        )

    triggers: list[str] = []
    if state.pipeline_context.injection_detected:
        triggers.append("prompt_injection_detected")
    if state.ml_completed and state.ml_confidence < role_policy.safe_mode_ml_confidence_threshold:
        triggers.append("low_ml_confidence")
    if state.rag_completed and state.pipeline_context.rag.weak_retrieval:
        triggers.append("weak_retrieval")
    if state.validation_status in {GraphValidationStatus.INVALID, GraphValidationStatus.REPAIRED}:
        triggers.append("validation_not_clean")
    has_loaded_evidence = bool(state.extracted_text or state.ocr_text or state.transcript_text)
    if has_loaded_evidence and state.evidence_quality_score <= phase12_settings.safe_mode_evidence_quality_threshold:
        triggers.append("low_evidence_quality")
    if failure_code is not None and failure_code in role_policy.safe_mode_failure_codes:
        triggers.append(failure_code.value)

    if not triggers:
        return Phase12PolicyDecision.deny(
            reason="No approved safe-mode trigger was present.",
            failure_code=Phase12FailureCode.POLICY_BLOCKED,
            metadata={"failure_code": failure_code.value if failure_code else None},
        )
    return Phase12PolicyDecision.allow(
        reason="safe_mode_downgrade_allowed",
        metadata={
            "triggers": triggers,
            "confidence_threshold": role_policy.safe_mode_ml_confidence_threshold,
        },
    )


def get_total_step_count(state: Phase12GraphState) -> int:
    """Return the current number of recorded node events."""

    return len(state.node_history)


def get_current_node_retry_count(state: Phase12GraphState, *, node_name: str | None = None) -> int:
    """Return the retry count for the active or requested node."""

    target = node_name or state.current_node
    if not target:
        return 0
    return state.retry_counts.get(target, 0)


def get_repair_loop_count(state: Phase12GraphState) -> int:
    """Count repair-related retries using existing node retry tracking."""

    total = 0
    for node_name, count in state.retry_counts.items():
        normalized = node_name.strip().lower()
        if any(token in normalized for token in ("repair", "validation", "fallback")):
            total += count
    return total


def is_total_step_limit_reached(
    state: Phase12GraphState,
    *,
    limits: Phase12ExecutionLimits | None = None,
) -> bool:
    """Check whether the graph has exhausted its total step budget."""

    active_limits = limits or build_execution_limits()
    return get_total_step_count(state) >= active_limits.max_total_graph_steps


def is_node_retry_limit_reached(
    state: Phase12GraphState,
    *,
    node_name: str | None = None,
    limits: Phase12ExecutionLimits | None = None,
) -> bool:
    """Check whether a node has exhausted its retry budget."""

    active_limits = limits or build_execution_limits()
    retry_count = get_current_node_retry_count(state, node_name=node_name)
    return retry_count > 0 and retry_count >= active_limits.max_retries_per_node


def is_retrieval_retry_limit_reached(
    state: Phase12GraphState,
    *,
    limits: Phase12ExecutionLimits | None = None,
) -> bool:
    """Check whether retrieval retry attempts have been exhausted."""

    active_limits = limits or build_execution_limits()
    return state.retrieval_attempts > 0 and state.retrieval_attempts >= active_limits.max_retrieval_retries


def is_repair_loop_limit_reached(
    state: Phase12GraphState,
    *,
    limits: Phase12ExecutionLimits | None = None,
) -> bool:
    """Check whether repair-loop retries have been exhausted."""

    active_limits = limits or build_execution_limits()
    repair_loops = get_repair_loop_count(state)
    return repair_loops > 0 and repair_loops >= active_limits.max_repair_loops


def is_tool_call_limit_reached(
    state: Phase12GraphState,
    *,
    limits: Phase12ExecutionLimits | None = None,
) -> bool:
    """Check whether bounded tool usage has exceeded policy limits."""

    active_limits = limits or build_execution_limits()
    return state.tool_call_count > 0 and state.tool_call_count >= active_limits.max_tool_calls


def is_repeated_transition_limit_reached(
    repeat_count: int,
    *,
    limits: Phase12ExecutionLimits | None = None,
) -> bool:
    """Check whether repeated state transitions have reached the loop threshold."""

    active_limits = limits or build_execution_limits()
    return repeat_count >= active_limits.max_repeated_state_transitions
