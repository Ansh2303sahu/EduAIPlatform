"""Phase 12 high-level policy check node.

Applies the role-specific Phase 12 policy bundle to graph selection, retrieval
audience access, and optional MCP eligibility without calling any tools.
"""

from __future__ import annotations

from ..policy import (
    Phase12FailureCode,
    build_policy_bundle,
    is_tool_call_limit_reached,
    validate_graph_for_role,
    validate_optional_mcp_usage,
    validate_retrieval_access,
)
from ..schemas import Phase12NodeDescriptor
from ..state import GraphExecutionStatus, Phase12GraphState
from ._helpers import fail_node, succeed_node

NODE_NAME = "policy_check"
NODE_SPEC = Phase12NodeDescriptor(
    name=NODE_NAME,
    description="Apply role-aware graph, retrieval, and optional MCP policy checks.",
    wrapped_modules=["app.langgraph.policy"],
    reads=["graph_name", "user_role", "selected_tools"],
    writes=["status", "warnings", "failure_reasons"],
)


async def policy_check_node(state: Phase12GraphState) -> Phase12GraphState:
    """Apply the current Phase 12 policy bundle at a high level.

    Inputs inspected:
    - graph/role pairing
    - retrieval audience
    - optional tool selection and bounded tool-call limit

    Outputs updated:
    - warnings and blocked/failure reasons
    - compact trace and decision history

    This node intentionally does not execute MCP or retrieval.
    """

    state.set_current_node(NODE_NAME)
    policy_bundle = build_policy_bundle(state.user_role)

    graph_decision = validate_graph_for_role(graph_name=state.graph_name, role=state.user_role)
    if not graph_decision.allowed:
        return fail_node(
            state,
            node_name=NODE_NAME,
            failure_code=graph_decision.failure_code or Phase12FailureCode.ROLE_BLOCKED,
            reason=graph_decision.reason or "Graph policy check failed.",
            final_status=GraphExecutionStatus.BLOCKED,
            detail=graph_decision.metadata,
        )

    retrieval_decision = validate_retrieval_access(
        role=state.user_role,
        requested_audience=state.role,
    )
    if not retrieval_decision.allowed:
        return fail_node(
            state,
            node_name=NODE_NAME,
            failure_code=retrieval_decision.failure_code or Phase12FailureCode.ROLE_BLOCKED,
            reason=retrieval_decision.reason or "Retrieval access is not allowed for this role.",
            final_status=GraphExecutionStatus.BLOCKED,
            detail=retrieval_decision.metadata,
        )

    if state.selected_tools:
        tool_decision = validate_optional_mcp_usage(
            role=state.user_role,
            selected_tools=state.selected_tools,
        )
        if not tool_decision.allowed:
            return fail_node(
                state,
                node_name=NODE_NAME,
                failure_code=tool_decision.failure_code or Phase12FailureCode.POLICY_BLOCKED,
                reason=tool_decision.reason or "Optional MCP usage is not allowed.",
                final_status=GraphExecutionStatus.BLOCKED,
                detail=tool_decision.metadata,
            )

    if is_tool_call_limit_reached(state, limits=policy_bundle.limits):
        return fail_node(
            state,
            node_name=NODE_NAME,
            failure_code=Phase12FailureCode.TOOL_BLOCKED,
            reason="Phase 12 tool-call limit has already been exceeded.",
            final_status=GraphExecutionStatus.BLOCKED,
        )

    return succeed_node(
        state,
        node_name=NODE_NAME,
        decision_type="policy_check",
        branch="allowed",
        reason="Role-aware Phase 12 policy checks passed.",
        detail={
            "retrieval_audience": policy_bundle.role_policy.retrieval_audience,
            "allow_optional_mcp_tools": policy_bundle.role_policy.allow_optional_mcp_tools,
            "selected_tool_count": len(state.selected_tools),
        },
    )
