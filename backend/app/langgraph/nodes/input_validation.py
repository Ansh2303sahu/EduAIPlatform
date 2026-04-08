"""Initial graph input validation node for Phase 12.

Inspects the internal request, graph identity, and wrapped Phase 10 context for
minimum structural validity. It does not load files, call services, or perform
generation.
"""

from __future__ import annotations

from ..policy import Phase12FailureCode, validate_graph_for_role
from ..schemas import Phase12NodeDescriptor
from ..state import GraphExecutionStatus, Phase12GraphState
from ._helpers import fail_node, succeed_node

NODE_NAME = "input_validation"
NODE_SPEC = Phase12NodeDescriptor(
    name=NODE_NAME,
    description="Validate graph role, identifiers, and minimal PipelineContext shape.",
    wrapped_modules=["app.langgraph.state", "app.langgraph.policy"],
    reads=["request", "pipeline_context", "graph_name"],
    writes=["status", "warnings", "failure_reasons"],
)


async def input_validation_node(state: Phase12GraphState) -> Phase12GraphState:
    """Validate graph identity and minimal internal context structure.

    Inputs inspected:
    - role, file_id, submission_id, user_id
    - graph name / role compatibility
    - wrapped Phase 10 PipelineContext identity fields

    Outputs updated:
    - warnings and failure reasons
    - blocked/failed status when invalid
    - compact trace and decision history

    This node intentionally does not call ingestion, ML, RAG, or MCP services.
    """

    state.set_current_node(NODE_NAME)

    if not state.request.role:
        return fail_node(
            state,
            node_name=NODE_NAME,
            failure_code=Phase12FailureCode.INPUT_INVALID,
            reason="Execution request is missing a role.",
        )
    if not state.user_id:
        return fail_node(
            state,
            node_name=NODE_NAME,
            failure_code=Phase12FailureCode.INPUT_INVALID,
            reason="Execution request is missing user_id.",
        )
    if not state.file_id:
        return fail_node(
            state,
            node_name=NODE_NAME,
            failure_code=Phase12FailureCode.INPUT_INVALID,
            reason="Execution request is missing file_id.",
        )

    graph_decision = validate_graph_for_role(graph_name=state.graph_name, role=state.user_role)
    if not graph_decision.allowed:
        return fail_node(
            state,
            node_name=NODE_NAME,
            failure_code=graph_decision.failure_code or Phase12FailureCode.ROLE_BLOCKED,
            reason=graph_decision.reason or "Graph/role compatibility check failed.",
            final_status=GraphExecutionStatus.BLOCKED,
            detail=graph_decision.metadata,
        )

    ctx = state.pipeline_context
    if ctx is None:
        return fail_node(
            state,
            node_name=NODE_NAME,
            failure_code=Phase12FailureCode.INPUT_INVALID,
            reason="Phase 10 PipelineContext is missing.",
        )
    if not ctx.request_id:
        return fail_node(
            state,
            node_name=NODE_NAME,
            failure_code=Phase12FailureCode.INPUT_INVALID,
            reason="PipelineContext is missing request_id.",
        )
    if not ctx.file_id or ctx.file_id != state.file_id:
        return fail_node(
            state,
            node_name=NODE_NAME,
            failure_code=Phase12FailureCode.INPUT_INVALID,
            reason="PipelineContext file_id is missing or inconsistent with the request.",
        )
    if ctx.role.value != state.role:
        return fail_node(
            state,
            node_name=NODE_NAME,
            failure_code=Phase12FailureCode.ROLE_BLOCKED,
            reason="PipelineContext role does not match the Phase 12 role.",
            final_status=GraphExecutionStatus.BLOCKED,
        )

    analysis_prefix = "student_" if state.role == "student" else "professor_"
    if not state.analysis_type.value.startswith(analysis_prefix):
        return fail_node(
            state,
            node_name=NODE_NAME,
            failure_code=Phase12FailureCode.ROLE_BLOCKED,
            reason="Analysis type does not match the current role boundary.",
            final_status=GraphExecutionStatus.BLOCKED,
        )

    warnings: list[str] = []
    if not state.submission_id:
        warnings.append("Submission ID is not set yet; ingestion may populate it later.")
    if ctx.execution_meta is None:
        return fail_node(
            state,
            node_name=NODE_NAME,
            failure_code=Phase12FailureCode.INPUT_INVALID,
            reason="PipelineContext execution metadata is missing.",
        )
    if warnings:
        for warning in warnings:
            state.add_warning(warning)

    return succeed_node(
        state,
        node_name=NODE_NAME,
        decision_type="input_validation",
        branch="valid",
        reason="Phase 12 state and wrapped Phase 10 context are structurally valid.",
        detail={
            "warnings": warnings,
            "analysis_type": state.analysis_type.value,
            "graph_name": state.graph_name,
        },
    )
