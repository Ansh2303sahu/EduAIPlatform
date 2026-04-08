"""Optional future MCP bridge node for Phase 12.

This node is intentionally not part of the default student/professor graph
order. When Phase 12 later needs bounded tool use, it should go through the
existing Phase 11 executor/orchestrator instead of calling tool handlers
directly.
"""

from typing import Any

from ..adapters import mcp
from ..schemas import Phase12NodeDescriptor
from ..state import Phase12GraphState
from ..tracing.graph_trace import record_event

NODE_NAME = "mcp_tools"
NODE_SPEC = Phase12NodeDescriptor(
    name=NODE_NAME,
    description="Optional future bridge to the existing Phase 11 MCP executor/orchestrator.",
    wrapped_modules=[
        "app.mcp.executor",
        "app.mcp.orchestrator",
    ],
    reads=["request.user_id", "request.role", "pipeline_context.file_id"],
    writes=["mcp_results"],
    optional=True,
)


async def run(state: Phase12GraphState) -> Phase12GraphState:
    """Graph node entry point — MCP extension placeholder for Phase 12.3.

    When ``allow_mcp_tools`` is True and the routing function sends execution
    here, this node records which tools were selected and returns state unchanged.
    Actual tool invocation is reserved for Phase 12.3.
    """
    record_event(
        state,
        NODE_NAME,
        {
            "note": "mcp_extension_point_placeholder",
            "selected_tools": list(state.selected_tools),
            "role": state.role,
        },
    )
    return state


async def run_tool(
    state: Phase12GraphState,
    *,
    tool_name: str,
    payload: dict[str, Any],
) -> Phase12GraphState:
    """Execute a single MCP tool through the standard Phase 11 executor."""

    result = await mcp.execute_tool_for_state(state, tool_name=tool_name, payload=payload)
    state.mcp_results.append(result)
    record_event(state, NODE_NAME, {"tool_name": tool_name})
    return state
