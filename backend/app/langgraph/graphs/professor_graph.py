"""Professor Phase 12 graph — full execution pipeline.

This graph is intentionally role-isolated from the student graph. It shares the
same set of control/execution node implementations (all role-aware internally)
but uses separate routing functions and a separate compiled graph instance.

Node execution order:
  START
    → input_validation     [control]  role/file_id/user_id/graph-role consistency
    → policy_check         [control]  graph access, retrieval policy, tool-call limit
    → ingestion            [execution] file loading + submission-kind detection
    → evidence_sufficiency [control]  classify evidence quality; never blocks
    → ml_decision          [control]  decide whether ML should run; never blocks
    → (if ml_required)
        → ml_inference     [execution] call professor ML service
        → ml_context       [execution] normalize raw ML output → MLContextResult
    → rag_decision         [control]  decide whether RAG should run; may block
    → (if rag_required)
        → retrieval        [execution] professor-audience RAG retrieval
    → safe_mode_decision   [control]  decide whether to enter restricted mode
    → generation           [execution] prompt build + professor LangChain chain
    → validation           [execution] JSON repair, normalize, validate output
    → (if invalid)
        → fallback         [execution] safe fallback payload
    → (if mcp_allowed)
        → mcp_tools        [execution] Phase 11 MCP extension point (opt-in)
                           professor tools: rubric_evaluator, consistency_checker
                           NOTE for Phase 12.3: if mcp_tools modifies
                           pipeline_context.report, insert a re-validation pass
                           here before final_guardrail.
    → final_guardrail      [control]  coherence check; determines terminal status
    → (if completed/partial)
        → persistence      [execution] ai_reports storage — only after guardrail approval
  END [completed | safe_mode_completed | blocked | failed]

Role isolation is enforced by the nodes themselves (PipelineRole check,
analysis_type prefix check, retrieval-audience policy) rather than by
separate node implementations.

The ``_safe_exec`` wrapper and module-reference import pattern are identical
to ``student_graph.py``. The separation ensures professor-specific branching
logic can diverge in future phases without affecting the student graph.

Why final_guardrail runs before persistence (same rationale as student graph):
  BLOCKED/FAILED runs skip persistence to avoid empty audit rows.
  The persisted model_versions.graph_trace includes the complete terminal_trace.
"""

from __future__ import annotations

from typing import Awaitable, Callable

from langgraph.graph import END, START, StateGraph

from ..config import phase12_settings
from ..loop_control import terminal_decision
from ..nodes import (
    evidence_sufficiency_node,
    final_guardrail_node,
    input_validation_node,
    ml_decision_node,
    policy_check_node,
    rag_decision_node,
    safe_mode_decision_node,
)
from ..nodes import fallback as _fallback_mod
from ..nodes import generation as _generation_mod
from ..nodes import ingestion as _ingestion_mod
from ..nodes import mcp_tools as _mcp_tools_mod
from ..nodes import ml_context as _ml_context_mod
from ..nodes import ml_inference as _ml_inference_mod
from ..nodes import persistence as _persistence_mod
from ..nodes import retrieval as _retrieval_mod
from ..nodes import validation as _validation_mod
from ..schemas import Phase12GraphSpec
from ..state import FailureType, GraphExecutionStatus, Phase12GraphState
from ..tracing.graph_trace import record_error, record_event

# ---------------------------------------------------------------------------
# Hard-stop statuses — routing functions exit immediately when these are set
# ---------------------------------------------------------------------------

_HARD_STOP_STATUSES = frozenset({
    GraphExecutionStatus.BLOCKED,
    GraphExecutionStatus.FAILED,
})


# ---------------------------------------------------------------------------
# _safe_exec — wraps every execution node with loop-prevention + error capture
# ---------------------------------------------------------------------------

async def _safe_exec(
    fn: Callable[[Phase12GraphState], Awaitable[Phase12GraphState]],
    state: Phase12GraphState,
    node_name: str,
) -> Phase12GraphState:
    """Execute an execution node with terminal guardrail checking and error capture.

    Before calling ``fn``:
    - Checks ``terminal_decision(state).should_stop``; skips the node if True.

    On exception:
    - Records a compact failure reason and a structured error trace entry.
    - Returns state so the graph continues to degrade gracefully.
    """
    guardrail = terminal_decision(state)
    if guardrail.should_stop:
        state.add_warning(
            f"{node_name} skipped: terminal guardrail active ({guardrail.reason or 'limit reached'})."
        )
        record_event(state, node_name, {"skipped": True, "reason": "terminal_guardrail"})
        return state
    try:
        return await fn(state)
    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {str(exc)[:160]}"
        state.add_failure_reason(err_msg, failure_type=FailureType.UNKNOWN)
        record_error(state, node_name, err_msg, failure_code=f"phase12.{node_name}.error")
        return state


# ---------------------------------------------------------------------------
# Execution node wrappers (resolve .run attribute at call time for patchability)
# ---------------------------------------------------------------------------

async def _ingestion_node(state: Phase12GraphState) -> Phase12GraphState:
    return await _safe_exec(_ingestion_mod.run, state, "ingestion")


async def _input_validation_node(state: Phase12GraphState) -> Phase12GraphState:
    state.graph_name = phase12_settings.professor_graph_name
    return await input_validation_node(state)


async def _ml_inference_node(state: Phase12GraphState) -> Phase12GraphState:
    return await _safe_exec(_ml_inference_mod.run, state, "ml_inference")


async def _ml_context_node(state: Phase12GraphState) -> Phase12GraphState:
    return await _safe_exec(_ml_context_mod.run, state, "ml_context")


async def _retrieval_node(state: Phase12GraphState) -> Phase12GraphState:
    return await _safe_exec(_retrieval_mod.run, state, "retrieval")


async def _generation_node(state: Phase12GraphState) -> Phase12GraphState:
    return await _safe_exec(_generation_mod.run, state, "generation")


async def _validation_node(state: Phase12GraphState) -> Phase12GraphState:
    return await _safe_exec(_validation_mod.run, state, "validation")


async def _fallback_node(state: Phase12GraphState) -> Phase12GraphState:
    return await _safe_exec(_fallback_mod.run, state, "fallback")


async def _mcp_tools_node(state: Phase12GraphState) -> Phase12GraphState:
    return await _safe_exec(_mcp_tools_mod.run, state, "mcp_tools")


async def _persistence_node(state: Phase12GraphState) -> Phase12GraphState:
    return await _safe_exec(_persistence_mod.run, state, "persistence")


# ---------------------------------------------------------------------------
# Conditional routing functions — professor-specific instances
# Kept separate from student routing to allow role-specific divergence.
# ---------------------------------------------------------------------------

def _route_after_input_validation(state: Phase12GraphState) -> str:
    """Exit immediately on role/identity/graph-compatibility failures.

    For professors, input_validation also confirms the graph name starts with
    'phase12_professor_graph' and that analysis_type starts with 'professor_'.
    """
    if state.status in _HARD_STOP_STATUSES or state.final_status in _HARD_STOP_STATUSES:
        return "end"
    return "policy_check"


def _route_after_policy_check(state: Phase12GraphState) -> str:
    """Exit immediately on graph-access or retrieval-policy block.

    Professor policy permits a wider retrieval audience and optionally allows
    MCP tools (rubric_evaluator, consistency_checker).
    """
    if state.status in _HARD_STOP_STATUSES or state.final_status in _HARD_STOP_STATUSES:
        return "end"
    return "ingestion"


def _route_after_ml_decision(state: Phase12GraphState) -> str:
    """Branch to ML inference or skip directly to RAG decision."""
    if state.ml_required:
        return "ml_inference"
    return "rag_decision"


def _route_after_rag_decision(state: Phase12GraphState) -> str:
    """Exit on retrieval-audience policy block; otherwise branch to retrieval or skip.

    ``rag_decision_node`` enforces professor-audience access and is the last
    control node that can hard-block execution.
    """
    if (
        state.status == GraphExecutionStatus.BLOCKED
        or state.final_status == GraphExecutionStatus.BLOCKED
    ):
        return "end"
    if state.rag_required:
        return "retrieval"
    return "safe_mode_decision"


def _mcp_allowed(state: Phase12GraphState) -> bool:
    """Return True when professor MCP tool use is permitted by settings."""
    return (
        phase12_settings.allow_mcp_tools
        and phase12_settings.allow_professor_mcp_tools
        and bool(state.selected_tools)
    )


def _route_after_validation(state: Phase12GraphState) -> str:
    """Branch to fallback (invalid), mcp_tools (opt-in), or directly to final_guardrail."""
    if not state.pipeline_context.validation_result.valid:
        return "fallback"
    if _mcp_allowed(state):
        return "mcp_tools"
    return "final_guardrail"


def _route_after_final_guardrail(state: Phase12GraphState) -> str:
    """Gate persistence on guardrail-approved terminal status.

    BLOCKED and FAILED skip persistence — no usable report exists. COMPLETED and
    PARTIAL both have a guardrail-approved report. Professor reports use different
    expected_keys (summary + feedback_explanation + safety vs student's issues +
    strengths); the guardrail node handles this distinction internally via role.
    At this point state.terminal_trace is fully populated by finalize_trace(), so
    the persisted model_versions includes the complete graph trace.
    """
    fs = state.final_status or state.status
    if fs in (GraphExecutionStatus.BLOCKED, GraphExecutionStatus.FAILED):
        return "end"
    return "persistence"


def _route_after_persistence(state: Phase12GraphState) -> str:
    """Map the persisted terminal status to a named exit label for trace clarity.

    All three branches resolve to END. PARTIAL + safe_mode → safe_mode_completed;
    PARTIAL without safe_mode → failed (the guardrail would have emitted COMPLETED
    for a clean, non-safe-mode run).
    """
    fs = state.final_status or state.status
    if fs == GraphExecutionStatus.COMPLETED:
        return "completed"
    if fs == GraphExecutionStatus.PARTIAL and state.safe_mode:
        return "safe_mode_completed"
    return "failed"


# ---------------------------------------------------------------------------
# Graph builder — returns an uncompiled StateGraph for inspection and testing
# ---------------------------------------------------------------------------

def _build_professor_graph() -> StateGraph:
    """Assemble the full professor Phase 12 StateGraph without compiling.

    The graph uses the same shared node implementations as the student graph.
    Role isolation is enforced at the node level (PipelineRole check,
    analysis_type prefix, retrieval-audience policy).
    """
    g = StateGraph(Phase12GraphState)

    # --- Control nodes (no external I/O) ---
    g.add_node("input_validation", _input_validation_node)
    g.add_node("policy_check", policy_check_node)
    g.add_node("evidence_sufficiency", evidence_sufficiency_node)
    g.add_node("ml_decision", ml_decision_node)
    g.add_node("rag_decision", rag_decision_node)
    g.add_node("safe_mode_decision", safe_mode_decision_node)
    g.add_node("final_guardrail", final_guardrail_node)

    # --- Execution nodes (all wrapped via _safe_exec) ---
    g.add_node("ingestion", _ingestion_node)
    g.add_node("ml_inference", _ml_inference_node)
    g.add_node("ml_context", _ml_context_node)
    g.add_node("retrieval", _retrieval_node)
    g.add_node("generation", _generation_node)
    g.add_node("validation", _validation_node)
    g.add_node("fallback", _fallback_node)
    g.add_node("mcp_tools", _mcp_tools_node)
    g.add_node("persistence", _persistence_node)

    # --- Entrypoint ---
    g.add_edge(START, "input_validation")

    # --- input_validation → policy_check (or hard stop) ---
    g.add_conditional_edges(
        "input_validation",
        _route_after_input_validation,
        {"end": END, "policy_check": "policy_check"},
    )

    # --- policy_check → ingestion (or hard stop) ---
    g.add_conditional_edges(
        "policy_check",
        _route_after_policy_check,
        {"end": END, "ingestion": "ingestion"},
    )

    # --- ingestion → evidence_sufficiency → ml_decision (unconditional) ---
    g.add_edge("ingestion", "evidence_sufficiency")
    g.add_edge("evidence_sufficiency", "ml_decision")

    # --- ml_decision → ml_inference | rag_decision ---
    g.add_conditional_edges(
        "ml_decision",
        _route_after_ml_decision,
        {"ml_inference": "ml_inference", "rag_decision": "rag_decision"},
    )

    # --- ml_inference → ml_context → rag_decision (unconditional) ---
    g.add_edge("ml_inference", "ml_context")
    g.add_edge("ml_context", "rag_decision")

    # --- rag_decision → retrieval | safe_mode_decision (or hard stop) ---
    g.add_conditional_edges(
        "rag_decision",
        _route_after_rag_decision,
        {"end": END, "retrieval": "retrieval", "safe_mode_decision": "safe_mode_decision"},
    )

    # --- retrieval → safe_mode_decision → generation (unconditional) ---
    g.add_edge("retrieval", "safe_mode_decision")
    g.add_edge("safe_mode_decision", "generation")

    # --- generation → validation (unconditional; _safe_exec handles failures) ---
    g.add_edge("generation", "validation")

    # --- validation → fallback | mcp_tools | final_guardrail ---
    g.add_conditional_edges(
        "validation",
        _route_after_validation,
        {"fallback": "fallback", "mcp_tools": "mcp_tools", "final_guardrail": "final_guardrail"},
    )

    # --- fallback → final_guardrail (unconditional; one bounded repair pass) ---
    g.add_edge("fallback", "final_guardrail")

    # --- mcp_tools → final_guardrail (unconditional) ---
    # Phase 12.3 note: if mcp_tools modifies pipeline_context.report, insert a
    # re-validation node here before final_guardrail instead of this direct edge.
    g.add_edge("mcp_tools", "final_guardrail")

    # --- final_guardrail → persistence (COMPLETED/PARTIAL) or END (BLOCKED/FAILED) ---
    g.add_conditional_edges(
        "final_guardrail",
        _route_after_final_guardrail,
        {"end": END, "persistence": "persistence"},
    )

    # --- persistence → END (terminal label for trace clarity) ---
    g.add_conditional_edges(
        "persistence",
        _route_after_persistence,
        {"completed": END, "safe_mode_completed": END, "failed": END},
    )

    return g


def get_professor_compiled_graph():
    """Compile and return the full professor Phase 12 graph.

    The compiled graph accepts a Phase12GraphState (professor role) and after
    invocation returns a dict reconstructable via ``Phase12GraphState.model_validate()``.
    Callers should cache this result — compilation validates the full topology.
    """
    return _build_professor_graph().compile()


# ---------------------------------------------------------------------------
# Static spec — preserved for Phase12Engine.select_graph() compatibility
# ---------------------------------------------------------------------------

PROFESSOR_GRAPH_SPEC = Phase12GraphSpec(
    name="phase12_professor_graph",
    role="professor",
    description=(
        "Full execution LangGraph for the professor pipeline. "
        "Runs policy, input validation, file ingestion, evidence classification, "
        "ML inference, RAG retrieval, safe-mode decision, generation, "
        "output validation, optional fallback and MCP extension "
        "(rubric_evaluator, consistency_checker), and persistence."
    ),
    entrypoint="input_validation",
    node_order=[
        "input_validation",
        "policy_check",
        "ingestion",
        "evidence_sufficiency",
        "ml_decision",
        "ml_inference",
        "ml_context",
        "rag_decision",
        "retrieval",
        "safe_mode_decision",
        "generation",
        "validation",
        "fallback",
        "mcp_tools",
        "final_guardrail",
        "persistence",
    ],
    terminal_nodes=["final_guardrail"],
    optional_nodes=["ml_inference", "ml_context", "retrieval", "fallback", "mcp_tools"],
    wrapped_modules=[
        "app.langgraph.nodes.input_validation",
        "app.langgraph.nodes.policy_check",
        "app.langgraph.nodes.ingestion",
        "app.langgraph.nodes.evidence_sufficiency",
        "app.langgraph.nodes.ml_decision",
        "app.langgraph.nodes.ml_inference",
        "app.langgraph.nodes.ml_context",
        "app.langgraph.nodes.rag_decision",
        "app.langgraph.nodes.retrieval",
        "app.langgraph.nodes.safe_mode_decision",
        "app.langgraph.nodes.generation",
        "app.langgraph.nodes.validation",
        "app.langgraph.nodes.fallback",
        "app.langgraph.nodes.mcp_tools",
        "app.langgraph.nodes.persistence",
        "app.langgraph.nodes.final_guardrail",
    ],
)


def get_professor_graph_spec() -> Phase12GraphSpec:
    """Return the deterministic professor graph specification."""
    return PROFESSOR_GRAPH_SPEC
