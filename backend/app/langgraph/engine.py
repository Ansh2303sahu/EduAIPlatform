"""Internal Phase 12 graph selector, compiler cache, and execution entry point.

This module stays intentionally thin:
- Selects the compiled graph by role (student or professor).
- Caches compiled graphs at process level (compile once, reuse).
- Exposes ``get_compiled_graph(role)`` for internal callers.
- Runs the compiled graph when ``phase12_settings.enabled`` is True;
  returns a prepared-but-unrun state otherwise (safe default).

No router wiring. No public API changes. No side effects on import.
"""

from __future__ import annotations

from typing import Any

from .config import phase12_settings
from .graphs.professor_graph import get_professor_compiled_graph, get_professor_graph_spec
from .graphs.student_graph import get_student_compiled_graph, get_student_graph_spec
from .policy import Phase12PolicyBundle, build_policy_bundle, validate_graph_for_role
from .schemas import GraphRole, Phase12ExecutionRequest, Phase12GraphSpec
from .state import Phase12GraphState

# ---------------------------------------------------------------------------
# Process-level compiled graph cache
# Populated lazily on first call to get_compiled_graph() for each role.
# ---------------------------------------------------------------------------

_COMPILED_GRAPHS: dict[str, Any] = {}


def _compile_graph_for_role(role: GraphRole) -> Any:
    """Compile the graph for a given role (called once per role per process)."""
    if role == "student":
        return get_student_compiled_graph()
    return get_professor_compiled_graph()


# ---------------------------------------------------------------------------
# Public engine
# ---------------------------------------------------------------------------


class Phase12Engine:
    """Select, compile, and optionally run role-specific Phase 12 graphs.

    Intended for internal use only. Public API contracts (routes, response
    schemas) are not changed here.
    """

    # --- Graph metadata ---

    def describe_graphs(self) -> list[Phase12GraphSpec]:
        return [get_student_graph_spec(), get_professor_graph_spec()]

    def policy_bundle_for(self, role: GraphRole) -> Phase12PolicyBundle:
        """Expose the internal policy bundle for a role-specific graph run."""
        return build_policy_bundle(role)

    def select_graph(self, role: GraphRole) -> Phase12GraphSpec:
        spec = get_student_graph_spec() if role == "student" else get_professor_graph_spec()
        decision = validate_graph_for_role(graph_name=spec.name, role=role)
        if not decision.allowed:
            raise ValueError(decision.reason or "Phase 12 graph selection blocked by policy.")
        return spec

    # --- State preparation ---

    def prepare_state(self, request: Phase12ExecutionRequest) -> Phase12GraphState:
        return Phase12GraphState.create(request)

    # --- Compiled graph access (cached) ---

    def get_compiled_graph(self, role: GraphRole) -> Any:
        """Return the compiled LangGraph for the given role.

        Compiles on first access and caches the result for the lifetime of
        the process. Each role has its own isolated compiled graph instance.
        """
        if role not in _COMPILED_GRAPHS:
            _COMPILED_GRAPHS[role] = _compile_graph_for_role(role)
        return _COMPILED_GRAPHS[role]

    # --- Execution ---

    async def run(self, request: Phase12ExecutionRequest) -> Phase12GraphState:
        """Prepare state and, when Phase 12 is enabled, invoke the compiled graph.

        When ``phase12_settings.enabled`` is False (the current default), this
        method returns the prepared state only. This preserves the existing Phase 10
        behaviour while the graph layer is validated in isolation.

        When enabled, the compiled graph is invoked via ``ainvoke``. LangGraph
        returns a dict; the result is reconstructed as a Phase12GraphState so
        callers always receive a typed state object.
        """
        state = self.prepare_state(request)

        if not phase12_settings.enabled:
            # Safe default: return prepared state without running any nodes.
            # The caller (typically a Phase 10 router shim) handles execution.
            return state

        compiled = self.get_compiled_graph(request.role)
        raw = await compiled.ainvoke(state)

        # LangGraph returns a dict even when the state schema is a Pydantic model.
        # Reconstruct the typed state from the returned dict.
        if isinstance(raw, Phase12GraphState):
            return raw
        if isinstance(raw, dict):
            return Phase12GraphState.model_validate(raw)

        # Defensive: if invocation returned an unexpected type, return the
        # pre-invocation state rather than raising — the graph is additive.
        return state
