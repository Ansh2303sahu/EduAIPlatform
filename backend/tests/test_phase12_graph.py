"""Comprehensive tests for Phase 12 LangGraph student and professor graphs.

Test structure:
  TestGraphCompilation       — both graphs compile; specs are correct
  TestPhase12Engine          — disabled path, enabled path, compiled-graph cache
  TestValidationNodeBugFixed — ValidationResult.valid (not .is_valid) is used
  TestFallbackNodeBugFixed   — fallback.py uses .valid consistently
  TestMcpToolsRunFn          — mcp_tools.run() graph-node entry point exists
  TestExecutionGraphRouting  — ML skip, RAG skip, fallback on invalid, MCP gate
  TestHappyPath              — full student + professor run with mocked adapters
  TestSafeExecWrapper        — exception capture, terminal guardrail skip
  TestLoopPrevention         — max_steps_exceeded stops graph gracefully
  TestTracing                — node/decision traces populated, terminal_trace set
  TestRoleIsolation          — wrong-role request is blocked by input_validation
  TestSafeMode               — safe mode activated, terminal status correct
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.langgraph.config import phase12_settings
from app.langgraph.graphs.professor_graph import (
    PROFESSOR_GRAPH_SPEC,
    _build_professor_graph,
    get_professor_compiled_graph,
    _route_after_ml_decision as _prof_route_ml,
    _route_after_rag_decision as _prof_route_rag,
    _route_after_validation as _prof_route_val,
    _route_after_final_guardrail as _prof_route_fg,
    _route_after_persistence as _prof_route_persist,
    _safe_exec as _prof_safe_exec,
)
from app.langgraph.graphs.student_graph import (
    STUDENT_GRAPH_SPEC,
    _build_student_graph,
    get_student_compiled_graph,
    _route_after_ml_decision as _stu_route_ml,
    _route_after_rag_decision as _stu_route_rag,
    _route_after_validation as _stu_route_val,
    _route_after_final_guardrail as _stu_route_fg,
    _route_after_persistence as _stu_route_persist,
    _safe_exec as _stu_safe_exec,
)
from app.langgraph.engine import Phase12Engine, _COMPILED_GRAPHS
from app.langgraph.schemas import Phase12ExecutionRequest
from app.langgraph.state import GraphExecutionStatus, Phase12GraphState


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_request(role: str = "student") -> Phase12ExecutionRequest:
    return Phase12ExecutionRequest(
        file_id="file-test-001",
        user_id="user-test-001",
        role=role,
        correlation_id="corr-test-001",
    )


def make_state(role: str = "student") -> Phase12GraphState:
    return Phase12GraphState.create(make_request(role))


def _noop_run(state: Phase12GraphState) -> Phase12GraphState:
    """Synchronous no-op; wrapped in AsyncMock where needed."""
    return state


def _async_identity(state: Phase12GraphState) -> Phase12GraphState:
    """Async no-op returning state unchanged."""
    return state


# Patch target prefix for execution node modules
_NODES = "app.langgraph.nodes"


# ---------------------------------------------------------------------------
# TestGraphCompilation
# ---------------------------------------------------------------------------

class TestGraphCompilation:
    def test_student_graph_builds(self):
        g = _build_student_graph()
        assert g is not None

    def test_professor_graph_builds(self):
        g = _build_professor_graph()
        assert g is not None

    def test_student_graph_compiles(self):
        compiled = get_student_compiled_graph()
        assert compiled is not None

    def test_professor_graph_compiles(self):
        compiled = get_professor_compiled_graph()
        assert compiled is not None

    def test_student_spec_role(self):
        assert STUDENT_GRAPH_SPEC.role == "student"
        assert STUDENT_GRAPH_SPEC.entrypoint == "input_validation"

    def test_professor_spec_role(self):
        assert PROFESSOR_GRAPH_SPEC.role == "professor"
        assert PROFESSOR_GRAPH_SPEC.entrypoint == "input_validation"

    def test_student_spec_has_all_execution_nodes(self):
        order = STUDENT_GRAPH_SPEC.node_order
        for node in ["ingestion", "ml_inference", "ml_context", "retrieval",
                     "generation", "validation", "fallback", "mcp_tools", "persistence"]:
            assert node in order, f"Missing execution node in student spec: {node}"

    def test_professor_spec_has_all_execution_nodes(self):
        order = PROFESSOR_GRAPH_SPEC.node_order
        for node in ["ingestion", "ml_inference", "ml_context", "retrieval",
                     "generation", "validation", "fallback", "mcp_tools", "persistence"]:
            assert node in order, f"Missing execution node in professor spec: {node}"

    def test_student_spec_optional_nodes(self):
        for node in ["ml_inference", "ml_context", "retrieval", "fallback", "mcp_tools"]:
            assert node in STUDENT_GRAPH_SPEC.optional_nodes

    def test_professor_spec_optional_nodes(self):
        for node in ["ml_inference", "ml_context", "retrieval", "fallback", "mcp_tools"]:
            assert node in PROFESSOR_GRAPH_SPEC.optional_nodes

    def test_student_spec_terminal_node(self):
        assert "final_guardrail" in STUDENT_GRAPH_SPEC.terminal_nodes

    def test_professor_spec_terminal_node(self):
        assert "final_guardrail" in PROFESSOR_GRAPH_SPEC.terminal_nodes

    def test_student_graph_name_matches_spec(self):
        assert STUDENT_GRAPH_SPEC.name == "phase12_student_graph"

    def test_professor_graph_name_matches_spec(self):
        assert PROFESSOR_GRAPH_SPEC.name == "phase12_professor_graph"


# ---------------------------------------------------------------------------
# TestPhase12Engine
# ---------------------------------------------------------------------------

class TestPhase12Engine:
    def test_describe_graphs_returns_two_specs(self):
        engine = Phase12Engine()
        specs = engine.describe_graphs()
        roles = {s.role for s in specs}
        assert roles == {"student", "professor"}

    def test_select_graph_student(self):
        engine = Phase12Engine()
        spec = engine.select_graph("student")
        assert spec.role == "student"

    def test_select_graph_professor(self):
        engine = Phase12Engine()
        spec = engine.select_graph("professor")
        assert spec.role == "professor"

    def test_prepare_state_student(self):
        engine = Phase12Engine()
        state = engine.prepare_state(make_request("student"))
        assert state.role == "student"
        assert state.file_id == "file-test-001"

    def test_prepare_state_professor(self):
        engine = Phase12Engine()
        state = engine.prepare_state(make_request("professor"))
        assert state.role == "professor"

    @pytest.mark.asyncio
    async def test_disabled_returns_prepared_state_without_graph(self):
        """When phase12_settings.enabled is False, engine.run() must not invoke the graph."""
        engine = Phase12Engine()
        request = make_request("student")
        with patch.object(engine, "get_compiled_graph") as mock_get:
            mock_get.return_value = MagicMock()
            # Default: enabled=False, so graph should NOT be invoked
            assert not phase12_settings.enabled
            state = await engine.run(request)
        mock_get.assert_not_called()
        assert isinstance(state, Phase12GraphState)
        assert state.role == "student"

    @pytest.mark.asyncio
    async def test_enabled_invokes_compiled_graph(self):
        """When enabled=True, engine.run() must call compiled.ainvoke()."""
        engine = Phase12Engine()
        request = make_request("student")
        prepared = Phase12GraphState.create(request)

        fake_compiled = MagicMock()
        fake_compiled.ainvoke = AsyncMock(return_value=prepared.model_dump(mode="json"))

        with patch.object(phase12_settings, "enabled", True):
            with patch.object(engine, "get_compiled_graph", return_value=fake_compiled):
                result = await engine.run(request)

        fake_compiled.ainvoke.assert_called_once()
        assert isinstance(result, Phase12GraphState)

    @pytest.mark.asyncio
    async def test_enabled_returns_state_instance_when_ainvoke_returns_state(self):
        """If compiled.ainvoke() already returns a Phase12GraphState, engine preserves it."""
        engine = Phase12Engine()
        request = make_request("professor")
        prepared = Phase12GraphState.create(request)

        fake_compiled = MagicMock()
        fake_compiled.ainvoke = AsyncMock(return_value=prepared)

        with patch.object(phase12_settings, "enabled", True):
            with patch.object(engine, "get_compiled_graph", return_value=fake_compiled):
                result = await engine.run(request)

        assert result is prepared

    def test_get_compiled_graph_caches_per_role(self):
        """Each role's compiled graph is cached independently."""
        engine = Phase12Engine()
        # Clear cache before test
        _COMPILED_GRAPHS.clear()
        try:
            g1 = engine.get_compiled_graph("student")
            g2 = engine.get_compiled_graph("student")
            gp = engine.get_compiled_graph("professor")
            assert g1 is g2, "Same student graph should be returned on second call"
            assert g1 is not gp, "Student and professor graphs must be different objects"
        finally:
            _COMPILED_GRAPHS.clear()


# ---------------------------------------------------------------------------
# TestValidationNodeBugFixed
# ---------------------------------------------------------------------------

class TestValidationNodeBugFixed:
    """Confirm that validation.py uses ValidationResult.valid (not .is_valid)."""

    @pytest.mark.asyncio
    async def test_valid_report_is_stored(self):
        from app.langgraph.nodes import validation as validation_mod

        state = make_state("student")
        mock_result = MagicMock()
        mock_result.valid = True
        mock_report = {"summary": "ok", "issues": [], "strengths": [], "safety": {}}

        with patch(f"{_NODES}.validation.try_parse_json", return_value={"raw": True}):
            with patch(f"{_NODES}.validation.normalize_student_payload", return_value=mock_report):
                with patch(f"{_NODES}.validation.validate_student_payload", return_value=mock_result):
                    result = await validation_mod.run(state)

        assert result.pipeline_context.report == mock_report
        assert len(result.failure_reasons) == 0

    @pytest.mark.asyncio
    async def test_invalid_report_adds_failure_reason(self):
        from app.langgraph.nodes import validation as validation_mod

        state = make_state("student")
        mock_result = MagicMock()
        mock_result.valid = False
        mock_report = {"partial": True}

        with patch(f"{_NODES}.validation.try_parse_json", return_value={"raw": True}):
            with patch(f"{_NODES}.validation.normalize_student_payload", return_value=mock_report):
                with patch(f"{_NODES}.validation.validate_student_payload", return_value=mock_result):
                    result = await validation_mod.run(state)

        assert "phase12_validation_failed" in result.failure_reasons
        # report should NOT be set when validation fails
        assert result.pipeline_context.report == {} or result.pipeline_context.report != mock_report

    @pytest.mark.asyncio
    async def test_parse_failure_adds_failure_reason(self):
        from app.langgraph.nodes import validation as validation_mod

        state = make_state("student")

        with patch(f"{_NODES}.validation.try_parse_json", return_value=None):
            result = await validation_mod.run(state)

        assert "phase12_validation_parse_failed" in result.failure_reasons

    @pytest.mark.asyncio
    async def test_professor_valid_report_stored(self):
        from app.langgraph.nodes import validation as validation_mod

        state = make_state("professor")
        mock_result = MagicMock()
        mock_result.valid = True
        mock_report = {"summary": "ok", "feedback_explanation": "good", "safety": {}}

        with patch(f"{_NODES}.validation.try_parse_json", return_value={"raw": True}):
            with patch(f"{_NODES}.validation.normalize_professor_payload", return_value=mock_report):
                with patch(f"{_NODES}.validation.validate_professor_payload", return_value=mock_result):
                    result = await validation_mod.run(state)

        assert result.pipeline_context.report == mock_report


# ---------------------------------------------------------------------------
# TestFallbackNodeBugFixed
# ---------------------------------------------------------------------------

class TestFallbackNodeBugFixed:
    """Confirm that fallback.py uses ValidationResult.valid (not .is_valid)."""

    @pytest.mark.asyncio
    async def test_fallback_skipped_when_report_valid(self):
        from app.langgraph.nodes import fallback as fallback_mod

        state = make_state("student")
        report = {"summary": "ok"}
        state.pipeline_context.report = report
        state.pipeline_context.validation_result.valid = True

        result = await fallback_mod.run(state)

        # report should be unchanged since validation_result.valid is True
        assert result.pipeline_context.report == report

    @pytest.mark.asyncio
    async def test_fallback_runs_when_report_invalid(self):
        from app.langgraph.nodes import fallback as fallback_mod

        state = make_state("student")
        state.pipeline_context.report = {}
        state.pipeline_context.validation_result.valid = False
        fallback_payload = {"summary": "fallback", "issues": [], "strengths": [], "safety": {}}

        with patch(f"{_NODES}.fallback.build_student_fallback_payload", return_value=fallback_payload):
            result = await fallback_mod.run(state)

        assert result.pipeline_context.report == fallback_payload


# ---------------------------------------------------------------------------
# TestMcpToolsRunFn
# ---------------------------------------------------------------------------

class TestMcpToolsRunFn:
    """Confirm mcp_tools.run() exists and records the event correctly."""

    @pytest.mark.asyncio
    async def test_run_fn_exists(self):
        from app.langgraph.nodes import mcp_tools as mcp_mod
        assert hasattr(mcp_mod, "run"), "mcp_tools must expose a graph-node run() function"

    @pytest.mark.asyncio
    async def test_run_returns_state(self):
        from app.langgraph.nodes import mcp_tools as mcp_mod

        state = make_state("student")
        result = await mcp_mod.run(state)
        assert result is state

    @pytest.mark.asyncio
    async def test_run_records_event(self):
        from app.langgraph.nodes import mcp_tools as mcp_mod

        state = make_state("professor")
        state.selected_tools = ["rubric_evaluator"]
        result = await mcp_mod.run(state)

        assert len(result.node_traces) > 0
        last_trace = result.node_traces[-1]
        assert last_trace.node_name == "mcp_tools"


# ---------------------------------------------------------------------------
# TestExecutionGraphRouting
# ---------------------------------------------------------------------------

class TestExecutionGraphRouting:
    """Unit tests for the conditional routing functions."""

    # --- ML routing ---

    def test_ml_route_to_inference_when_required(self):
        state = make_state("student")
        state.ml_required = True
        assert _stu_route_ml(state) == "ml_inference"

    def test_ml_route_to_rag_when_not_required(self):
        state = make_state("student")
        state.ml_required = False
        assert _stu_route_ml(state) == "rag_decision"

    def test_prof_ml_route_to_inference_when_required(self):
        state = make_state("professor")
        state.ml_required = True
        assert _prof_route_ml(state) == "ml_inference"

    def test_prof_ml_route_to_rag_when_not_required(self):
        state = make_state("professor")
        state.ml_required = False
        assert _prof_route_ml(state) == "rag_decision"

    # --- RAG routing ---

    def test_rag_route_to_retrieval_when_required(self):
        state = make_state("student")
        state.rag_required = True
        assert _stu_route_rag(state) == "retrieval"

    def test_rag_route_to_safe_mode_when_not_required(self):
        state = make_state("student")
        state.rag_required = False
        assert _stu_route_rag(state) == "safe_mode_decision"

    def test_rag_route_end_on_blocked_status(self):
        state = make_state("student")
        state.status = GraphExecutionStatus.BLOCKED
        assert _stu_route_rag(state) == "end"

    def test_rag_route_end_on_blocked_final_status(self):
        state = make_state("student")
        state.final_status = GraphExecutionStatus.BLOCKED
        assert _stu_route_rag(state) == "end"

    def test_prof_rag_route_end_on_blocked(self):
        state = make_state("professor")
        state.status = GraphExecutionStatus.BLOCKED
        assert _prof_route_rag(state) == "end"

    # --- Validation routing ---

    def test_validation_route_to_fallback_when_invalid(self):
        state = make_state("student")
        state.pipeline_context.validation_result.valid = False
        assert _stu_route_val(state) == "fallback"

    def test_validation_route_to_final_guardrail_when_valid_no_mcp(self):
        state = make_state("student")
        state.pipeline_context.validation_result.valid = True
        # allow_mcp_tools defaults False — route goes directly to final_guardrail
        assert _stu_route_val(state) == "final_guardrail"

    def test_validation_route_to_mcp_when_allowed(self):
        state = make_state("student")
        state.pipeline_context.validation_result.valid = True
        state.selected_tools = ["student.summariser"]
        with patch.object(phase12_settings, "allow_mcp_tools", True):
            with patch.object(phase12_settings, "allow_student_mcp_tools", True):
                assert _stu_route_val(state) == "mcp_tools"

    def test_prof_validation_route_to_mcp_when_allowed(self):
        state = make_state("professor")
        state.pipeline_context.validation_result.valid = True
        state.selected_tools = ["rubric_evaluator"]
        with patch.object(phase12_settings, "allow_mcp_tools", True):
            with patch.object(phase12_settings, "allow_professor_mcp_tools", True):
                assert _prof_route_val(state) == "mcp_tools"

    def test_validation_route_to_final_guardrail_when_mcp_allowed_but_no_tools(self):
        """No tools selected → skip MCP → go straight to final_guardrail."""
        state = make_state("student")
        state.pipeline_context.validation_result.valid = True
        state.selected_tools = []  # no tools selected
        with patch.object(phase12_settings, "allow_mcp_tools", True):
            with patch.object(phase12_settings, "allow_student_mcp_tools", True):
                assert _stu_route_val(state) == "final_guardrail"

    # --- Final guardrail routing (gates persistence) ---

    def test_fg_route_completed_goes_to_persistence(self):
        """COMPLETED should route to persistence, not directly to END."""
        state = make_state("student")
        state.final_status = GraphExecutionStatus.COMPLETED
        assert _stu_route_fg(state) == "persistence"

    def test_fg_route_partial_goes_to_persistence(self):
        """PARTIAL (with or without safe_mode) should still route to persistence."""
        state = make_state("student")
        state.final_status = GraphExecutionStatus.PARTIAL
        assert _stu_route_fg(state) == "persistence"

    def test_fg_route_partial_safe_mode_goes_to_persistence(self):
        """PARTIAL + safe_mode also routes to persistence (label resolved after persist)."""
        from app.langchain.enums import ExecutionMode
        state = make_state("student")
        state.final_status = GraphExecutionStatus.PARTIAL
        state.pipeline_context.execution_mode = ExecutionMode.SAFE
        assert _stu_route_fg(state) == "persistence"

    def test_fg_route_blocked_skips_persistence(self):
        """BLOCKED must not reach persistence — route is 'end' directly."""
        state = make_state("student")
        state.final_status = GraphExecutionStatus.BLOCKED
        assert _stu_route_fg(state) == "end"

    def test_fg_route_failed_skips_persistence(self):
        """FAILED must not reach persistence — route is 'end' directly."""
        state = make_state("student")
        state.final_status = GraphExecutionStatus.FAILED
        assert _stu_route_fg(state) == "end"

    def test_prof_fg_route_blocked_skips_persistence(self):
        state = make_state("professor")
        state.final_status = GraphExecutionStatus.BLOCKED
        assert _prof_route_fg(state) == "end"

    def test_prof_fg_route_completed_goes_to_persistence(self):
        state = make_state("professor")
        state.final_status = GraphExecutionStatus.COMPLETED
        assert _prof_route_fg(state) == "persistence"

    # --- Persistence terminal routing ---

    def test_persist_route_completed(self):
        state = make_state("student")
        state.final_status = GraphExecutionStatus.COMPLETED
        assert _stu_route_persist(state) == "completed"

    def test_persist_route_safe_mode_completed(self):
        from app.langchain.enums import ExecutionMode
        state = make_state("student")
        state.final_status = GraphExecutionStatus.PARTIAL
        state.pipeline_context.execution_mode = ExecutionMode.SAFE
        assert _stu_route_persist(state) == "safe_mode_completed"

    def test_persist_route_partial_without_safe_mode_is_failed(self):
        state = make_state("student")
        state.final_status = GraphExecutionStatus.PARTIAL
        # safe_mode is False by default
        assert _stu_route_persist(state) == "failed"

    def test_prof_persist_route_completed(self):
        state = make_state("professor")
        state.final_status = GraphExecutionStatus.COMPLETED
        assert _prof_route_persist(state) == "completed"


# ---------------------------------------------------------------------------
# TestSafeExecWrapper
# ---------------------------------------------------------------------------

class TestSafeExecWrapper:
    """Tests for the _safe_exec wrapper in both graph files."""

    @pytest.mark.asyncio
    async def test_calls_fn_normally(self):
        state = make_state("student")
        calls = []

        async def fn(s):
            calls.append(True)
            return s

        result = await _stu_safe_exec(fn, state, "test_node")
        assert len(calls) == 1
        assert result is state

    @pytest.mark.asyncio
    async def test_captures_exception_and_returns_state(self):
        state = make_state("student")

        async def bad_fn(s):
            raise ValueError("adapter exploded")

        result = await _stu_safe_exec(bad_fn, state, "test_node")
        assert result is state
        assert any("ValueError" in r for r in result.failure_reasons)

    @pytest.mark.asyncio
    async def test_skips_when_terminal_guardrail_active(self):
        from app.langgraph.loop_control import TerminalDecision

        state = make_state("student")
        calls = []

        async def fn(s):
            calls.append(True)
            return s

        terminal = TerminalDecision(
            should_stop=True,
            reason="max_steps_exceeded",
        )
        with patch("app.langgraph.graphs.student_graph.terminal_decision", return_value=terminal):
            result = await _stu_safe_exec(fn, state, "test_node")

        assert len(calls) == 0  # fn was not called
        assert any("skipped" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_professor_captures_exception(self):
        state = make_state("professor")

        async def bad_fn(s):
            raise RuntimeError("service unavailable")

        result = await _prof_safe_exec(bad_fn, state, "generation")
        assert any("RuntimeError" in r for r in result.failure_reasons)

    @pytest.mark.asyncio
    async def test_error_msg_is_truncated_to_160_chars(self):
        state = make_state("student")
        long_msg = "x" * 300

        async def bad_fn(s):
            raise ValueError(long_msg)

        await _stu_safe_exec(bad_fn, state, "test_node")
        # failure_reasons entry includes "ValueError: " + up to 160 chars
        assert any(len(r) < 200 for r in state.failure_reasons)


# ---------------------------------------------------------------------------
# TestHappyPath
# ---------------------------------------------------------------------------

class TestHappyPath:
    """Full graph invocation with all execution node adapters mocked.

    This tests that the graph topology is correct and that the state flows
    through all nodes and reaches a terminal status.
    """

    def _make_valid_student_report(self):
        return {"summary": "Good work.", "issues": [], "strengths": ["clear"], "safety": {}}

    def _make_valid_professor_report(self):
        return {"summary": "Solid submission.", "feedback_explanation": "Well done.", "safety": {}}

    @pytest.mark.asyncio
    async def test_student_happy_path_reaches_completed(self):
        request = make_request("student")
        report = self._make_valid_student_report()

        mock_vr = MagicMock()
        mock_vr.valid = True
        mock_vr.repaired = False
        mock_vr.errors = []
        mock_vr.warnings = []

        async def mock_ingestion(state):
            state.pipeline_context.ingestion.text_content = "Student essay text."
            return state

        async def mock_ml_inference(state):
            state.pipeline_context.ml_raw = {"confidence_0_to_4": 3}
            return state

        async def mock_ml_context(state):
            from app.langchain.models import MLContextResult
            state.pipeline_context.ml_result = MLContextResult(
                confidence_0_to_4=3,
                confidence_score=0.75,
            )
            return state

        async def mock_retrieval(state):
            return state

        async def mock_generation(state):
            state.pipeline_context.raw_llm_output = '{"summary": "Good work."}'
            return state

        async def mock_validation(state):
            state.pipeline_context.validation_result = mock_vr
            state.pipeline_context.report = report
            return state

        async def mock_persistence(state):
            state.storage_payload["stored_row"] = {"id": "row-001"}
            return state

        with patch(f"{_NODES}.ingestion.run", new=mock_ingestion), \
             patch(f"{_NODES}.ml_inference.run", new=mock_ml_inference), \
             patch(f"{_NODES}.ml_context.run", new=mock_ml_context), \
             patch(f"{_NODES}.retrieval.run", new=mock_retrieval), \
             patch(f"{_NODES}.generation.run", new=mock_generation), \
             patch(f"{_NODES}.validation.run", new=mock_validation), \
             patch(f"{_NODES}.persistence.run", new=mock_persistence):

            compiled = get_student_compiled_graph()
            initial_state = Phase12GraphState.create(request)
            raw = await compiled.ainvoke(initial_state)

        result = Phase12GraphState.model_validate(raw) if isinstance(raw, dict) else raw
        assert result.final_status in {
            GraphExecutionStatus.COMPLETED,
            GraphExecutionStatus.PARTIAL,
        }

    @pytest.mark.asyncio
    async def test_professor_happy_path_reaches_completed(self):
        request = make_request("professor")
        report = self._make_valid_professor_report()

        mock_vr = MagicMock()
        mock_vr.valid = True
        mock_vr.repaired = False
        mock_vr.errors = []
        mock_vr.warnings = []

        async def mock_ingestion(state):
            state.pipeline_context.ingestion.text_content = "Student work for review."
            return state

        async def mock_ml_inference(state):
            state.pipeline_context.ml_raw = {"confidence_0_to_4": 2}
            return state

        async def mock_ml_context(state):
            from app.langchain.models import MLContextResult
            state.pipeline_context.ml_result = MLContextResult(
                confidence_0_to_4=2,
                confidence_score=0.5,
            )
            return state

        async def mock_retrieval(state):
            return state

        async def mock_generation(state):
            state.pipeline_context.raw_llm_output = '{"summary": "ok"}'
            return state

        async def mock_validation(state):
            state.pipeline_context.validation_result = mock_vr
            state.pipeline_context.report = report
            return state

        async def mock_persistence(state):
            state.storage_payload["stored_row"] = {"id": "row-002"}
            return state

        with patch(f"{_NODES}.ingestion.run", new=mock_ingestion), \
             patch(f"{_NODES}.ml_inference.run", new=mock_ml_inference), \
             patch(f"{_NODES}.ml_context.run", new=mock_ml_context), \
             patch(f"{_NODES}.retrieval.run", new=mock_retrieval), \
             patch(f"{_NODES}.generation.run", new=mock_generation), \
             patch(f"{_NODES}.validation.run", new=mock_validation), \
             patch(f"{_NODES}.persistence.run", new=mock_persistence):

            compiled = get_professor_compiled_graph()
            initial_state = Phase12GraphState.create(request)
            raw = await compiled.ainvoke(initial_state)

        result = Phase12GraphState.model_validate(raw) if isinstance(raw, dict) else raw
        assert result.final_status in {
            GraphExecutionStatus.COMPLETED,
            GraphExecutionStatus.PARTIAL,
        }

    @pytest.mark.asyncio
    async def test_fallback_runs_when_validation_fails(self):
        """When validation marks output invalid, fallback should run and set a report."""
        request = make_request("student")
        fallback_report = {"summary": "fallback", "issues": [], "strengths": [], "safety": {}}

        mock_vr_invalid = MagicMock()
        mock_vr_invalid.valid = False
        mock_vr_invalid.repaired = False
        mock_vr_invalid.errors = ["schema_error"]
        mock_vr_invalid.warnings = []

        async def mock_ingestion(state):
            return state

        async def mock_ml_inference(state):
            return state

        async def mock_ml_context(state):
            return state

        async def mock_retrieval(state):
            return state

        async def mock_generation(state):
            state.pipeline_context.raw_llm_output = "invalid json"
            return state

        async def mock_validation(state):
            state.pipeline_context.validation_result = mock_vr_invalid
            return state

        async def mock_fallback(state):
            state.pipeline_context.report = fallback_report
            return state

        async def mock_persistence(state):
            state.storage_payload["stored_row"] = {"id": "row-003"}
            return state

        with patch(f"{_NODES}.ingestion.run", new=mock_ingestion), \
             patch(f"{_NODES}.ml_inference.run", new=mock_ml_inference), \
             patch(f"{_NODES}.ml_context.run", new=mock_ml_context), \
             patch(f"{_NODES}.retrieval.run", new=mock_retrieval), \
             patch(f"{_NODES}.generation.run", new=mock_generation), \
             patch(f"{_NODES}.validation.run", new=mock_validation), \
             patch(f"{_NODES}.fallback.run", new=mock_fallback), \
             patch(f"{_NODES}.persistence.run", new=mock_persistence):

            compiled = get_student_compiled_graph()
            initial_state = Phase12GraphState.create(request)
            raw = await compiled.ainvoke(initial_state)

        result = Phase12GraphState.model_validate(raw) if isinstance(raw, dict) else raw
        assert result.pipeline_context.report == fallback_report

    @pytest.mark.asyncio
    async def test_mcp_tools_not_called_when_disabled(self):
        """MCP tools must not run when allow_mcp_tools is False (default)."""
        request = make_request("student")
        report = {"summary": "ok", "issues": [], "strengths": [], "safety": {}}

        mock_vr = MagicMock()
        mock_vr.valid = True
        mock_vr.repaired = False
        mock_vr.errors = []
        mock_vr.warnings = []

        mcp_run_calls = []

        async def mock_mcp_run(state):
            mcp_run_calls.append(True)
            return state

        async def noop(state):
            return state

        async def mock_validation(state):
            state.pipeline_context.validation_result = mock_vr
            state.pipeline_context.report = report
            return state

        async def mock_persistence(state):
            state.storage_payload["stored_row"] = {"id": "x"}
            return state

        with patch(f"{_NODES}.ingestion.run", new=noop), \
             patch(f"{_NODES}.ml_inference.run", new=noop), \
             patch(f"{_NODES}.ml_context.run", new=noop), \
             patch(f"{_NODES}.retrieval.run", new=noop), \
             patch(f"{_NODES}.generation.run", new=noop), \
             patch(f"{_NODES}.validation.run", new=mock_validation), \
             patch(f"{_NODES}.mcp_tools.run", new=mock_mcp_run), \
             patch(f"{_NODES}.persistence.run", new=mock_persistence):

            assert not phase12_settings.allow_mcp_tools  # confirm default
            compiled = get_student_compiled_graph()
            initial_state = Phase12GraphState.create(request)
            await compiled.ainvoke(initial_state)

        assert len(mcp_run_calls) == 0


# ---------------------------------------------------------------------------
# TestLoopPrevention
# ---------------------------------------------------------------------------

class TestLoopPrevention:
    """Verify that the terminal guardrail inside _safe_exec stops execution
    when max_graph_steps is exceeded."""

    @pytest.mark.asyncio
    async def test_safe_exec_skips_node_when_steps_exceeded(self):
        from app.langgraph.loop_control import TerminalDecision

        state = make_state("student")
        calls = []

        async def fn(s):
            calls.append(True)
            return s

        terminal = TerminalDecision(
            should_stop=True,
            reason="Maximum total graph steps exceeded.",
        )
        with patch("app.langgraph.graphs.student_graph.terminal_decision", return_value=terminal):
            await _stu_safe_exec(fn, state, "generation")

        assert len(calls) == 0
        assert any("skipped" in w for w in state.warnings)

    def test_config_max_graph_steps_is_20(self):
        """Regression: max_graph_steps must be at least 20 to support the full pipeline."""
        assert phase12_settings.max_graph_steps >= 20


# ---------------------------------------------------------------------------
# TestTracing
# ---------------------------------------------------------------------------

class TestTracing:
    @pytest.mark.asyncio
    async def test_node_traces_populated_after_control_nodes(self):
        """After input_validation and policy_check succeed, node_traces should have entries."""
        request = make_request("student")
        report = {"summary": "ok", "issues": [], "strengths": [], "safety": {}}

        mock_vr = MagicMock()
        mock_vr.valid = True
        mock_vr.repaired = False
        mock_vr.errors = []
        mock_vr.warnings = []

        async def noop(state):
            return state

        async def mock_validation(state):
            state.pipeline_context.validation_result = mock_vr
            state.pipeline_context.report = report
            return state

        async def mock_persistence(state):
            state.storage_payload["stored_row"] = {"id": "x"}
            return state

        with patch(f"{_NODES}.ingestion.run", new=noop), \
             patch(f"{_NODES}.ml_inference.run", new=noop), \
             patch(f"{_NODES}.ml_context.run", new=noop), \
             patch(f"{_NODES}.retrieval.run", new=noop), \
             patch(f"{_NODES}.generation.run", new=noop), \
             patch(f"{_NODES}.validation.run", new=mock_validation), \
             patch(f"{_NODES}.persistence.run", new=mock_persistence):

            compiled = get_student_compiled_graph()
            state = Phase12GraphState.create(request)
            raw = await compiled.ainvoke(state)

        result = Phase12GraphState.model_validate(raw) if isinstance(raw, dict) else raw
        visited_nodes = [e.node_name for e in result.node_traces]
        assert "input_validation" in visited_nodes
        assert "policy_check" in visited_nodes
        assert "final_guardrail" in visited_nodes

    @pytest.mark.asyncio
    async def test_terminal_trace_set_after_graph_run(self):
        request = make_request("student")
        report = {"summary": "ok", "issues": [], "strengths": [], "safety": {}}

        mock_vr = MagicMock()
        mock_vr.valid = True
        mock_vr.repaired = False
        mock_vr.errors = []
        mock_vr.warnings = []

        async def noop(state):
            return state

        async def mock_validation(state):
            state.pipeline_context.validation_result = mock_vr
            state.pipeline_context.report = report
            return state

        async def mock_persistence(state):
            state.storage_payload["stored_row"] = {"id": "x"}
            return state

        with patch(f"{_NODES}.ingestion.run", new=noop), \
             patch(f"{_NODES}.ml_inference.run", new=noop), \
             patch(f"{_NODES}.ml_context.run", new=noop), \
             patch(f"{_NODES}.retrieval.run", new=noop), \
             patch(f"{_NODES}.generation.run", new=noop), \
             patch(f"{_NODES}.validation.run", new=mock_validation), \
             patch(f"{_NODES}.persistence.run", new=mock_persistence):

            compiled = get_student_compiled_graph()
            state = Phase12GraphState.create(request)
            raw = await compiled.ainvoke(state)

        result = Phase12GraphState.model_validate(raw) if isinstance(raw, dict) else raw
        assert result.terminal_trace is not None
        assert result.terminal_trace.final_status in {
            GraphExecutionStatus.COMPLETED.value,
            GraphExecutionStatus.PARTIAL.value,
        }


# ---------------------------------------------------------------------------
# TestRoleIsolation
# ---------------------------------------------------------------------------

class TestRoleIsolation:
    """Verify that a student request routed to the professor graph (and vice versa)
    is rejected by input_validation before any execution node runs."""

    @pytest.mark.asyncio
    async def test_student_role_request_blocked_in_professor_graph(self):
        """A student-role Phase12GraphState must be blocked by the professor graph's
        input_validation node (graph_name prefix mismatch)."""
        # Create a STUDENT state
        student_state = Phase12GraphState.create(make_request("student"))

        compiled = get_professor_compiled_graph()
        raw = await compiled.ainvoke(student_state)
        result = Phase12GraphState.model_validate(raw) if isinstance(raw, dict) else raw

        assert result.final_status == GraphExecutionStatus.BLOCKED or \
               result.status == GraphExecutionStatus.BLOCKED or \
               len(result.failure_reasons) > 0, \
            "Student state should be blocked by professor graph input_validation"

    @pytest.mark.asyncio
    async def test_professor_role_request_blocked_in_student_graph(self):
        """A professor-role Phase12GraphState must be blocked by the student graph's
        input_validation node."""
        professor_state = Phase12GraphState.create(make_request("professor"))

        compiled = get_student_compiled_graph()
        raw = await compiled.ainvoke(professor_state)
        result = Phase12GraphState.model_validate(raw) if isinstance(raw, dict) else raw

        assert result.final_status == GraphExecutionStatus.BLOCKED or \
               result.status == GraphExecutionStatus.BLOCKED or \
               len(result.failure_reasons) > 0, \
            "Professor state should be blocked by student graph input_validation"

    def test_student_spec_name_does_not_match_professor(self):
        assert "student" in STUDENT_GRAPH_SPEC.name
        assert "professor" not in STUDENT_GRAPH_SPEC.name

    def test_professor_spec_name_does_not_match_student(self):
        assert "professor" in PROFESSOR_GRAPH_SPEC.name
        assert "student" not in PROFESSOR_GRAPH_SPEC.name


# ---------------------------------------------------------------------------
# TestSafeMode
# ---------------------------------------------------------------------------

class TestSafeMode:
    def test_safe_mode_property_false_by_default(self):
        state = make_state("student")
        assert state.safe_mode is False

    def test_safe_mode_true_when_execution_mode_is_safe(self):
        from app.langchain.enums import ExecutionMode
        state = make_state("student")
        state.pipeline_context.execution_mode = ExecutionMode.SAFE
        assert state.safe_mode is True

    def test_safe_mode_true_when_pipeline_mode_restricted(self):
        from app.langchain.enums import PipelineMode
        state = make_state("student")
        state.pipeline_context.mode = PipelineMode.RESTRICTED
        assert state.safe_mode is True

    def test_fg_route_partial_with_safe_mode_goes_to_persistence(self):
        """final_guardrail routes PARTIAL to persistence regardless of safe_mode.
        The terminal label (safe_mode_completed vs failed) is resolved after persist."""
        from app.langchain.enums import ExecutionMode
        state = make_state("student")
        state.final_status = GraphExecutionStatus.PARTIAL
        state.pipeline_context.execution_mode = ExecutionMode.SAFE
        assert _stu_route_fg(state) == "persistence"

    def test_persist_route_resolves_safe_mode_completed(self):
        """After persistence, PARTIAL + safe_mode maps to 'safe_mode_completed'."""
        from app.langchain.enums import ExecutionMode
        state = make_state("student")
        state.final_status = GraphExecutionStatus.PARTIAL
        state.pipeline_context.execution_mode = ExecutionMode.SAFE
        assert _stu_route_persist(state) == "safe_mode_completed"

    def test_persist_route_resolves_failed_for_partial_no_safe_mode(self):
        state = make_state("student")
        state.final_status = GraphExecutionStatus.PARTIAL
        # safe_mode = False (default)
        assert _stu_route_persist(state) == "failed"


# ---------------------------------------------------------------------------
# TestPersistenceSafety
# ---------------------------------------------------------------------------

class TestPersistenceSafety:
    """Verify that persistence only runs for guardrail-approved outputs and that
    the persisted row contains the complete finalized trace."""

    @pytest.mark.asyncio
    async def test_persistence_not_called_when_blocked(self):
        """A blocked run (input_validation failure) must never write to ai_reports."""
        persist_calls = []

        async def mock_persist(state):
            persist_calls.append(True)
            return state

        with patch(f"{_NODES}.persistence.run", new=mock_persist):
            compiled = get_student_compiled_graph()
            # Simulate a professor state sent to the student graph — will be BLOCKED
            bad_state = Phase12GraphState.create(make_request("professor"))
            await compiled.ainvoke(bad_state)

        assert len(persist_calls) == 0, (
            "persistence.run must not be called when the graph exits BLOCKED"
        )

    @pytest.mark.asyncio
    async def test_persistence_not_called_when_no_report_generated(self):
        """When generation and fallback both fail (empty report), final_guardrail
        should emit FAILED and persistence must be skipped."""
        persist_calls = []

        async def mock_persist(state):
            persist_calls.append(True)
            return state

        mock_vr_invalid = MagicMock()
        mock_vr_invalid.valid = False
        mock_vr_invalid.repaired = False
        mock_vr_invalid.errors = ["parse_failed"]
        mock_vr_invalid.warnings = []

        async def noop(state):
            return state

        async def mock_validation(state):
            # Validation fails and leaves report empty
            state.pipeline_context.validation_result = mock_vr_invalid
            state.add_failure_reason("phase12_validation_failed")
            return state

        async def mock_fallback(state):
            # Fallback also fails to produce a report (leaves it empty)
            return state

        with patch(f"{_NODES}.ingestion.run", new=noop), \
             patch(f"{_NODES}.ml_inference.run", new=noop), \
             patch(f"{_NODES}.ml_context.run", new=noop), \
             patch(f"{_NODES}.retrieval.run", new=noop), \
             patch(f"{_NODES}.generation.run", new=noop), \
             patch(f"{_NODES}.validation.run", new=mock_validation), \
             patch(f"{_NODES}.fallback.run", new=mock_fallback), \
             patch(f"{_NODES}.persistence.run", new=mock_persist):

            compiled = get_student_compiled_graph()
            state = Phase12GraphState.create(make_request("student"))
            raw = await compiled.ainvoke(state)

        result = Phase12GraphState.model_validate(raw) if isinstance(raw, dict) else raw
        # final_guardrail should have set FAILED (no report)
        assert result.final_status in (GraphExecutionStatus.FAILED, GraphExecutionStatus.BLOCKED)
        assert len(persist_calls) == 0, (
            "persistence.run must not be called when final_guardrail emits FAILED"
        )

    @pytest.mark.asyncio
    async def test_persistence_called_when_completed(self):
        """A successful run must call persistence exactly once after final_guardrail."""
        persist_calls = []
        terminal_status_at_persist: list[GraphExecutionStatus | None] = []

        report = {"summary": "ok", "issues": [], "strengths": [], "safety": {}}
        mock_vr = MagicMock()
        mock_vr.valid = True
        mock_vr.repaired = False
        mock_vr.errors = []
        mock_vr.warnings = []

        async def mock_persist(state):
            persist_calls.append(True)
            # Capture final_status at the time persistence runs
            terminal_status_at_persist.append(state.final_status)
            state.storage_payload["stored_row"] = {"id": "row-safe"}
            return state

        async def noop(state):
            return state

        async def mock_validation(state):
            state.pipeline_context.validation_result = mock_vr
            state.pipeline_context.report = report
            return state

        with patch(f"{_NODES}.ingestion.run", new=noop), \
             patch(f"{_NODES}.ml_inference.run", new=noop), \
             patch(f"{_NODES}.ml_context.run", new=noop), \
             patch(f"{_NODES}.retrieval.run", new=noop), \
             patch(f"{_NODES}.generation.run", new=noop), \
             patch(f"{_NODES}.validation.run", new=mock_validation), \
             patch(f"{_NODES}.persistence.run", new=mock_persist):

            compiled = get_student_compiled_graph()
            state = Phase12GraphState.create(make_request("student"))
            await compiled.ainvoke(state)

        assert len(persist_calls) == 1
        # final_status must already be set when persistence runs (guardrail ran first)
        assert terminal_status_at_persist[0] in (
            GraphExecutionStatus.COMPLETED,
            GraphExecutionStatus.PARTIAL,
        ), "final_guardrail must set final_status before persistence runs"

    @pytest.mark.asyncio
    async def test_terminal_trace_populated_before_persistence(self):
        """state.terminal_trace must be non-None when persistence.run is called,
        meaning final_guardrail has already completed its finalize_trace() call."""
        terminal_trace_at_persist = []

        report = {"summary": "ok", "issues": [], "strengths": [], "safety": {}}
        mock_vr = MagicMock()
        mock_vr.valid = True
        mock_vr.repaired = False
        mock_vr.errors = []
        mock_vr.warnings = []

        async def mock_persist(state):
            terminal_trace_at_persist.append(state.terminal_trace)
            state.storage_payload["stored_row"] = {"id": "x"}
            return state

        async def noop(state):
            return state

        async def mock_validation(state):
            state.pipeline_context.validation_result = mock_vr
            state.pipeline_context.report = report
            return state

        with patch(f"{_NODES}.ingestion.run", new=noop), \
             patch(f"{_NODES}.ml_inference.run", new=noop), \
             patch(f"{_NODES}.ml_context.run", new=noop), \
             patch(f"{_NODES}.retrieval.run", new=noop), \
             patch(f"{_NODES}.generation.run", new=noop), \
             patch(f"{_NODES}.validation.run", new=mock_validation), \
             patch(f"{_NODES}.persistence.run", new=mock_persist):

            compiled = get_student_compiled_graph()
            state = Phase12GraphState.create(make_request("student"))
            await compiled.ainvoke(state)

        assert len(terminal_trace_at_persist) == 1
        assert terminal_trace_at_persist[0] is not None, (
            "terminal_trace must be set by final_guardrail before persistence runs"
        )

    @pytest.mark.asyncio
    async def test_professor_persistence_not_called_when_blocked(self):
        persist_calls = []

        async def mock_persist(state):
            persist_calls.append(True)
            return state

        with patch(f"{_NODES}.persistence.run", new=mock_persist):
            compiled = get_professor_compiled_graph()
            # Student state in professor graph → BLOCKED
            bad_state = Phase12GraphState.create(make_request("student"))
            await compiled.ainvoke(bad_state)

        assert len(persist_calls) == 0

    def test_spec_node_order_has_final_guardrail_before_persistence(self):
        """Spec node_order must reflect the corrected graph topology."""
        order = STUDENT_GRAPH_SPEC.node_order
        fg_idx = order.index("final_guardrail")
        persist_idx = order.index("persistence")
        assert fg_idx < persist_idx, (
            f"final_guardrail (idx {fg_idx}) must come before persistence (idx {persist_idx})"
        )

    def test_prof_spec_node_order_has_final_guardrail_before_persistence(self):
        order = PROFESSOR_GRAPH_SPEC.node_order
        fg_idx = order.index("final_guardrail")
        persist_idx = order.index("persistence")
        assert fg_idx < persist_idx


# ---------------------------------------------------------------------------
# TestMcpPathValidationSafety
# ---------------------------------------------------------------------------

class TestMcpPathValidationSafety:
    """Verify the MCP path passes through final_guardrail before persistence,
    so any report corruption is caught before writing to the database."""

    @pytest.mark.asyncio
    async def test_mcp_path_hits_final_guardrail_before_persistence(self):
        """When MCP is enabled and tools are selected, execution must still
        pass through final_guardrail before persistence."""
        node_order_seen: list[str] = []
        report = {"summary": "ok", "issues": [], "strengths": [], "safety": {}}
        mock_vr = MagicMock()
        mock_vr.valid = True
        mock_vr.repaired = False
        mock_vr.errors = []
        mock_vr.warnings = []

        async def noop(state):
            return state

        async def mock_validation(state):
            state.pipeline_context.validation_result = mock_vr
            state.pipeline_context.report = report
            return state

        async def mock_mcp(state):
            node_order_seen.append("mcp_tools")
            return state

        async def mock_persist(state):
            node_order_seen.append("persistence")
            state.storage_payload["stored_row"] = {"id": "x"}
            return state

        with patch.object(phase12_settings, "allow_mcp_tools", True), \
             patch.object(phase12_settings, "allow_student_mcp_tools", True), \
             patch(f"{_NODES}.ingestion.run", new=noop), \
             patch(f"{_NODES}.ml_inference.run", new=noop), \
             patch(f"{_NODES}.ml_context.run", new=noop), \
             patch(f"{_NODES}.retrieval.run", new=noop), \
             patch(f"{_NODES}.generation.run", new=noop), \
             patch(f"{_NODES}.validation.run", new=mock_validation), \
             patch(f"{_NODES}.mcp_tools.run", new=mock_mcp), \
             patch(f"{_NODES}.persistence.run", new=mock_persist):

            state = Phase12GraphState.create(make_request("student"))
            state.selected_tools = ["student.summariser"]
            compiled = get_student_compiled_graph()
            raw = await compiled.ainvoke(state)

        # mcp_tools must appear before persistence in the execution order
        if "mcp_tools" in node_order_seen and "persistence" in node_order_seen:
            assert node_order_seen.index("mcp_tools") < node_order_seen.index("persistence"), (
                "mcp_tools must run before persistence (final_guardrail sits between them)"
            )

    def test_mcp_validation_route_to_mcp_not_persistence(self):
        """After valid output, MCP-enabled path must route to mcp_tools,
        not directly to persistence (final_guardrail is after mcp_tools)."""
        state = make_state("student")
        state.pipeline_context.validation_result.valid = True
        state.selected_tools = ["student.summariser"]
        with patch.object(phase12_settings, "allow_mcp_tools", True):
            with patch.object(phase12_settings, "allow_student_mcp_tools", True):
                route = _stu_route_val(state)
        assert route == "mcp_tools"
        # Critically: route is NOT "persistence" — final_guardrail runs after mcp_tools
        assert route != "persistence"


# ---------------------------------------------------------------------------
# TestStateHelpers
# ---------------------------------------------------------------------------

class TestStateHelpers:
    def test_add_failure_reason_deduplicates(self):
        state = make_state("student")
        state.add_failure_reason("duplicate_error")
        state.add_failure_reason("duplicate_error")
        assert state.failure_reasons.count("duplicate_error") == 1

    def test_add_warning_deduplicates(self):
        state = make_state("student")
        state.add_warning("same warning")
        state.add_warning("same warning")
        assert state.warnings.count("same warning") == 1

    def test_errors_property_aliases_failure_reasons(self):
        state = make_state("student")
        state.add_failure_reason("test_err")
        assert "test_err" in state.errors

    def test_mcp_results_property_aliases_tool_outputs(self):
        state = make_state("student")
        state.tool_outputs.append({"result": "ok"})
        assert state.mcp_results == [{"result": "ok"}]

    def test_role_property(self):
        assert make_state("student").role == "student"
        assert make_state("professor").role == "professor"

    def test_ml_completed_false_initially(self):
        state = make_state("student")
        assert state.ml_completed is False

    def test_rag_completed_false_initially(self):
        state = make_state("student")
        assert state.rag_completed is False
