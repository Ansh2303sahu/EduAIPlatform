"""Phase 15/16 professor generative graph."""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from ..config import phase12_settings
from ..nodes import (
    evidence_sufficiency_node,
    final_guardrail_node,
    input_validation_node,
    ml_decision_node,
    policy_check_node,
    rag_decision_node,
    safe_mode_decision_node,
)
from ..nodes.analysis import analysis_node
from ..nodes.critic import critic_node
from ..nodes.generator import generator_node
from ..nodes.planner import planner_node
from ..nodes.refiner import refiner_node
from ..nodes.validation import validation_node
from ..state import GraphExecutionStatus, Phase12GraphState


def _blocked_or_failed(state: Phase12GraphState) -> bool:
    final_status = state.final_status or state.status
    return final_status in {GraphExecutionStatus.BLOCKED, GraphExecutionStatus.FAILED}


def _route_after_gate(state: Phase12GraphState) -> str:
    return "end" if _blocked_or_failed(state) else "next"


async def _input_validation_node(state: Phase12GraphState) -> Phase12GraphState:
    state.graph_name = phase12_settings.professor_graph_name
    return await input_validation_node(state)


def build_professor_generative_graph():
    graph = StateGraph(Phase12GraphState)
    graph.add_node("input_validation", _input_validation_node)
    graph.add_node("policy_check", policy_check_node)
    graph.add_node("evidence_sufficiency", evidence_sufficiency_node)
    graph.add_node("ml_decision", ml_decision_node)
    graph.add_node("rag_decision", rag_decision_node)
    graph.add_node("safe_mode_decision", safe_mode_decision_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("planner", planner_node)
    graph.add_node("generator", generator_node)
    graph.add_node("critic", critic_node)
    graph.add_node("refiner", refiner_node)
    graph.add_node("validation", validation_node)
    graph.add_node("final_guardrail", final_guardrail_node)

    graph.add_edge(START, "input_validation")
    graph.add_conditional_edges("input_validation", _route_after_gate, {"next": "policy_check", "end": END})
    graph.add_conditional_edges("policy_check", _route_after_gate, {"next": "evidence_sufficiency", "end": END})
    graph.add_edge("evidence_sufficiency", "ml_decision")
    graph.add_edge("ml_decision", "rag_decision")
    graph.add_conditional_edges("rag_decision", _route_after_gate, {"next": "safe_mode_decision", "end": END})
    graph.add_edge("safe_mode_decision", "analysis")
    graph.add_edge("analysis", "planner")
    graph.add_edge("planner", "generator")
    graph.add_edge("generator", "critic")
    graph.add_edge("critic", "refiner")
    graph.add_edge("refiner", "validation")
    graph.add_edge("validation", "final_guardrail")
    graph.add_edge("final_guardrail", END)
    return graph


@lru_cache(maxsize=1)
def get_professor_generative_compiled_graph():
    return build_professor_generative_graph().compile()
