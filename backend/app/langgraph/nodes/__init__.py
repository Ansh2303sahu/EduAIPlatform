"""Phase 12 graph nodes.

These nodes are thin orchestration wrappers around existing Phase 10 and
Phase 11 services. They intentionally avoid introducing parallel business logic.

Control nodes are exported as callables for direct use in graph builders.
Execution nodes are available as submodules for graph builders to import via
``from ..nodes import ingestion as _ingestion_mod`` so patches target the
correct attribute at call time rather than binding at import time.
"""

# Control node callables (used directly as LangGraph node functions)
from .analysis import analysis_node
from .critic import critic_node
from .evidence_sufficiency import evidence_sufficiency_node
from .final_guardrail import final_guardrail_node
from .generator import generator_node
from .input_validation import input_validation_node
from .ml_decision import ml_decision_node
from .planner import planner_node
from .policy_check import policy_check_node
from .rag_decision import rag_decision_node
from .refiner import refiner_node
from .safe_mode_decision import safe_mode_decision_node
from .validation import validation_node

# Execution node modules (imported as modules in graph builders)
from . import fallback
from . import generation
from . import ingestion
from . import mcp_tools
from . import ml_context
from . import ml_inference
from . import persistence
from . import retrieval
from . import validation

__all__ = [
    # control node callables
    "analysis_node",
    "critic_node",
    "evidence_sufficiency_node",
    "final_guardrail_node",
    "generator_node",
    "input_validation_node",
    "ml_decision_node",
    "planner_node",
    "policy_check_node",
    "rag_decision_node",
    "refiner_node",
    "safe_mode_decision_node",
    "validation_node",
    # execution node modules
    "fallback",
    "generation",
    "ingestion",
    "mcp_tools",
    "ml_context",
    "ml_inference",
    "persistence",
    "retrieval",
    "validation",
]
