"""Internal Phase 12 LangGraph scaffolding for EduAI backend.

The package exports are loaded lazily so modules can import
``app.langgraph.state`` without pulling the full engine/graph stack into the
import chain and creating circular imports.
"""

from __future__ import annotations

from typing import Any

from .config import Phase12LangGraphSettings, phase12_settings

__all__ = [
    "Phase12Engine",
    "Phase12ExecutionLimits",
    "Phase12ExecutionRequest",
    "Phase12FailureCode",
    "Phase12GraphSpec",
    "Phase12GraphState",
    "Phase12LangGraphSettings",
    "Phase12NodeDescriptor",
    "Phase12PolicyBundle",
    "Phase12PolicyDecision",
    "Phase12RolePolicy",
    "Phase12TimeoutBudgets",
    "LoopCheckResult",
    "TerminalDecision",
    "phase12_settings",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - import indirection
    if name == "Phase12Engine":
        from .engine import Phase12Engine

        return Phase12Engine
    if name in {
        "LoopCheckResult",
        "TerminalDecision",
    }:
        from .loop_control import LoopCheckResult, TerminalDecision

        return {"LoopCheckResult": LoopCheckResult, "TerminalDecision": TerminalDecision}[name]
    if name in {
        "Phase12ExecutionLimits",
        "Phase12FailureCode",
        "Phase12PolicyBundle",
        "Phase12PolicyDecision",
        "Phase12RolePolicy",
        "Phase12TimeoutBudgets",
    }:
        from .policy import (
            Phase12ExecutionLimits,
            Phase12FailureCode,
            Phase12PolicyBundle,
            Phase12PolicyDecision,
            Phase12RolePolicy,
            Phase12TimeoutBudgets,
        )

        return {
            "Phase12ExecutionLimits": Phase12ExecutionLimits,
            "Phase12FailureCode": Phase12FailureCode,
            "Phase12PolicyBundle": Phase12PolicyBundle,
            "Phase12PolicyDecision": Phase12PolicyDecision,
            "Phase12RolePolicy": Phase12RolePolicy,
            "Phase12TimeoutBudgets": Phase12TimeoutBudgets,
        }[name]
    if name in {
        "Phase12ExecutionRequest",
        "Phase12GraphSpec",
        "Phase12NodeDescriptor",
    }:
        from .schemas import Phase12ExecutionRequest, Phase12GraphSpec, Phase12NodeDescriptor

        return {
            "Phase12ExecutionRequest": Phase12ExecutionRequest,
            "Phase12GraphSpec": Phase12GraphSpec,
            "Phase12NodeDescriptor": Phase12NodeDescriptor,
        }[name]
    if name == "Phase12GraphState":
        from .state import Phase12GraphState

        return Phase12GraphState
    raise AttributeError(name)
