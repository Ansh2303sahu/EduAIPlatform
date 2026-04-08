"""Internal schemas for Phase 12 LangGraph planning and graph description.

These models are private to the backend. They describe graph requests and graph
shape while reusing existing Phase 10 storage, ML, and generation contracts.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


GraphRole = Literal["student", "professor"]


class Phase12ExecutionRequest(BaseModel):
    """Internal execution request for selecting and preparing a graph run."""

    file_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    role: GraphRole
    correlation_id: str = Field(min_length=1)
    force: bool = False
    submission_id: str = ""
    user_email: str | None = None
    raw_claims: dict[str, Any] = Field(default_factory=dict)
    file_metadata: dict[str, Any] = Field(default_factory=dict)


class Phase12NodeDescriptor(BaseModel):
    """Describe a single graph node and the Phase 10/11 modules it wraps."""

    name: str
    description: str
    wrapped_modules: list[str] = Field(default_factory=list)
    reads: list[str] = Field(default_factory=list)
    writes: list[str] = Field(default_factory=list)
    optional: bool = False


class Phase12GraphSpec(BaseModel):
    """Describe a deterministic internal graph for a specific role."""

    name: str
    role: GraphRole
    description: str
    entrypoint: str
    node_order: list[str]
    terminal_nodes: list[str]
    optional_nodes: list[str] = Field(default_factory=list)
    wrapped_modules: list[str] = Field(default_factory=list)
