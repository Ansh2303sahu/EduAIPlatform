"""Configuration for the internal Phase 12 LangGraph layer.

This module is intentionally light. It defines feature flags and execution
limits for the graph scaffolding without wiring any runtime routes yet.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Phase12LangGraphSettings(BaseSettings):
    """Execution limits and feature flags for Phase 12 graph scaffolding."""

    model_config = SettingsConfigDict(
        env_prefix="PHASE12_",
        case_sensitive=False,
        extra="ignore",
    )

    enabled: bool = Field(default=False)
    enable_trace_capture: bool = Field(default=True)
    persist_trace_to_model_versions: bool = Field(default=True)
    trace_schema_version: str = Field(default="1.0")
    trace_summary_max_items: int = Field(default=8, ge=3, le=20)
    graph_timeout_seconds: float = Field(default=180.0, ge=5.0, le=600.0)
    student_graph_name: str = Field(default="phase12_student_graph")
    professor_graph_name: str = Field(default="phase12_professor_graph")
    max_graph_steps: int = Field(default=20, ge=1, le=50)
    max_node_retries: int = Field(default=0, ge=0, le=3)
    max_retrieval_retries: int = Field(default=1, ge=0, le=10)
    max_repair_loops: int = Field(default=1, ge=0, le=10)
    max_tool_calls: int = Field(default=2, ge=0, le=20)
    max_repeated_state_transitions: int = Field(default=2, ge=1, le=20)
    allow_mcp_tools: bool = Field(default=False)
    allow_student_mcp_tools: bool = Field(default=False)
    allow_professor_mcp_tools: bool = Field(default=False)
    student_safe_mode_ml_confidence_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    professor_safe_mode_ml_confidence_threshold: float = Field(default=0.40, ge=0.0, le=1.0)
    safe_mode_evidence_quality_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    ml_timeout_budget_seconds: float = Field(default=20.0, ge=1.0, le=300.0)
    retrieval_timeout_budget_seconds: float = Field(default=15.0, ge=1.0, le=300.0)
    generation_timeout_budget_seconds: float = Field(default=120.0, ge=1.0, le=600.0)
    validation_timeout_budget_seconds: float = Field(default=10.0, ge=1.0, le=120.0)
    tool_timeout_budget_seconds: float = Field(default=20.0, ge=1.0, le=300.0)
    default_engine_label: str = Field(default="phase12_langgraph")


@lru_cache(maxsize=1)
def get_phase12_settings() -> Phase12LangGraphSettings:
    """Return the cached Phase 12 settings instance."""

    return Phase12LangGraphSettings()


phase12_settings = get_phase12_settings()
