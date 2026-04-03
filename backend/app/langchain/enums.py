"""
Shared enumerations for Phase 10 LangChain pipelines.

All enums are ``str`` subclasses so they serialise transparently to/from JSON
and can be compared directly against plain strings.
"""

from enum import Enum


# ---------------------------------------------------------------------------
# Public Phase 10 enums (new in this phase)
# ---------------------------------------------------------------------------

class ChainRole(str, Enum):
    """Which audience the chain generates output for."""

    STUDENT = "student"
    PROFESSOR = "professor"


class ExecutionMode(str, Enum):
    """
    How the chain was executed.

    NORMAL   — standard path, full generation.
    SAFE     — restricted/safe-mode path (injection detected, low ML confidence).
    FALLBACK — primary model failed; fell back to secondary model or hardcoded response.
    """

    NORMAL = "normal"
    SAFE = "safe"
    FALLBACK = "fallback"


class DecisionSource(str, Enum):
    """
    Which system(s) drove the final output.

    ML       — output shaped entirely by ML classifier signals.
    LLM      — output shaped entirely by the LLM generation.
    HYBRID   — output shaped by both ML signals and LLM generation.
    FALLBACK — output is a hardcoded safe-mode fallback (no live inference).
    """

    ML = "ml"
    LLM = "llm"
    HYBRID = "hybrid"
    FALLBACK = "fallback"


# ---------------------------------------------------------------------------
# Legacy / pipeline-internal enums (kept for backward compat with
# student_pipeline.py, professor_pipeline.py, chain_factory.py, etc.)
# ---------------------------------------------------------------------------

class PipelineRole(str, Enum):
    """Alias of ChainRole — kept so existing services do not require changes."""

    STUDENT = "student"
    PROFESSOR = "professor"


class PipelineMode(str, Enum):
    """
    Safety mode for a pipeline run.

    NORMAL     — full generation.
    RESTRICTED — restricted generation (triggers safe output + needs_review flag).
    """

    NORMAL = "normal"
    RESTRICTED = "restricted"


class AnalysisType(str, Enum):
    """Submission analysis variant — kept in sync with Phase 7 string literals."""

    STUDENT_PROJECT = "student_project_review"
    STUDENT_ACADEMIC = "student_academic_review"
    PROFESSOR_PROJECT = "professor_project_review"
    PROFESSOR_ACADEMIC = "professor_academic_review"


class LLMProvider(str, Enum):
    """Underlying model provider."""

    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class PipelineStage(str, Enum):
    """Named stages used in execution logging."""

    SANITIZE = "sanitize"
    RETRIEVE = "retrieve"
    BUILD_ML = "build_ml"
    BUILD_PROMPT = "build_prompt"
    GENERATE = "generate"
    PARSE = "parse"
    VALIDATE = "validate"
    STORE = "store"


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def chain_role_from_pipeline_role(role: PipelineRole) -> ChainRole:
    """Convert a ``PipelineRole`` to the equivalent ``ChainRole``."""
    return ChainRole(role.value)


def execution_mode_from_pipeline_mode(mode: PipelineMode) -> ExecutionMode:
    """Map ``PipelineMode.RESTRICTED`` → ``ExecutionMode.SAFE``."""
    if mode == PipelineMode.RESTRICTED:
        return ExecutionMode.SAFE
    return ExecutionMode.NORMAL
