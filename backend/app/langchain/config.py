"""
Phase 10 LangChain settings.

Uses the identical ``BaseSettings`` / ``pydantic-settings`` import pattern as
``app.core.config`` so that env-var loading is consistent across the project.

Integration with the central settings class
-------------------------------------------
``LangChainSettings`` does NOT extend ``app.core.config.Settings``; that would
create a heavyweight dependency and force re-validation of every top-level
setting on every Phase 10 import.  Instead, *shared* env vars (``LLM_PROVIDER``,
``OLLAMA_BASE_URL``, ``ANTHROPIC_API_KEY``, etc.) use the **same alias names**
as the central settings class so that a single ``.env`` file configures both.
``LangChainSettings`` is intentionally additive — it does not override any
existing Phase 7 or RAG settings.

Backward compatibility
----------------------
All attribute names used by existing Phase 10 service files are preserved
(``llm_provider``, ``llm_temperature``, ``llm_max_retries``, etc.) as
read-only properties that delegate to the new canonical fields.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Union, cast

from pydantic import ConfigDict, Field, model_validator

try:
    _ps = import_module("pydantic_settings")
    _BaseSettings = _ps.BaseSettings
except ModuleNotFoundError:  # pragma: no cover
    from pydantic import BaseModel as _BaseSettings  # type: ignore[assignment]

from app.langchain.enums import ChainRole, LLMProvider

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILES = tuple(
    str(path)
    for path in (_BACKEND_ROOT / ".env",)
    if path.exists()
)


class LangChainSettings(_BaseSettings):  # type: ignore[misc]
    """
    All Phase 10 configuration values loaded from environment variables.

    Fields follow the ``Field(default=..., alias="ENV_VAR_NAME")`` pattern
    used in ``app.core.config.Settings``.  ``case_sensitive=False`` means env
    vars can be upper- or lower-case.
    """

    model_config = cast(
        ConfigDict,
        {
            "case_sensitive": False,
            "env_file": _ENV_FILES,
            "env_file_encoding": "utf-8",
            "extra": "ignore",
        },
    )

    # ── Feature flag ────────────────────────────────────────────────────────
    enabled: bool = Field(default=True, alias="PHASE10_ENABLED")

    # ── Provider selection ───────────────────────────────────────────────────
    provider: str = Field(default="ollama", alias="LLM_PROVIDER")

    # ── Resolved model names (filled by validator if left blank) ─────────────
    primary_model: str = Field(default="", alias="PHASE10_PRIMARY_MODEL")
    fallback_model: str = Field(default="", alias="PHASE10_FALLBACK_MODEL")

    # ── Ollama ───────────────────────────────────────────────────────────────
    ollama_base_url: str = Field(
        default="http://host.docker.internal:11434",
        alias="OLLAMA_BASE_URL",
    )
    ollama_primary_model: str = Field(default="gemma3:latest", alias="OLLAMA_PRIMARY_MODEL")
    ollama_fallback_model: str = Field(default="mistral:latest", alias="OLLAMA_FALLBACK_MODEL")

    # ── Anthropic / Claude ───────────────────────────────────────────────────
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    anthropic_primary_model: str = Field(
        default="claude-sonnet-4-6",
        alias="ANTHROPIC_PRIMARY_MODEL",
    )
    anthropic_fallback_model: str = Field(
        default="claude-haiku-4-5-20251001",
        alias="ANTHROPIC_FALLBACK_MODEL",
    )

    # ── Generation limits ────────────────────────────────────────────────────
    timeout_seconds: int = Field(default=120, alias="PHASE10_TIMEOUT_SECONDS")
    max_retries: int = Field(default=2, alias="PHASE10_MAX_RETRIES")
    max_context_chars: int = Field(default=50_000, alias="PHASE10_MAX_CONTEXT_CHARS")
    max_output_chars: int = Field(default=4096, alias="PHASE10_MAX_OUTPUT_CHARS")

    # ── Per-role temperatures ────────────────────────────────────────────────
    student_temperature: float = Field(default=0.1, alias="PHASE10_STUDENT_TEMPERATURE")
    professor_temperature: float = Field(default=0.05, alias="PHASE10_PROFESSOR_TEMPERATURE")

    # ── Feature toggles ──────────────────────────────────────────────────────
    store_raw_output: bool = Field(default=False, alias="PHASE10_STORE_RAW_OUTPUT")
    enable_execution_logs: bool = Field(default=True, alias="PHASE10_ENABLE_EXECUTION_LOGS")
    enable_safe_mode: bool = Field(default=True, alias="PHASE10_ENABLE_SAFE_MODE")
    enable_json_repair: bool = Field(default=True, alias="PHASE10_ENABLE_JSON_REPAIR")

    # ── Versioning ───────────────────────────────────────────────────────────
    chain_version: str = Field(default="1.0", alias="PHASE10_CHAIN_VERSION")
    student_prompt_version: str = Field(default="v1", alias="PHASE10_STUDENT_PROMPT_VERSION")
    professor_prompt_version: str = Field(default="v1", alias="PHASE10_PROFESSOR_PROMPT_VERSION")
    schema_version: str = Field(default="1.0", alias="PHASE10_SCHEMA_VERSION")

    # ── Validator: resolve primary/fallback model from provider defaults ──────

    @model_validator(mode="after")
    def _resolve_models(self) -> "LangChainSettings":
        """
        If ``PHASE10_PRIMARY_MODEL`` / ``PHASE10_FALLBACK_MODEL`` are not set,
        derive them from the provider-specific model env vars.
        """
        if not self.primary_model:
            self.primary_model = (
                self.anthropic_primary_model
                if self.provider == LLMProvider.ANTHROPIC.value
                else self.ollama_primary_model
            )
        if not self.fallback_model:
            self.fallback_model = (
                self.anthropic_fallback_model
                if self.provider == LLMProvider.ANTHROPIC.value
                else self.ollama_fallback_model
            )
        return self

    # ── Helper method ────────────────────────────────────────────────────────

    def temperature_for(self, role: Union[str, ChainRole]) -> float:
        """
        Return the appropriate generation temperature for *role*.

        Accepts a ``ChainRole`` enum value or a plain string
        (``"student"`` / ``"professor"``).
        """
        role_str = role.value if isinstance(role, ChainRole) else str(role).lower()
        if role_str == ChainRole.PROFESSOR.value:
            return self.professor_temperature
        return self.student_temperature

    # ── Backward-compatible aliases (used by existing service files) ─────────
    # These are plain properties so they do not participate in pydantic
    # validation and do not appear in model_dump().

    @property
    def llm_provider(self) -> str:
        """Alias for ``provider`` (backward compat)."""
        return self.provider

    @property
    def llm_temperature(self) -> float:
        """Returns ``student_temperature`` (backward compat default)."""
        return self.student_temperature

    @property
    def llm_max_retries(self) -> int:
        """Alias for ``max_retries`` (backward compat)."""
        return self.max_retries

    @property
    def llm_timeout_seconds(self) -> int:
        """Alias for ``timeout_seconds`` (backward compat)."""
        return self.timeout_seconds

    @property
    def max_input_chars(self) -> int:
        """Alias for ``max_context_chars`` (backward compat)."""
        return self.max_context_chars

    @property
    def anthropic_max_tokens(self) -> int:
        """Alias for ``max_output_chars`` (backward compat)."""
        return self.max_output_chars

    @property
    def llm_primary_label(self) -> str:
        """Human-readable label for the primary model (backward compat)."""
        return self.primary_model

    @property
    def llm_fallback_label(self) -> str:
        """Human-readable label for the fallback model (backward compat)."""
        return self.fallback_model


# Module-level singleton — imported by all Phase 10 service files.
phase10_settings = LangChainSettings()
