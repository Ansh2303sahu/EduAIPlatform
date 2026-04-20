"""Configuration for the Phase 15/16 generative AI layer."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILES = tuple(
    str(path)
    for path in (_BACKEND_ROOT / ".env",)
    if path.exists()
)


class GenAISettings(BaseSettings):
    """Execution settings for the structured generative/reporting layer."""

    model_config = SettingsConfigDict(
        env_prefix="GENAI_",
        env_file=_ENV_FILES,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    enabled: bool = Field(default=True)
    primary_model: str = Field(default="gemma3:4b")
    validator_model: str = Field(default="phi3:mini")
    graph_version: str = Field(default="15.16.0")
    prompt_version: str = Field(default="phase15_16.v2")
    output_version: str = Field(default="phase15_16.output.v1")
    schema_version: str = Field(default="phase15_16.schema.v1")
    pipeline_label: str = Field(default="phase15_phase16_genai")
    max_strengths: int = Field(default=4, ge=1, le=8)
    max_weaknesses: int = Field(default=5, ge=1, le=10)
    max_suggestions: int = Field(default=5, ge=1, le=10)
    max_evidence_references: int = Field(default=6, ge=1, le=12)
    max_reasoning_steps: int = Field(default=6, ge=2, le=12)
    include_pdf_in_audit: bool = Field(default=False)
    pdf_enabled: bool = Field(default=True)
    pdf_title_prefix: str = Field(default="EduAIPlatform Report")
    explanation_max_features: int = Field(default=5, ge=3, le=10)
    confidence_high_threshold: float = Field(default=0.75, gt=0.0, lt=1.0)
    confidence_medium_threshold: float = Field(default=0.45, gt=0.0, lt=1.0)


@lru_cache(maxsize=1)
def get_genai_settings() -> GenAISettings:
    return GenAISettings()


genai_settings = get_genai_settings()
