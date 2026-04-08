"""Phase 15/16 generative AI services for EduAIPlatform."""

from __future__ import annotations

from typing import Any

from .config import genai_settings

__all__ = ["GenAIService", "genai_settings"]


def __getattr__(name: str) -> Any:  # pragma: no cover - import indirection
    if name == "GenAIService":
        from .service import GenAIService

        return GenAIService
    raise AttributeError(name)
