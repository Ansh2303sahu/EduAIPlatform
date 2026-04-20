"""LLM invocation helpers for Phase 15/16 graph nodes."""

from __future__ import annotations

from typing import Any

from app.genai.config import genai_settings
from app.langchain.parsers.json_repair import try_parse_json
from app.langchain.services.chain_factory import (
    build_generation_chain,
    build_professor_model,
    build_student_model,
)


async def invoke_json_prompt(
    *,
    role: str,
    prompt_text: str,
    model_name: str,
) -> tuple[dict[str, Any] | None, str, str]:
    """Invoke llm-service via the secure chain factory and parse JSON output."""

    model = (
        build_student_model(primary=True, model_name=model_name)
        if role == "student"
        else build_professor_model(primary=True, model_name=model_name)
    )
    chain = build_generation_chain(model)
    try:
        raw_text = await chain.ainvoke(
            {
                "prompt_text": prompt_text,
                "submission_chars": len(prompt_text),
            }
        )
    except Exception as exc:
        return None, model_name, str(exc)

    parsed, error = try_parse_json(raw_text)
    return parsed, model_name, error or ""


def primary_model_name() -> str:
    return genai_settings.primary_model


def validator_model_name() -> str:
    return genai_settings.validator_model
