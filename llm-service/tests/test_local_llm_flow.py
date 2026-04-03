from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))


def _purge_service_modules() -> None:
    for name in list(sys.modules):
        if name == "main" or name == "app" or name.startswith("app."):
            sys.modules.pop(name, None)


def _load_service_main(monkeypatch: pytest.MonkeyPatch, **env: str):
    keys = {
        "LLM_PROVIDER",
        "ANTHROPIC_API_KEY",
        "OLLAMA_PRIMARY_MODEL",
        "OLLAMA_FALLBACK_MODEL",
        "LLM_SERVICE_SECRET",
    }
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    _purge_service_modules()
    return importlib.import_module("main")


def _student_payload(service_main):
    return service_main.StudentReportIn.model_validate(
        {
            "submission_id": "sub-1",
            "ingestion": {
                "text_content": "This submission describes a small web app with FastAPI and React.",
                "ocr_text": "",
                "audio_transcript": "",
                "tables_json": None,
            },
            "ml": {
                "feedback_category": "project_review",
                "quality_band": "med",
                "confidence_0_to_4": 3,
            },
            "analysis_type": "student_project_review",
        }
    )


def _professor_payload(service_main):
    return service_main.ProfessorReportIn.model_validate(
        {
            "submission_id": "sub-2",
            "ingestion": {
                "text_content": "The submission evaluates software architecture trade-offs.",
                "ocr_text": "",
                "audio_transcript": "",
                "tables_json": None,
            },
            "ml": {
                "rubric_band": "merit",
                "argument_depth": "med",
                "moderation_consistency": "med",
            },
            "analysis_type": "professor_academic_review",
        }
    )


def _student_json(summary: str = "A grounded student review.") -> str:
    return json.dumps(
        {
            "summary": summary,
            "issues": [],
            "strengths": [],
            "architecture_review": {
                "overview": "Clear overall structure.",
                "backend": "FastAPI service layer present.",
                "frontend": "React client present.",
                "database": "Not assessed.",
                "security": "Not assessed.",
            },
            "implementation_review": {
                "features_built": ["Dashboard"],
                "technical_quality": "Technically coherent.",
                "integration_quality": "Reasonable component integration.",
            },
            "evaluation_review": {
                "testing_present": "Limited evidence.",
                "limitations": "Some detail is missing.",
                "academic_quality": "Acceptable technical reflection.",
            },
            "improvement_plan": [
                {
                    "action": "Add tests",
                    "why": "Coverage is limited.",
                    "how": "Introduce unit and integration tests.",
                    "priority": 1,
                }
            ],
            "checklist": [{"item": "Add tests", "done": False}],
            "confidence": {"mode": "normal", "overall": 0.74},
            "model_agreement": {
                "ml_confidence": 0.75,
                "llm_confidence": 0.73,
                "final_confidence": 0.74,
            },
            "safety": {"needs_review": False, "reason": ""},
        }
    )


def _professor_json(feedback: str = "A defensible moderation summary.") -> str:
    return json.dumps(
        {
            "rubric_breakdown": [
                {
                    "criterion": "Overall academic quality",
                    "band": "Merit",
                    "justification": "The submission is coherent and reasonably evidenced.",
                }
            ],
            "feedback_explanation": feedback,
            "moderation_notes": [
                {
                    "risk": "Evidence scope",
                    "note": "Some claims would benefit from fuller support.",
                }
            ],
            "safety": {"needs_review": False, "reason": ""},
        }
    )


def test_placeholder_anthropic_key_forces_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    service_main = _load_service_main(
        monkeypatch,
        LLM_PROVIDER="anthropic",
        ANTHROPIC_API_KEY="sk-ant-...",
        OLLAMA_PRIMARY_MODEL="gemma3:latest",
        OLLAMA_FALLBACK_MODEL="mistral:latest",
    )

    assert service_main.settings.effective_provider == "ollama"
    assert service_main._ACTIVE_PROVIDER == "ollama"
    assert service_main._generate_with_specific_model is not None


@pytest.mark.asyncio
async def test_generate_prompt_honors_requested_model(monkeypatch: pytest.MonkeyPatch) -> None:
    service_main = _load_service_main(
        monkeypatch,
        LLM_PROVIDER="ollama",
        OLLAMA_PRIMARY_MODEL="gemma3:latest",
        OLLAMA_FALLBACK_MODEL="mistral:latest",
        LLM_SERVICE_SECRET="dev_llm_secret",
    )
    calls: dict[str, object] = {}

    async def fake_specific(model: str, payload: dict[str, object]) -> dict[str, object]:
        calls["model"] = model
        calls["payload"] = payload
        return {
            "model_used": model,
            "response": '{"ok": true}',
            "done": True,
            "done_reason": "stop",
            "raw": {"response": '{"ok": true}'},
        }

    monkeypatch.setattr(service_main, "_generate_with_specific_model", fake_specific)

    response = await service_main.generate_prompt(
        service_main.PromptGenerateIn(
            prompt="Return a JSON object",
            role="student",
            requested_model="mistral:latest",
        ),
        x_ai_secret="dev_llm_secret",
    )

    body = json.loads(response.body)
    assert calls["model"] == "mistral:latest"
    assert body["model_used"] == "mistral:latest"
    assert body["response"] == '{"ok": true}'


def test_parse_llm_json_extracts_from_prose_and_fences(monkeypatch: pytest.MonkeyPatch) -> None:
    service_main = _load_service_main(monkeypatch, LLM_PROVIDER="ollama")
    raw = """
Here is the generated report:

```json
{"summary": "Good work", "issues": [], "safety": {"needs_review": false, "reason": ""}}
```

Thanks.
"""

    parsed = service_main._parse_llm_json(raw)
    assert parsed["summary"] == "Good work"
    assert parsed["safety"]["needs_review"] is False


def test_parse_llm_json_repairs_common_local_model_mistakes(monkeypatch: pytest.MonkeyPatch) -> None:
    service_main = _load_service_main(monkeypatch, LLM_PROVIDER="ollama")
    raw = '{summary: "Good work", issues: [], safety: {needs_review: false, reason: ""},}'

    parsed = service_main._parse_llm_json(raw)
    assert parsed["summary"] == "Good work"
    assert parsed["issues"] == []


@pytest.mark.asyncio
async def test_student_report_uses_mistral_repair_after_gemma_json_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service_main = _load_service_main(
        monkeypatch,
        LLM_PROVIDER="ollama",
        OLLAMA_PRIMARY_MODEL="gemma3:latest",
        OLLAMA_FALLBACK_MODEL="mistral:latest",
        LLM_SERVICE_SECRET="dev_llm_secret",
    )

    async def fake_generate(_prompt: str) -> dict[str, object]:
        return {
            "model_used": "gemma3:latest",
            "response": "Here is your report: definitely not JSON",
            "done": True,
            "done_reason": "stop",
            "raw": {},
        }

    async def fake_repair(_prompt: str) -> dict[str, object]:
        return {
            "model_used": "mistral:latest",
            "response": _student_json("Recovered by the fallback model."),
            "done": True,
            "done_reason": "stop",
            "raw": {},
        }

    monkeypatch.setattr(service_main, "generate_with_fallback", fake_generate)
    monkeypatch.setattr(service_main, "_call_repair_model", fake_repair)

    response = await service_main.student_report(
        _student_payload(service_main),
        x_ai_secret="dev_llm_secret",
    )

    body = json.loads(response.body)
    assert response.headers["x-llm-model-used"] == "mistral:latest"
    assert body["summary"] == "Recovered by the fallback model."
    assert body["rag_meta"]["enabled"] is False


@pytest.mark.asyncio
async def test_professor_report_returns_schema_safe_fallback_when_both_models_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service_main = _load_service_main(
        monkeypatch,
        LLM_PROVIDER="ollama",
        OLLAMA_PRIMARY_MODEL="gemma3:latest",
        OLLAMA_FALLBACK_MODEL="mistral:latest",
        LLM_SERVICE_SECRET="dev_llm_secret",
    )

    async def fake_generate(_prompt: str) -> dict[str, object]:
        return {
            "model_used": "gemma3:latest",
            "response": "Not JSON",
            "done": True,
            "done_reason": "stop",
            "raw": {},
        }

    async def fake_repair(_prompt: str) -> dict[str, object]:
        return {
            "model_used": "mistral:latest",
            "response": "Still not JSON either",
            "done": True,
            "done_reason": "stop",
            "raw": {},
        }

    monkeypatch.setattr(service_main, "generate_with_fallback", fake_generate)
    monkeypatch.setattr(service_main, "_call_repair_model", fake_repair)

    response = await service_main.professor_report(
        _professor_payload(service_main),
        x_ai_secret="dev_llm_secret",
    )

    body = json.loads(response.body)
    assert response.status_code == 200
    assert response.headers["x-llm-model-used"] == "mistral:latest"
    assert response.headers["x-llm-fallback"] == "true"
    assert body["safety"]["needs_review"] is True
    assert body["rubric_breakdown"][0]["band"] == "Needs review"


@pytest.mark.asyncio
async def test_valid_student_json_path_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    service_main = _load_service_main(
        monkeypatch,
        LLM_PROVIDER="ollama",
        OLLAMA_PRIMARY_MODEL="gemma3:latest",
        OLLAMA_FALLBACK_MODEL="mistral:latest",
        LLM_SERVICE_SECRET="dev_llm_secret",
    )

    async def fake_generate(_prompt: str) -> dict[str, object]:
        return {
            "model_used": "gemma3:latest",
            "response": _student_json("Primary model returned valid JSON."),
            "done": True,
            "done_reason": "stop",
            "raw": {},
        }

    monkeypatch.setattr(service_main, "generate_with_fallback", fake_generate)

    response = await service_main.student_report(
        _student_payload(service_main),
        x_ai_secret="dev_llm_secret",
    )

    body = json.loads(response.body)
    assert response.status_code == 200
    assert response.headers["x-llm-model-used"] == "gemma3:latest"
    assert "x-llm-fallback" not in response.headers
    assert body["summary"] == "Primary model returned valid JSON."
