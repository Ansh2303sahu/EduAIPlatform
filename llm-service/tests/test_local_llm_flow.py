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
            "retrieval_confidence_score": 0.91,
            "retrieval_confidence_label": "high",
            "retrieval_safe_review": False,
        }
    )


def _representative_project_payload(service_main):
    return service_main.StudentReportIn.model_validate(
        {
            "submission_id": "sub-live-like-1",
            "ingestion": {
                "text_content": (
                    "ANGLIA RUSKIN UNIVERSITY Final Year Dissertation. "
                    "This project presents StockIntel, a Django-based stock portfolio management platform with BUY and SELL transaction handling, "
                    "portfolio analytics, Chart.js visualizations, Django authentication, and a Gemini-powered AI assistant. "
                    "The system computes portfolio value, unrealised profit and loss, Sharpe ratio, beta, and maximum drawdown from NASDAQ 100 data. "
                    "Development followed an iterative Agile-inspired process across eight phases. "
                    "A forty-case functional acceptance test suite achieved a 100 percent pass rate, and four Django unit tests were added for regression coverage. "
                    "The report discusses dashboard render time, ORM query count, scalability limits, and AI output evaluation by two independent evaluators. "
                    "Security controls include authenticated access, sell-quantity validation, and guarded AI context handling. "
                    "The dissertation also reflects on architecture trade-offs, current limitations, and future enhancements for performance and maintainability."
                ),
                "ocr_text": "",
                "audio_transcript": "",
                "tables_json": {"metrics": ["Sharpe ratio", "beta", "drawdown"]},
            },
            "ml": {
                "feedback_category": "clarity",
                "quality_band": "high",
                "confidence_0_to_4": 4,
            },
            "analysis_type": "student_project_review",
            "query": "student software engineering project architecture testing evaluation analytics security",
            "top_k": 6,
            "retrieval_confidence_score": 0.88,
            "retrieval_confidence_label": "high",
            "retrieval_safe_review": False,
            "grounding_context": (
                "[Source 1] Clear project reports explain the implemented architecture, testing evidence, and limitations.\n"
                "[Source 2] Improvement actions should be specific and technically actionable."
            ),
            "grounding_citations": [
                {"index": 1, "title": "Project Reporting", "section": "architecture"},
                {"index": 2, "title": "Actionable Feedback", "section": "improvement"},
            ],
        }
    )


def _essay_like_payload(service_main):
    return service_main.StudentReportIn.model_validate(
        {
            "submission_id": "essay-1",
            "ingestion": {
                "text_content": (
                    "This essay argues that cloud governance frameworks must balance compliance, operational agility, and cost control. "
                    "The discussion references industry guidance, compares centralised and federated governance models, and evaluates how policy enforcement affects risk. "
                    "The introduction establishes the topic clearly, but the analysis also notes limitations in over-standardised governance and the need for evidence-led decision making. "
                    "References and citation language are discussed explicitly in the concluding section."
                ),
                "ocr_text": "",
                "audio_transcript": "",
                "tables_json": None,
            },
            "ml": {
                "feedback_category": "clarity",
                "quality_band": "high",
                "confidence_0_to_4": 3,
            },
            "analysis_type": "student_academic_review",
        }
    )


def _code_like_non_project_payload(service_main):
    return service_main.StudentReportIn.model_validate(
        {
            "submission_id": "code-1",
            "ingestion": {
                "text_content": (
                    "The repository contains a FastAPI API with authentication middleware, SQL database models, endpoint validation, unit testing, and helper functions for report generation. "
                    "Several functions handle error cases and class-based services coordinate data access, but integration testing is still limited."
                ),
                "ocr_text": "",
                "audio_transcript": "",
                "tables_json": None,
            },
            "ml": {
                "feedback_category": "clarity",
                "quality_band": "med",
                "confidence_0_to_4": 2,
            },
            "analysis_type": "student_academic_review",
        }
    )


def _representative_essay_payload(service_main):
    return service_main.StudentReportIn.model_validate(
        {
            "submission_id": "essay-live-like-1",
            "ingestion": {
                "text_content": (
                    "ABSTRACT This essay evaluates whether platform regulation should prioritise innovation or public protection. "
                    "It argues that regulation is most effective when evidence-led and proportionate rather than purely reactive. "
                    "The discussion compares two regulatory approaches, cites academic literature, and considers how weak evidence can produce shallow policy claims. "
                    "Several paragraphs explain the importance of critical analysis, coherent paragraph structure, and consistent citation practice. "
                    "The conclusion returns to the thesis and highlights the need for stronger comparative reasoning when weighing policy trade-offs."
                ),
                "ocr_text": "",
                "audio_transcript": "",
                "tables_json": None,
            },
            "ml": {
                "feedback_category": "clarity",
                "quality_band": "high",
                "confidence_0_to_4": 3,
            },
            "analysis_type": "student_academic_review",
            "grounding_context": (
                "[Source 1] Strong academic writing makes its thesis explicit and supports claims with evidence.\n"
                "[Source 2] Higher-quality essays compare viewpoints and explain why evidence matters."
            ),
            "grounding_citations": [
                {"index": 1, "title": "Academic Writing", "section": "thesis"},
                {"index": 2, "title": "Critical Analysis", "section": "evaluation"},
            ],
            "retrieval_confidence_score": 0.82,
            "retrieval_confidence_label": "high",
            "retrieval_safe_review": False,
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


def _student_substantive_json() -> str:
    return json.dumps(
        {
            "summary": (
                "StockIntel is a clearly scoped Django dissertation project with implemented transaction management, "
                "portfolio analytics, charting, and an AI assistant. The report gives concrete evidence of the stack, "
                "core features, and evaluation process, although some architecture justification and deeper risk analysis "
                "could still be sharper."
            ),
            "issues": [
                {
                    "title": "Architecture trade-offs need clearer justification",
                    "evidence": (
                        "The report names Django, Chart.js, and Gemini integration, but it only briefly explains why those choices "
                        "fit the scalability and maintainability goals of the system."
                    ),
                    "severity": "med",
                },
                {
                    "title": "Testing breadth is narrower than the feature scope",
                    "evidence": (
                        "The dissertation cites forty acceptance tests and four unit tests, which is useful evidence, "
                        "but the coverage story for edge cases, failure handling, and integration risk remains limited."
                    ),
                    "severity": "med",
                },
            ],
            "strengths": [
                {
                    "title": "Implemented feature scope is concrete",
                    "evidence": "The submission explicitly names transaction flows, analytics metrics, charting, authentication, and AI assistance."
                }
            ],
            "architecture_review": {
                "overview": "The project has a coherent full-stack structure centred on Django with analytics and AI support.",
                "backend": "The backend appears to combine transaction handling, analytics calculations, and AI request orchestration in a practical service layer.",
                "frontend": "The UI evidence points to dashboards and visual analytics rather than a purely form-based interface.",
                "database": "The data model likely supports portfolio positions, transactions, and historical market data, but deeper schema reasoning is only lightly explained.",
                "security": "Authentication and sell-quantity validation are concrete positives, though broader abuse-case handling is under-discussed."
            },
            "implementation_review": {
                "features_built": [
                    "BUY and SELL transaction management",
                    "Portfolio analytics dashboard",
                    "Chart.js price visualisation",
                    "Gemini AI assistant"
                ],
                "technical_quality": "The implementation appears substantial and technically credible for a final-year project.",
                "integration_quality": "The submission shows meaningful integration between portfolio data, analytics, charting, and AI-assisted insight generation."
            },
            "evaluation_review": {
                "testing_present": "The report includes functional testing, unit tests, scalability checks, and AI evaluation evidence.",
                "limitations": "The current evidence says less about security edge cases, performance under heavier loads, and long-term maintainability trade-offs.",
                "academic_quality": "The dissertation is technically specific and evaluative, but some decisions could be justified more explicitly."
            },
            "improvement_plan": [
                {
                    "action": "Explain architecture decisions more explicitly",
                    "why": "The stack is named clearly, but the reasons behind major design choices need stronger justification.",
                    "how": "Add a short architecture rationale section covering Django, analytics processing, AI integration boundaries, and scalability trade-offs.",
                    "priority": 1,
                },
                {
                    "action": "Expand deeper integration and edge-case testing",
                    "why": "Current testing evidence is useful but does not fully cover failure handling and riskier user journeys.",
                    "how": "Add tests for invalid transactions, AI assistant failure cases, and multi-step portfolio update flows.",
                    "priority": 2,
                },
            ],
            "checklist": [
                {"item": "Add architecture rationale for major framework choices", "done": False},
                {"item": "Add edge-case and integration tests for portfolio workflows", "done": False},
            ],
            "confidence": {"mode": "normal", "overall": 0.82},
            "model_agreement": {
                "ml_confidence": 0.9,
                "llm_confidence": 0.82,
                "final_confidence": 0.84,
            },
            "safety": {"needs_review": False, "reason": ""},
        }
    )


def _student_placeholder_json() -> str:
    return json.dumps(
        {
            "summary": "Automated review generated with limited confidence.",
            "issues": [],
            "strengths": [],
            "architecture_review": {
                "overview": "Not assessed.",
                "backend": "Not assessed.",
                "frontend": "Not assessed.",
                "database": "Not assessed.",
                "security": "Not assessed.",
            },
            "implementation_review": {
                "features_built": [],
                "technical_quality": "Not assessed.",
                "integration_quality": "Not assessed.",
            },
            "evaluation_review": {
                "testing_present": "Not assessed.",
                "limitations": "Not assessed.",
                "academic_quality": "Not assessed.",
            },
            "improvement_plan": [],
            "checklist": [],
            "confidence": {"mode": "normal", "overall": 0.75},
            "model_agreement": {
                "ml_confidence": 0.0,
                "llm_confidence": 0.0,
                "final_confidence": 0.0,
            },
            "safety": {"needs_review": False, "reason": ""},
        }
    )


def _student_alias_json(summary: str = "A concrete review from aliased keys.") -> dict[str, object]:
    return {
        "overview": summary,
        "weaknesses": [
            {
                "title": "Testing evidence is thin",
                "evidence": "The submission describes features clearly. It does not provide much direct testing evidence.",
                "severity": "med",
            }
        ],
        "positives": [
            {
                "title": "Architecture is explicit",
                "evidence": "The stack and component structure are named directly.",
            }
        ],
        "architecture": {
            "overview": "The application has a coherent structure.",
            "backend": "FastAPI handles APIs and services.",
            "frontend": "React provides the student-facing UI.",
            "database": "SQLite or PostgreSQL persistence is implied by the stack.",
            "security": "Authentication is mentioned but needs deeper evaluation.",
        },
        "implementation": {
            "features_built": ["Dashboard", "Submission upload"],
            "technical_quality": "The implementation appears coherent overall.",
            "integration_quality": "The described features suggest reasonable integration.",
        },
        "evaluation": {
            "testing_present": "Only limited testing evidence is shown.",
            "limitations": "Evidence for robustness is still thin.",
            "academic_quality": "The explanation is technically clear.",
        },
        "recommendations": [
            {
                "action": "Add integration tests",
                "why": "Testing evidence is currently limited.",
                "how": "Exercise upload and report-generation flows end to end.",
                "priority": 1,
            },
            {
                "action": "Clarify data flow",
                "why": "The report could better justify architecture choices.",
                "how": "Add a diagram or explicit request-response walkthrough.",
                "priority": 2,
            },
        ],
        "action_items": [
            {"item": "Add integration tests", "done": False},
            {"item": "Document the request flow", "done": False},
        ],
        "confidence": {"mode": "normal", "overall": 0.71},
        "safety": {"needs_review": False, "reason": ""},
    }


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
        OLLAMA_PRIMARY_MODEL="mistral:latest",
        OLLAMA_FALLBACK_MODEL="gemma3:latest",
    )

    assert service_main.settings.effective_provider == "ollama"
    assert service_main._ACTIVE_PROVIDER == "ollama"
    assert service_main._generate_with_specific_model is not None


def test_config_defaults_mistral_primary_gemma_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """When no env vars are set, mistral must be primary and gemma3 must be fallback."""
    service_main = _load_service_main(monkeypatch, LLM_PROVIDER="ollama")

    assert service_main.settings.primary_model == "mistral:latest"
    assert service_main.settings.fallback_model == "gemma3:latest"


@pytest.mark.asyncio
async def test_generate_prompt_honors_requested_model(monkeypatch: pytest.MonkeyPatch) -> None:
    service_main = _load_service_main(
        monkeypatch,
        LLM_PROVIDER="ollama",
        OLLAMA_PRIMARY_MODEL="mistral:latest",
        OLLAMA_FALLBACK_MODEL="gemma3:latest",
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
async def test_student_report_uses_gemma_repair_after_mistral_json_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Primary (mistral) returns non-JSON → repair routes to gemma3 (fallback) which succeeds."""
    service_main = _load_service_main(
        monkeypatch,
        LLM_PROVIDER="ollama",
        OLLAMA_PRIMARY_MODEL="mistral:latest",
        OLLAMA_FALLBACK_MODEL="gemma3:latest",
        LLM_SERVICE_SECRET="dev_llm_secret",
    )

    async def fake_generate(_prompt: str) -> dict[str, object]:
        return {
            "model_used": "mistral:latest",
            "response": "Here is your report: definitely not JSON",
            "done": True,
            "done_reason": "stop",
            "raw": {},
        }

    async def fake_repair(_prompt: str) -> dict[str, object]:
        return {
            "model_used": "gemma3:latest",
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
    assert response.headers["x-llm-model-used"] == "gemma3:latest"
    assert body["summary"] == "Recovered by the fallback model."
    assert body["rag_meta"]["enabled"] is False


@pytest.mark.asyncio
async def test_student_report_low_content_triggers_full_prompt_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Primary returns sparse parseable JSON → fallback model reruns the full prompt."""
    service_main = _load_service_main(
        monkeypatch,
        LLM_PROVIDER="ollama",
        OLLAMA_PRIMARY_MODEL="mistral:latest",
        OLLAMA_FALLBACK_MODEL="gemma3:latest",
        LLM_SERVICE_SECRET="dev_llm_secret",
    )

    schema_invalid_json = json.dumps({"notes": "incomplete output from model"})

    async def fake_generate(_prompt: str) -> dict[str, object]:
        return {
            "model_used": "mistral:latest",
            "response": schema_invalid_json,
            "done": True,
            "done_reason": "stop",
            "raw": {},
        }

    retry_called: list[bool] = []

    async def fake_content_retry(_prompt: str) -> dict[str, object]:
        retry_called.append(True)
        return {
            "model_used": "gemma3:latest",
            "response": _student_json("Gemma3 recovered from validation failure."),
            "done": True,
            "done_reason": "stop",
            "raw": {},
        }

    monkeypatch.setattr(service_main, "generate_with_fallback", fake_generate)
    monkeypatch.setattr(service_main, "_call_content_retry_model", fake_content_retry)

    response = await service_main.student_report(
        _student_payload(service_main),
        x_ai_secret="dev_llm_secret",
    )

    body = json.loads(response.body)
    assert retry_called, "fallback model was never called after low-content output"
    assert response.headers["x-llm-model-used"] == "gemma3:latest"
    assert body["summary"] == "Gemma3 recovered from validation failure."
    assert body["safety"]["needs_review"] is False


@pytest.mark.asyncio
async def test_professor_report_returns_schema_safe_fallback_when_both_models_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service_main = _load_service_main(
        monkeypatch,
        LLM_PROVIDER="ollama",
        OLLAMA_PRIMARY_MODEL="mistral:latest",
        OLLAMA_FALLBACK_MODEL="gemma3:latest",
        LLM_SERVICE_SECRET="dev_llm_secret",
    )

    async def fake_generate(_prompt: str) -> dict[str, object]:
        return {
            "model_used": "mistral:latest",
            "response": "Not JSON",
            "done": True,
            "done_reason": "stop",
            "raw": {},
        }

    async def fake_repair(_prompt: str) -> dict[str, object]:
        return {
            "model_used": "gemma3:latest",
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
    assert response.headers["x-llm-model-used"] == "gemma3:latest"
    assert response.headers["x-llm-fallback"] == "true"
    assert body["safety"]["needs_review"] is True
    assert body["rubric_breakdown"][0]["band"] == "Needs review"


@pytest.mark.asyncio
async def test_valid_student_json_path_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    service_main = _load_service_main(
        monkeypatch,
        LLM_PROVIDER="ollama",
        OLLAMA_PRIMARY_MODEL="mistral:latest",
        OLLAMA_FALLBACK_MODEL="gemma3:latest",
        LLM_SERVICE_SECRET="dev_llm_secret",
    )

    async def fake_generate(_prompt: str) -> dict[str, object]:
        return {
            "model_used": "mistral:latest",
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
    assert response.headers["x-llm-model-used"] == "mistral:latest"
    assert "x-llm-fallback" not in response.headers
    assert body["summary"] == "Primary model returned valid JSON."
    assert body["safety"]["needs_review"] is False
    assert body["safety"]["reason"] == ""
    assert response.headers["x-llm-primary-model"] == "mistral:latest"
    assert response.headers["x-llm-fallback-model"] == "gemma3:latest"


@pytest.mark.asyncio
async def test_placeholder_student_json_is_flagged_low_content_quality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service_main = _load_service_main(
        monkeypatch,
        LLM_PROVIDER="ollama",
        OLLAMA_PRIMARY_MODEL="mistral:latest",
        OLLAMA_FALLBACK_MODEL="gemma3:latest",
        LLM_SERVICE_SECRET="dev_llm_secret",
    )

    async def fake_generate(_prompt: str) -> dict[str, object]:
        return {
            "model_used": "mistral:latest",
            "response": _student_placeholder_json(),
            "done": True,
            "done_reason": "stop",
            "raw": {},
        }

    async def fake_content_retry(_prompt: str) -> dict[str, object]:
        return {
            "model_used": "gemma3:latest",
            "response": _student_placeholder_json(),
            "done": True,
            "done_reason": "stop",
            "raw": {},
        }

    monkeypatch.setattr(service_main, "generate_with_fallback", fake_generate)
    monkeypatch.setattr(service_main, "_call_content_retry_model", fake_content_retry)

    response = await service_main.student_report(
        _student_payload(service_main),
        x_ai_secret="dev_llm_secret",
    )

    body = json.loads(response.body)
    assert response.status_code == 200
    assert response.headers["x-llm-model-used"] == "gemma3:latest"
    assert response.headers["x-llm-fallback"] == "true"
    assert body["summary"] == "Automated review generated with limited confidence."
    assert body["safety"]["needs_review"] is True
    assert body["safety"]["reason"] == "low_content_quality"
    assert body["confidence"]["overall"] == pytest.approx(0.35)
    assert body["rag_meta"]["quality_gate"] == "low_content_quality"


@pytest.mark.asyncio
async def test_representative_project_payload_produces_substantive_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service_main = _load_service_main(
        monkeypatch,
        LLM_PROVIDER="ollama",
        OLLAMA_PRIMARY_MODEL="gemma3:latest",
        OLLAMA_FALLBACK_MODEL="mistral:latest",
        LLM_SERVICE_SECRET="dev_llm_secret",
    )

    payload = _representative_project_payload(service_main)
    captured: dict[str, str] = {}

    async def fake_generate(prompt: str) -> dict[str, object]:
        captured["prompt"] = prompt
        return {
            "model_used": "gemma3:latest",
            "response": _student_placeholder_json(),
            "done": True,
            "done_reason": "stop",
            "raw": {},
        }

    async def fake_content_retry(_prompt: str) -> dict[str, object]:
        return {
            "model_used": "mistral:latest",
            "response": _student_placeholder_json(),
            "done": True,
            "done_reason": "stop",
            "raw": {},
        }

    monkeypatch.setattr(service_main, "generate_with_fallback", fake_generate)
    monkeypatch.setattr(service_main, "_call_content_retry_model", fake_content_retry)

    response = await service_main.student_report(
        payload,
        x_ai_secret="dev_llm_secret",
    )

    body = json.loads(response.body)
    prompt_text = captured["prompt"]
    assert "Submission evidence digest:" in prompt_text
    assert "do not leave issues, improvement_plan, and checklist all empty" in prompt_text.lower()
    assert "StockIntel" in prompt_text
    assert len(prompt_text) < 16000
    assert service_main._is_student_placeholder_summary(body["summary"]) is False
    assert len(body["issues"]) >= 1
    assert len(body["strengths"]) >= 1
    assert len(body["improvement_plan"]) >= 1
    assert len(body["checklist"]) >= 1
    assert body["confidence"]["overall"] > 0.0
    assert body["model_agreement"]["final_confidence"] > 0.0
    assert body["safety"]["needs_review"] is False
    assert body["rag_meta"]["recovery"] == "heuristic_student_builder"


def test_student_prompt_specializes_for_essay_style(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service_main = _load_service_main(monkeypatch, LLM_PROVIDER="ollama")
    payload = _essay_like_payload(service_main)

    prompt_text = service_main.student_prompt(payload, safe_mode=False)

    assert service_main.student_feedback_style(payload) == "essay"
    assert "strict university marker" in prompt_text.lower()
    assert "critical analysis" in prompt_text.lower()
    assert "referencing" in prompt_text.lower()
    assert "2 strengths, 2-3 issues, 2-3 improvement actions, and 3 checklist items" in prompt_text.lower()


def test_student_prompt_treats_security_analysis_report_as_essay_style(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service_main = _load_service_main(monkeypatch, LLM_PROVIDER="ollama")
    payload = service_main.StudentReportIn.model_validate(
        {
            "submission_id": "essay-security-1",
            "ingestion": {
                "text_content": (
                    "This report analyses Mars University's cybersecurity risks using the CIA triad, NIST RMF, and STRIDE. "
                    "It identifies threats, evaluates mitigation options, and proposes an incident response framework for the institution. "
                    "The discussion references policy evidence and concludes with recommendations for governance and risk prioritisation."
                ),
                "ocr_text": "",
                "audio_transcript": "",
                "tables_json": None,
            },
            "ml": {
                "feedback_category": "clarity",
                "quality_band": "high",
                "confidence_0_to_4": 3,
            },
            "analysis_type": "student_academic_review",
        }
    )

    prompt_text = service_main.student_prompt(payload, safe_mode=False).lower()

    assert service_main.student_feedback_style(payload) == "essay"
    assert "strict university marker" in prompt_text
    assert "do not use generic software-review language" in prompt_text


def test_student_prompt_specializes_for_code_style_even_without_project_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service_main = _load_service_main(monkeypatch, LLM_PROVIDER="ollama")
    payload = _code_like_non_project_payload(service_main)

    prompt_text = service_main.student_prompt(payload, safe_mode=False)

    assert service_main.student_feedback_style(payload) == "code"
    assert "senior software reviewer" in prompt_text.lower()
    assert "correctness" in prompt_text.lower()
    assert "maintainability" in prompt_text.lower()
    assert "testing" in prompt_text.lower()
    assert "security" in prompt_text.lower()


@pytest.mark.asyncio
async def test_representative_essay_payload_produces_substantive_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service_main = _load_service_main(
        monkeypatch,
        LLM_PROVIDER="ollama",
        OLLAMA_PRIMARY_MODEL="gemma3:latest",
        OLLAMA_FALLBACK_MODEL="mistral:latest",
        LLM_SERVICE_SECRET="dev_llm_secret",
    )

    payload = _representative_essay_payload(service_main)
    captured: dict[str, str] = {}

    async def fake_generate(prompt: str) -> dict[str, object]:
        captured["prompt"] = prompt
        return {
            "model_used": "gemma3:latest",
            "response": _student_placeholder_json(),
            "done": True,
            "done_reason": "stop",
            "raw": {},
        }

    async def fake_content_retry(_prompt: str) -> dict[str, object]:
        return {
            "model_used": "mistral:latest",
            "response": _student_placeholder_json(),
            "done": True,
            "done_reason": "stop",
            "raw": {},
        }

    monkeypatch.setattr(service_main, "generate_with_fallback", fake_generate)
    monkeypatch.setattr(service_main, "_call_content_retry_model", fake_content_retry)

    response = await service_main.student_report(
        payload,
        x_ai_secret="dev_llm_secret",
    )

    body = json.loads(response.body)
    prompt_text = captured["prompt"].lower()
    assert "strict university marker" in prompt_text
    assert "critical analysis" in prompt_text
    assert "submission evidence digest:" in prompt_text
    assert service_main._is_student_placeholder_summary(body["summary"]) is False
    assert len(body["issues"]) >= 1
    assert len(body["strengths"]) >= 1
    assert len(body["improvement_plan"]) >= 1
    assert len(body["checklist"]) >= 1
    assert body["confidence"]["overall"] > 0.0
    assert body["model_agreement"]["final_confidence"] > 0.0
    assert any(issue["severity"] in {"med", "high"} for issue in body["issues"])
    assert body["rag_meta"]["recovery"] == "heuristic_student_builder"


def test_student_normalization_preserves_meaningful_alias_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service_main = _load_service_main(monkeypatch, LLM_PROVIDER="ollama")

    normalized = service_main._normalize_student_llm_json(
        _student_alias_json(),
        safe_mode=False,
    )

    assert normalized["summary"] == "A concrete review from aliased keys."
    assert normalized["issues"][0]["title"] == "Testing evidence is thin"
    assert normalized["strengths"][0]["title"] == "Architecture is explicit"
    assert normalized["improvement_plan"][0]["action"] == "Add integration tests"
    assert normalized["checklist"][0]["item"] == "Add integration tests"
    assert normalized["model_agreement"]["llm_confidence"] == pytest.approx(0.71)
    assert service_main._student_report_low_content_quality(normalized) is False


# ---------------------------------------------------------------------------
# Prompt trimming / resilience tests
# ---------------------------------------------------------------------------

def _make_student_payload_with_text(text: str, analysis_type: str = "") -> dict:
    return {
        "submission_id": "sub-trim-1",
        "ingestion": {
            "text_content": text,
            "ocr_text": "",
            "audio_transcript": "",
            "tables_json": None,
        },
        "ml": {
            "feedback_category": "project_review",
            "quality_band": "med",
            "confidence_0_to_4": 3,
        },
        "analysis_type": analysis_type,
    }


def test_long_essay_input_produces_compact_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """A long essay-style submission must produce a prompt under a sane char budget
    and must flag was_trimmed=True, while preserving essay specialization."""
    from app.prompts import student_prompt, LONG_INPUT_THRESHOLD, _EXCERPT_ESSAY_LONG
    from app.schemas import StudentReportIn

    long_text = "This essay critically analyses the CIA triad framework. " * 300  # ~16 000 chars
    assert len(long_text) > LONG_INPUT_THRESHOLD

    payload = StudentReportIn.model_validate(
        _make_student_payload_with_text(long_text, analysis_type="")
    )
    prompt_text, excerpt_cap, was_trimmed = student_prompt(payload, safe_mode=False)

    assert was_trimmed is True
    assert excerpt_cap == _EXCERPT_ESSAY_LONG
    # The essay persona must appear (not the code persona)
    assert "university marker" in prompt_text or "academic assessor" in prompt_text
    assert "ASSESSOR STYLE" in prompt_text
    # No code-review language in the essay branch
    assert "software engineering" not in prompt_text.lower() or "assessor" in prompt_text
    # Prompt must stay well under Mistral's practical context limit (~32k tokens ≈ 120k chars)
    assert len(prompt_text) < 50_000


def test_long_code_input_produces_compact_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """A long code-style submission must produce a trimmed prompt while keeping code specialization."""
    from app.prompts import student_prompt, LONG_INPUT_THRESHOLD, _EXCERPT_CODE_LONG
    from app.schemas import StudentReportIn

    long_text = (
        "The backend uses FastAPI with SQLAlchemy ORM and JWT authentication. "
        "The frontend is built with React and TailwindCSS. "
    ) * 200  # ~20 000 chars
    assert len(long_text) > LONG_INPUT_THRESHOLD

    payload = StudentReportIn.model_validate(
        _make_student_payload_with_text(long_text, analysis_type="student_project_review")
    )
    prompt_text, excerpt_cap, was_trimmed = student_prompt(payload, safe_mode=False)

    assert was_trimmed is True
    assert excerpt_cap == _EXCERPT_CODE_LONG
    # Code reviewer persona must appear
    assert "software" in prompt_text.lower() or "computing assessor" in prompt_text
    # Key code-mode instructions must survive trimming
    assert "architecture_review" in prompt_text
    assert "features_built" in prompt_text
    assert len(prompt_text) < 50_000


def test_essay_code_specialization_survives_trimming(monkeypatch: pytest.MonkeyPatch) -> None:
    """Essay and code prompts must remain distinct even when both inputs are long enough to trim."""
    from app.prompts import student_prompt, LONG_INPUT_THRESHOLD
    from app.schemas import StudentReportIn

    long_text = "x " * (LONG_INPUT_THRESHOLD // 2 + 500)  # just over threshold

    essay_payload = StudentReportIn.model_validate(
        _make_student_payload_with_text(long_text + " thesis argument essay", analysis_type="")
    )
    code_payload = StudentReportIn.model_validate(
        _make_student_payload_with_text(long_text + " FastAPI React backend", analysis_type="student_project_review")
    )

    essay_prompt, essay_cap, essay_trimmed = student_prompt(essay_payload, safe_mode=False)
    code_prompt, code_cap, code_trimmed = student_prompt(code_payload, safe_mode=False)

    # Caps must differ (essay < code for the same length)
    assert essay_cap < code_cap
    # Essay prompt must not contain code-only schema keys
    assert "architecture_review" not in essay_prompt
    assert "features_built" not in essay_prompt
    # Code prompt must contain code-only schema keys
    assert "architecture_review" in code_prompt
    assert "features_built" in code_prompt
