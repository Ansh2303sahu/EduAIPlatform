from app.langchain.services.fallback_service import (
    build_professor_fallback_payload,
    build_student_fallback_payload,
)


_RAW_OLLAMA_DETAIL = (
    "RuntimeError: ollama primary failed: HTTPStatusError: Client error '404 Not Found' "
    "for url 'http://host.docker.internal:11434/api/generate' | "
    "fallback failed: HTTPStatusError: Client error '404 Not Found'"
)

_RAW_OLLAMA_CONNECT_DETAIL = (
    "RuntimeError: ollama primary failed: RuntimeError: Could not reach Ollama at "
    "http://host.docker.internal:11434. Start the Ollama service and ensure the "
    "configured OLLAMA_BASE_URL is reachable from the llm-service container. "
    "If Ollama runs elsewhere, update OLLAMA_BASE_URL. Requested model: gemma3:4b. | "
    "fallback failed: RuntimeError: Could not reach Ollama at http://host.docker.internal:11434."
)


def test_student_fallback_sanitizes_ollama_detail() -> None:
    payload = build_student_fallback_payload("ollama_unavailable", detail=_RAW_OLLAMA_DETAIL)

    description = payload.issues[0].description
    assert "configured local Ollama model could not be found" in description
    assert "http://host.docker.internal:11434/api/generate" not in description
    assert "HTTPStatusError" not in description


def test_professor_fallback_sanitizes_ollama_detail() -> None:
    payload = build_professor_fallback_payload("ollama_unavailable", detail=_RAW_OLLAMA_DETAIL)

    justification = payload.rubric_breakdown[0].justification
    assert "configured local Ollama model could not be found" in justification
    assert "http://host.docker.internal:11434/api/generate" not in justification
    assert "HTTPStatusError" not in justification


def test_student_fallback_sanitizes_ollama_connect_detail() -> None:
    payload = build_student_fallback_payload("ollama_unavailable", detail=_RAW_OLLAMA_CONNECT_DETAIL)

    description = payload.issues[0].description
    assert "local Ollama service could not be reached" in description
    assert "host.docker.internal" not in description
    assert "OLLAMA_BASE_URL" not in description
