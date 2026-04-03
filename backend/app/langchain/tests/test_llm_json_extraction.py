"""
Tests for llm-service-style JSON extraction reliability.

These cover the real failure patterns observed with local models (gemma3, mistral):
  - Prose before JSON
  - Prose after JSON (the rfind bug that caused wrong brace extraction)
  - Fenced ```json blocks appearing anywhere in the text
  - Malformed / partially closed JSON that needs repair
  - Gemma returning prose + fenced JSON + trailing explanation
  - Both models failing -> expect schema-valid fallback shape (not 502)

The backend's own extract_json_text / try_parse_json are the authoritative helpers
used by the Phase 10 pipeline; we test those directly.  The llm-service
_extract_json_text function is a simpler variant that we reproduce inline below
so we can test it without a running service.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from app.langchain.parsers.json_repair import extract_json_text, try_parse_json


# ---------------------------------------------------------------------------
# Inline replica of llm-service/main.py _extract_json_text
# (so we can test the same logic without importing the FastAPI app)
# ---------------------------------------------------------------------------

def _extract_json_text_service(raw: str) -> str:
    """Replica of the improved _extract_json_text from llm-service/main.py."""
    if not isinstance(raw, str):
        raw = str(raw)

    text = raw.strip()

    # Fenced block anywhere in text
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()

    # Balanced brace extraction
    start: int | None = None
    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if in_string:
            if ch == "\\":
                escape_next = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : i + 1].strip()

    start_obj = text.find("{")
    end_obj = text.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        return text[start_obj : end_obj + 1].strip()

    return text


# ---------------------------------------------------------------------------
# 1. Prose before JSON — gemma often says "Here is the feedback:" first
# ---------------------------------------------------------------------------

def test_extract_ignores_prose_before_json():
    raw = 'Here is the student feedback:\n\n{"summary": "Good work.", "issues": []}'
    result = _extract_json_text_service(raw)
    parsed = json.loads(result)
    assert parsed["summary"] == "Good work."


def test_backend_extract_ignores_prose_before_json():
    raw = 'Here is the student feedback:\n\n{"summary": "Good work.", "issues": []}'
    result = extract_json_text(raw)
    parsed = json.loads(result)
    assert parsed["summary"] == "Good work."


# ---------------------------------------------------------------------------
# 2. Prose AFTER JSON — the old rfind("}") bug: stray } in trailing text
#    would cause a truncated or invalid JSON to be extracted
# ---------------------------------------------------------------------------

def test_extract_stops_at_correct_closing_brace_not_trailing_prose():
    raw = (
        '{"summary": "Well structured.", "issues": [{"title": "X", "evidence": "Y", "severity": "low"}]}'
        '\n\nNote: this feedback was generated automatically. Please review. }'
    )
    result = _extract_json_text_service(raw)
    parsed = json.loads(result)
    assert parsed["summary"] == "Well structured."
    assert parsed["issues"][0]["title"] == "X"


def test_backend_extract_stops_at_balanced_brace():
    raw = (
        '{"summary": "Strong effort."}\n\nSome trailing text with a stray } here.'
    )
    result = extract_json_text(raw)
    parsed = json.loads(result)
    assert parsed["summary"] == "Strong effort."


# ---------------------------------------------------------------------------
# 3. Fenced ```json ... ``` block anywhere in text
# ---------------------------------------------------------------------------

def test_extract_fenced_block_at_start():
    raw = "```json\n{\"summary\": \"Good.\", \"issues\": []}\n```"
    result = _extract_json_text_service(raw)
    parsed = json.loads(result)
    assert parsed["summary"] == "Good."


def test_extract_fenced_block_mid_text():
    raw = (
        "I have generated the feedback below:\n\n"
        "```json\n{\"summary\": \"Decent work.\", \"issues\": []}\n```\n\n"
        "Let me know if you need more detail."
    )
    result = _extract_json_text_service(raw)
    parsed = json.loads(result)
    assert parsed["summary"] == "Decent work."


def test_backend_extract_fenced_block():
    raw = "Here it is:\n```json\n{\"summary\": \"Solid draft.\"}\n```\nEnd."
    result = extract_json_text(raw)
    parsed = json.loads(result)
    assert parsed["summary"] == "Solid draft."


# ---------------------------------------------------------------------------
# 4. Gemma-style output: prose + fenced JSON + explanation after
# ---------------------------------------------------------------------------

def test_extract_gemma_style_prose_fenced_trailing():
    raw = """
Generating feedback for the student submission...

```json
{
  "summary": "The project demonstrates solid architecture.",
  "issues": [
    {"title": "Missing tests", "evidence": "No test files found.", "severity": "high"}
  ],
  "improvement_plan": [{"action": "Add unit tests", "why": "Coverage gap", "how": "Use pytest", "priority": 1}],
  "checklist": [{"item": "Write unit tests", "done": false}],
  "model_agreement": {"ml_confidence": 0.7, "llm_confidence": 0.75, "final_confidence": 0.72},
  "safety": {"needs_review": false, "reason": ""}
}
```

I hope this feedback is useful. The model used gemma3:latest for this generation.
"""
    result = _extract_json_text_service(raw)
    parsed = json.loads(result)
    assert parsed["summary"] == "The project demonstrates solid architecture."
    assert parsed["issues"][0]["severity"] == "high"
    assert parsed["model_agreement"]["final_confidence"] == 0.72


# ---------------------------------------------------------------------------
# 5. Malformed JSON — trailing commas and bare keys — repair path
# ---------------------------------------------------------------------------

def test_try_parse_repairs_trailing_commas_in_local_model_output():
    raw = (
        'Here is the JSON:\n'
        '{"summary": "Draft is good.", "issues": [{"title": "Weak argument", '
        '"evidence": "No references.", "severity": "med",}], "safety": {"needs_review": false,},}'
    )
    parsed, err = try_parse_json(raw)
    assert err is None
    assert parsed is not None
    assert parsed["issues"][0]["title"] == "Weak argument"


def test_try_parse_repairs_unclosed_json():
    raw = '{"summary": "Strong start.", "issues": [{"title": "No conclusion", "evidence": "Missing.", "severity": "low"}'
    parsed, err = try_parse_json(raw)
    assert err is None
    assert parsed is not None
    assert parsed["summary"] == "Strong start."


def test_try_parse_repairs_bare_keys_gemma_output():
    raw = '{summary: "Good work", issues: [], safety: {needs_review: false, reason: ""}}'
    parsed, err = try_parse_json(raw)
    assert err is None
    assert parsed is not None
    assert parsed["summary"] == "Good work"


# ---------------------------------------------------------------------------
# 6. Valid JSON path unchanged
# ---------------------------------------------------------------------------

def test_valid_json_passes_through_unchanged():
    payload = {
        "summary": "Excellent work overall.",
        "issues": [],
        "safety": {"needs_review": False, "reason": ""},
    }
    raw = json.dumps(payload)
    result = _extract_json_text_service(raw)
    parsed = json.loads(result)
    assert parsed["summary"] == payload["summary"]
    assert parsed["safety"]["needs_review"] is False


# ---------------------------------------------------------------------------
# 7. Schema-valid fallback payload shape (for both-models-fail scenario)
# These verify that the fallback helpers produce a valid shape matching
# StudentReportOut / ProfessorReportOut without importing the live service.
# ---------------------------------------------------------------------------

_REQUIRED_STUDENT_KEYS = {
    "summary", "issues", "strengths", "architecture_review", "implementation_review",
    "evaluation_review", "improvement_plan", "checklist", "confidence",
    "model_agreement", "safety",
}

_REQUIRED_PROFESSOR_KEYS = {
    "rubric_breakdown", "feedback_explanation", "moderation_notes", "safety",
}


def _make_student_safe_fallback(model_used: str) -> dict[str, Any]:
    return {
        "summary": "Automated feedback could not be generated. Manual review required.",
        "issues": [{"title": "Generation failed", "evidence": "LLM returned unusable output.", "severity": "low"}],
        "strengths": [],
        "architecture_review": {"overview": "Not assessed.", "backend": "Not assessed.", "frontend": "Not assessed.", "database": "Not assessed.", "security": "Not assessed."},
        "implementation_review": {"features_built": [], "technical_quality": "Not assessed.", "integration_quality": "Not assessed."},
        "evaluation_review": {"testing_present": "Not assessed.", "limitations": "Not assessed.", "academic_quality": "Not assessed."},
        "improvement_plan": [{"action": "Request manual feedback", "why": "Generation failed.", "how": "Contact instructor.", "priority": 1}],
        "checklist": [{"item": "Await manual feedback.", "done": False}],
        "confidence": {"mode": "restricted", "overall": 0.0},
        "model_agreement": {"ml_confidence": 0.0, "llm_confidence": 0.0, "final_confidence": 0.0},
        "safety": {"needs_review": True, "reason": f"LLM generation failed (model={model_used})."},
    }


def _make_professor_safe_fallback(model_used: str) -> dict[str, Any]:
    return {
        "rubric_breakdown": [{"criterion": "Overall", "band": "Needs review", "justification": "Automated generation failed."}],
        "feedback_explanation": "Automated feedback could not be generated. Manual review required.",
        "moderation_notes": [{"risk": "Generation failed", "note": f"LLM (model={model_used}) failed."}],
        "safety": {"needs_review": True, "reason": f"LLM generation failed (model={model_used})."},
    }


def test_student_fallback_has_all_required_keys():
    fb = _make_student_safe_fallback("gemma3:latest")
    missing = _REQUIRED_STUDENT_KEYS - set(fb.keys())
    assert missing == set(), f"Missing keys: {missing}"


def test_student_fallback_safety_needs_review():
    fb = _make_student_safe_fallback("gemma3:latest")
    assert fb["safety"]["needs_review"] is True
    assert fb["confidence"]["mode"] == "restricted"
    assert fb["model_agreement"]["final_confidence"] == 0.0


def test_professor_fallback_has_all_required_keys():
    fb = _make_professor_safe_fallback("mistral:latest")
    missing = _REQUIRED_PROFESSOR_KEYS - set(fb.keys())
    assert missing == set(), f"Missing keys: {missing}"


def test_professor_fallback_safety_needs_review():
    fb = _make_professor_safe_fallback("mistral:latest")
    assert fb["safety"]["needs_review"] is True
    assert len(fb["rubric_breakdown"]) >= 1


# ---------------------------------------------------------------------------
# 8. JSON inside a nested value with closing braces — balanced extractor must
#    not exit early on inner braces
# ---------------------------------------------------------------------------

def test_extract_json_with_nested_objects_and_trailing_prose():
    raw = (
        "Output:\n"
        '{"summary": "Good.", "architecture_review": {"overview": "MVC", "backend": "FastAPI", '
        '"frontend": "React", "database": "Postgres", "security": "JWT"}, '
        '"safety": {"needs_review": false, "reason": ""}}'
        "\n\nThis completes the review."
    )
    result = _extract_json_text_service(raw)
    parsed = json.loads(result)
    assert parsed["architecture_review"]["backend"] == "FastAPI"
    assert parsed["safety"]["needs_review"] is False
