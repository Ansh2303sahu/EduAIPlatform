"""
Tests for parser extraction, repair, validation, and normalization helpers.
"""

from __future__ import annotations

import json

from app.langchain.parsers.json_repair import extract_json_text, repair_json_text, try_parse_json
from app.langchain.parsers.normalizer import (
    normalize_professor_payload,
    normalize_student_payload,
)
from app.langchain.parsers.validators import (
    validate_professor_payload,
    validate_student_payload,
)


def test_extract_json_text_with_prose_and_fence():
    raw = """Here is the report:

```json
{"summary": "Good work", "issues": []}
```

Thanks.
"""
    assert extract_json_text(raw) == '{"summary": "Good work", "issues": []}'


def test_repair_json_text_repairs_python_style_payload():
    raw = """```json
{'summary': 'Good work', 'issues': [], 'strengths': [],}
```"""
    repaired = repair_json_text(raw)
    assert json.loads(repaired)["summary"] == "Good work"


def test_repair_json_text_normalizes_smart_quotes_and_bare_values():
    raw = '“summary”: “Strong draft”, “confidence”: {“mode”: normal, “overall”: 0.6}'
    repaired = repair_json_text("{" + raw + "}")

    parsed = json.loads(repaired)
    assert parsed["summary"] == "Strong draft"
    assert parsed["confidence"]["mode"] == "normal"


def test_try_parse_json_repairs_bare_keys_and_trailing_commas():
    raw = (
        'before {summary: "Solid draft", issues: ["Needs more evidence"], '
        'confidence: {mode: normal, overall: 0.4}, safety: {needs_review: false,},} after'
    )
    parsed, err = try_parse_json(raw)
    assert err is None
    assert parsed is not None
    assert parsed["summary"] == "Solid draft"
    assert parsed["confidence"]["mode"] == "normal"


def test_try_parse_json_repairs_partial_missing_closer():
    raw = (
        '{"summary": "Test", "issues": [{"title": "x", "evidence": "y",}], '
        '"safety": {"needs_review": false}'
    )
    parsed, err = try_parse_json(raw)
    assert err is None
    assert parsed is not None
    assert parsed["issues"][0]["title"] == "x"


def test_try_parse_json_repairs_unescaped_quotes_inside_string_values():
    raw = (
        '{"summary": "Test", "issues": [{"title": "Quoted excerpt", '
        '"evidence": "The claim in "Week 2 This exercise demonstrates recursive drawing of triangles using midpoint calculations." is broader than the explanation.", '
        '"severity": "med"}], "safety": {"needs_review": false}}'
    )
    parsed, err = try_parse_json(raw)
    assert err is None
    assert parsed is not None
    assert parsed["issues"][0]["title"] == "Quoted excerpt"
    assert '"Week 2 This exercise demonstrates recursive drawing of triangles using midpoint calculations."' in parsed["issues"][0]["evidence"]


def test_validate_student_payload_reports_missing_summary():
    result = validate_student_payload({"issues": []})
    assert result.valid is False
    assert any("summary" in message for message in result.errors)


def test_validate_professor_payload_reports_wrong_shape():
    result = validate_professor_payload({"rubric_breakdown": "not-a-list"})
    assert result.valid is False
    assert any("rubric_breakdown" in message for message in result.errors)


def test_normalize_student_payload_applies_safe_defaults():
    report = normalize_student_payload(
        {
            "summary": "Working draft",
            "issues": ["No evidence for the main claim."],
            "strengths": [{"title": "Clear structure"}],
        }
    )
    assert report.summary == "Working draft"
    assert report.issues[0].title == "Issue 1"
    assert report.strengths[0].evidence == "Supporting evidence was not provided."
    assert report.architecture_review.overview == "Not assessed."


def test_normalize_professor_payload_coerces_string_rows():
    report = normalize_professor_payload(
        {
            "feedback_explanation": "Evidence is mixed.",
            "rubric_breakdown": ["Argument is underdeveloped."],
            "moderation_notes": ["Check source quality."],
        }
    )
    assert report.rubric_breakdown[0].criterion == "Criterion 1"
    assert report.moderation_notes[0].risk == "Risk 1"
    assert report.feedback_explanation == "Evidence is mixed."
