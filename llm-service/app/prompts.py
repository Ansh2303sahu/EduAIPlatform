import json
from .schemas import StudentReportIn, ProfessorReportIn

_JSON_RULES = """
Return ONLY valid JSON.
Do not include markdown.
Do not include ```json fences.
Do not include commentary before or after the JSON.
Use double quotes for all keys and string values.
Do not add extra keys.
Do not omit required keys.
""".strip()


def _compact_ingestion(ing) -> str:
    obj = {
        "text_content": (ing.text_content or "")[:12000],
        "ocr_text": (ing.ocr_text or "")[:8000],
        "audio_transcript": (ing.audio_transcript or "")[:8000],
        "tables_json": ing.tables_json or {},
    }
    return json.dumps(obj, ensure_ascii=False)


def student_prompt(payload: StudentReportIn, safe_mode: bool) -> str:
    schema = {
        "summary": "string",
        "issues": [
            {
                "title": "string",
                "evidence": "string",
                "severity": "low"
            }
        ],
        "improvement_plan": [
            {
                "action": "string",
                "why": "string",
                "how": "string",
                "priority": 1
            }
        ],
        "checklist": [
            {
                "item": "string",
                "done": False
            }
        ],
        "model_agreement": {
            "ml_confidence": 0.0,
            "llm_confidence": 0.0,
            "final_confidence": 0.0
        },
        "safety": {
            "needs_review": False,
            "reason": "string"
        }
    }

    mode = (
        """
SAFE MODE:
- Be cautious.
- Do not grade.
- Keep claims conservative.
- Set safety.needs_review to true.
- Explain the reason briefly in safety.reason.
"""
        if safe_mode
        else
        """
NORMAL MODE:
- Be specific and constructive.
- Do not grade.
- Set safety.needs_review to false unless there is a strong reason.
"""
    ).strip()

    return f"""
You are a STUDENT feedback engine for academic submissions.

{_JSON_RULES}

You MUST return exactly one JSON object matching this schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}

Extra constraints:
- "summary" must be a plain string.
- "issues" must be an array of objects with keys: title, evidence, severity.
- "severity" must be exactly one of: "low", "med", "high".
- "improvement_plan" must be an array of objects with keys: action, why, how, priority.
- "priority" must be an integer from 1 to 10.
- "checklist" must be an array of objects with keys: item, done.
- "done" must be a boolean.
- "model_agreement" values must be numbers between 0.0 and 1.0.
- "safety" must contain keys: needs_review, reason.
- No extra keys anywhere.

{mode}

ML signals:
{json.dumps({
    "feedback_category": payload.ml.feedback_category,
    "quality_band": payload.ml.quality_band,
    "confidence_0_to_4": payload.ml.confidence_0_to_4
}, ensure_ascii=False, indent=2)}

Submission content:
{_compact_ingestion(payload.ingestion)}
""".strip()


def professor_prompt(payload: ProfessorReportIn, needs_review: bool) -> str:
    schema = {
        "rubric_breakdown": [
            {
                "criterion": "string",
                "band": "string",
                "justification": "string"
            }
        ],
        "feedback_explanation": "string",
        "moderation_notes": [
            {
                "risk": "string",
                "note": "string"
            }
        ],
        "safety": {
            "needs_review": False,
            "reason": "string"
        }
    }

    mode = (
        """
REVIEW MODE:
- Focus on moderation risks and uncertainty.
- Set safety.needs_review to true.
- Explain the reason briefly in safety.reason.
"""
        if needs_review
        else
        """
NORMAL MODE:
- Be rubric-focused and moderation-safe.
- Set safety.needs_review to false unless there is a strong reason.
"""
    ).strip()

    return f"""
You are a PROFESSOR rubric engine for academic submissions.

{_JSON_RULES}

You MUST return exactly one JSON object matching this schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}

Extra constraints:
- "rubric_breakdown" must be an array of objects with keys: criterion, band, justification.
- "feedback_explanation" must be a plain string.
- "moderation_notes" must be an array of objects with keys: risk, note.
- "safety" must contain keys: needs_review, reason.
- No extra keys anywhere.

{mode}

ML signals:
{json.dumps({
    "rubric_band": payload.ml.rubric_band,
    "argument_depth": payload.ml.argument_depth,
    "moderation_consistency": payload.ml.moderation_consistency
}, ensure_ascii=False, indent=2)}

Submission content:
{_compact_ingestion(payload.ingestion)}
""".strip()


def fix_json_prompt(bad_output: str, target: str) -> str:
    if target == "student":
        schema = {
            "summary": "string",
            "issues": [
                {
                    "title": "string",
                    "evidence": "string",
                    "severity": "low"
                }
            ],
            "improvement_plan": [
                {
                    "action": "string",
                    "why": "string",
                    "how": "string",
                    "priority": 1
                }
            ],
            "checklist": [
                {
                    "item": "string",
                    "done": False
                }
            ],
            "model_agreement": {
                "ml_confidence": 0.0,
                "llm_confidence": 0.0,
                "final_confidence": 0.0
            },
            "safety": {
                "needs_review": False,
                "reason": "string"
            }
        }
    else:
        schema = {
            "rubric_breakdown": [
                {
                    "criterion": "string",
                    "band": "string",
                    "justification": "string"
                }
            ],
            "feedback_explanation": "string",
            "moderation_notes": [
                {
                    "risk": "string",
                    "note": "string"
                }
            ],
            "safety": {
                "needs_review": False,
                "reason": "string"
            }
        }

    return f"""
Repair the following model output into valid JSON.

{_JSON_RULES}

Target:
{target}

Return exactly one JSON object matching this schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}

Important:
- Preserve the meaning of the original content as much as possible.
- If a field is missing, fill it with a sensible minimal value.
- Do not add markdown.
- Do not add explanation text.

Broken output:
{bad_output}
""".strip()