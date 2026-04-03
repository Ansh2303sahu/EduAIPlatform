from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
import json
import logging
import re
from typing import Any, Literal

from app.config import settings
from app.schemas import (
    StudentReportIn,
    ProfessorReportIn,
    StudentReportOut,
    ProfessorReportOut,
)
from app.security import sanitize_input
from app.prompts import student_prompt, professor_prompt, fix_json_prompt

# Import provider client.  For Ollama we also grab the specific-model caller and
# the fallback model name so JSON-parse failures can be explicitly retried on
# mistral rather than retrying gemma (which already failed).
_generate_with_specific_model = None
_REPAIR_MODEL: str | None = None
_ACTIVE_PROVIDER = settings.effective_provider

if _ACTIVE_PROVIDER == "anthropic":
    from app.anthropic_client import generate_with_fallback
else:
    from app.ollama_client import (
        generate_with_fallback,
        generate_with_specific_model as _generate_with_specific_model,
        OLLAMA_FALLBACK_MODEL as _REPAIR_MODEL,
    )

app = FastAPI(title="llm-service", version="1.2")
logger = logging.getLogger("llm_service")

if settings.llm_provider != _ACTIVE_PROVIDER:
    logger.warning(
        "llm-service forcing local provider selection requested=%s effective=%s",
        settings.llm_provider,
        _ACTIVE_PROVIDER,
    )


class PromptGenerateIn(BaseModel):
    prompt: str = Field(..., min_length=1)
    role: Literal["student", "professor"] = "student"
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    requested_model: str | None = None
    options: dict[str, Any] | None = None


class PromptGenerateOut(BaseModel):
    response: str
    model_used: str


def _check_secret(x_ai_secret: str | None):
    if not settings.service_secret:
        raise HTTPException(status_code=500, detail="LLM_SERVICE_SECRET not set")
    if not x_ai_secret or x_ai_secret != settings.service_secret:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _balanced_extract(text: str) -> str | None:
    """Walk text char-by-char tracking brace depth and string context.
    Returns the first fully balanced JSON object string, or None."""
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
                candidate = text[start : i + 1].strip()
                if ":" in candidate:   # looks like a JSON object, not an empty {}
                    return candidate
                start = None            # empty brace pair — keep scanning
    return None


_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_PYTHON_BOOL_RE = re.compile(r"\b(True|False|None)\b")
_SMART_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})
_BARE_KEY_RE = re.compile(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_ -]*?)(\s*:(?!\s*/))')


def _cheap_repair(text: str) -> str:
    """Apply cheap, safe, regex-only repairs that do not need a model call.

    Handles the most common local-model formatting mistakes:
    - smart / curly quotes
    - Python-style True / False / None literals
    - trailing commas before ] or }
    - bare (unquoted) JSON keys
    """
    # Curly / smart quotes → straight quotes
    text = text.translate(_SMART_QUOTES)
    # Python literals → JSON literals
    text = _PYTHON_BOOL_RE.sub(
        lambda m: {"True": "true", "False": "false", "None": "null"}[m.group()], text
    )
    # Trailing commas before closing delimiter
    text = _TRAILING_COMMA_RE.sub(r"\1", text)
    # Bare keys  e.g.  {summary: "x"}  →  {"summary": "x"}
    text = _BARE_KEY_RE.sub(lambda m: f'{m.group(1)}"{m.group(2).strip()}"{m.group(3)}', text)
    return text


def _extract_json_text(raw: str) -> str:
    """Extract the first balanced JSON object from raw LLM output.

    Strategy (applied in order — first success wins):
    1. Fenced block: find ```(json)? ... ``` anywhere in text, extract inner
       content using the balanced extractor (not a lazy .*? — that breaks for
       deeply nested objects).
    2. Strip a leading fence then run the balanced extractor on the remainder.
    3. Balanced extractor on the raw text (handles prose before/after JSON).
    4. Cheap repair pass then balanced extractor.
    5. Last resort: first { … last } substring (lets downstream repair attempt).
    """
    if not isinstance(raw, str):
        raw = str(raw)

    text = raw.strip()

    # --- Strategy 1: fenced block anywhere in the text ---
    # Find each ``` boundary pair; extract inner text; run balanced extractor.
    fence_re = re.compile(r"```(?:json|JSON)?\s*", re.IGNORECASE)
    close_re = re.compile(r"\s*```")
    pos = 0
    while True:
        m_open = fence_re.search(text, pos)
        if not m_open:
            break
        inner_start = m_open.end()
        m_close = close_re.search(text, inner_start)
        inner_end = m_close.start() if m_close else len(text)
        block = text[inner_start:inner_end].strip()
        candidate = _balanced_extract(block)
        if candidate:
            return candidate
        # also try cheap-repaired version
        candidate = _balanced_extract(_cheap_repair(block))
        if candidate:
            return candidate
        pos = inner_end + 1
        if not m_close:
            break

    # --- Strategy 2: strip leading fence only, then balanced extract ---
    stripped = re.sub(r"^```(?:json|JSON)?\s*", "", text, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```\s*$", "", stripped, flags=re.MULTILINE).strip()
    if stripped != text:
        candidate = _balanced_extract(stripped)
        if candidate:
            return candidate

    # --- Strategy 3: balanced extract on raw text (prose before/after JSON) ---
    candidate = _balanced_extract(text)
    if candidate:
        return candidate

    # --- Strategy 4: cheap repair then balanced extract ---
    repaired = _cheap_repair(text)
    candidate = _balanced_extract(repaired)
    if candidate:
        return candidate

    # --- Strategy 5: last resort — first { to last } ---
    start_obj = text.find("{")
    end_obj = text.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        return text[start_obj : end_obj + 1].strip()

    return text


def _clip(value: str, limit: int = 600) -> str:
    text = value if isinstance(value, str) else str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...<truncated>"


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _truncate(value: Any, limit: int, default: str) -> str:
    if value is None:
        text = default
    elif isinstance(value, str):
        text = value.strip()
    else:
        text = str(value).strip()
    text = text or default
    return text[:limit]


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off", ""}:
            return False
    return default


def _coerce_float(
    value: Any,
    *,
    default: float = 0.0,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, number))


def _coerce_int(
    value: Any,
    *,
    default: int = 1,
    minimum: int = 1,
    maximum: int = 10,
) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        try:
            number = int(float(value))
        except (TypeError, ValueError):
            return default
    return max(minimum, min(maximum, number))


def _normalize_severity(value: Any) -> str:
    normalized = _truncate(value, 20, "low").lower()
    aliases = {
        "low": "low",
        "minor": "low",
        "med": "med",
        "medium": "med",
        "moderate": "med",
        "high": "high",
        "major": "high",
        "severe": "high",
        "critical": "high",
    }
    return aliases.get(normalized, "low")


def _normalize_student_llm_json(obj: Any, *, safe_mode: bool) -> dict[str, Any]:
    data = _as_dict(obj).copy()
    data.pop("rag_meta", None)

    issues = []
    for idx, item in enumerate(_as_list(data.get("issues"))):
        issue = _as_dict(item)
        issues.append(
            {
                "title": _truncate(issue.get("title"), 200, f"Issue {idx + 1}"),
                "evidence": _truncate(issue.get("evidence"), 2000, "Evidence was not provided."),
                "severity": _normalize_severity(issue.get("severity")),
            }
        )

    strengths = []
    for idx, item in enumerate(_as_list(data.get("strengths"))):
        strength = _as_dict(item)
        strengths.append(
            {
                "title": _truncate(strength.get("title"), 200, f"Strength {idx + 1}"),
                "evidence": _truncate(strength.get("evidence"), 2000, "Supporting evidence was not provided."),
            }
        )

    improvement_plan = []
    for idx, item in enumerate(_as_list(data.get("improvement_plan"))):
        action = _as_dict(item)
        improvement_plan.append(
            {
                "action": _truncate(action.get("action"), 300, f"Improvement action {idx + 1}"),
                "why": _truncate(action.get("why"), 800, "The reason was not provided."),
                "how": _truncate(action.get("how"), 800, "The implementation detail was not provided."),
                "priority": _coerce_int(action.get("priority"), default=min(idx + 1, 10)),
            }
        )

    checklist = []
    for idx, item in enumerate(_as_list(data.get("checklist"))):
        check = _as_dict(item)
        checklist.append(
            {
                "item": _truncate(check.get("item"), 200, f"Checklist item {idx + 1}"),
                "done": _coerce_bool(check.get("done"), default=False),
            }
        )

    architecture_review = _as_dict(data.get("architecture_review"))
    implementation_review = _as_dict(data.get("implementation_review"))
    evaluation_review = _as_dict(data.get("evaluation_review"))
    confidence = _as_dict(data.get("confidence"))
    model_agreement = _as_dict(data.get("model_agreement"))
    safety = _as_dict(data.get("safety"))
    normalized_final_confidence = _coerce_float(model_agreement.get("final_confidence"), default=-1.0)
    normalized_overall = _coerce_float(confidence.get("overall"), default=-1.0)

    if normalized_overall < 0.0:
        normalized_overall = (
            normalized_final_confidence
            if normalized_final_confidence >= 0.0
            else (0.35 if safe_mode else 0.75)
        )

    return {
        "summary": _truncate(
            data.get("summary"),
            1200,
            "Automated review generated with limited confidence.",
        ),
        "issues": issues,
        "strengths": strengths,
        "architecture_review": {
            "overview": _truncate(architecture_review.get("overview"), 1200, "Not assessed."),
            "backend": _truncate(architecture_review.get("backend"), 1200, "Not assessed."),
            "frontend": _truncate(architecture_review.get("frontend"), 1200, "Not assessed."),
            "database": _truncate(architecture_review.get("database"), 1200, "Not assessed."),
            "security": _truncate(architecture_review.get("security"), 1200, "Not assessed."),
        },
        "implementation_review": {
            "features_built": [
                _truncate(item, 300, "Feature not specified")
                for item in _as_list(implementation_review.get("features_built"))
                if _truncate(item, 300, "").strip()
            ],
            "technical_quality": _truncate(
                implementation_review.get("technical_quality"), 1200, "Not assessed."
            ),
            "integration_quality": _truncate(
                implementation_review.get("integration_quality"), 1200, "Not assessed."
            ),
        },
        "evaluation_review": {
            "testing_present": _truncate(evaluation_review.get("testing_present"), 1200, "Not assessed."),
            "limitations": _truncate(evaluation_review.get("limitations"), 1200, "Not assessed."),
            "academic_quality": _truncate(evaluation_review.get("academic_quality"), 1200, "Not assessed."),
        },
        "improvement_plan": improvement_plan,
        "checklist": checklist,
        "confidence": {
            "mode": "restricted" if safe_mode else "normal",
            "overall": normalized_overall,
        },
        "model_agreement": {
            "ml_confidence": _coerce_float(model_agreement.get("ml_confidence"), default=0.0),
            "llm_confidence": _coerce_float(model_agreement.get("llm_confidence"), default=0.0),
            "final_confidence": _coerce_float(model_agreement.get("final_confidence"), default=0.0),
        },
        "safety": {
            "needs_review": _coerce_bool(safety.get("needs_review"), default=bool(safe_mode)),
            "reason": _truncate(safety.get("reason"), 2000, ""),
        },
    }


def _normalize_professor_output(obj: Any) -> dict[str, Any]:
    data = _as_dict(obj).copy()
    data.pop("rag_meta", None)

    rubric_breakdown = []
    for idx, item in enumerate(_as_list(data.get("rubric_breakdown"))):
        row = _as_dict(item)
        rubric_breakdown.append(
            {
                "criterion": _truncate(row.get("criterion"), 200, f"Criterion {idx + 1}"),
                "band": _truncate(row.get("band"), 80, "Needs review"),
                "justification": _truncate(
                    row.get("justification"), 1200, "A detailed justification was not provided."
                ),
            }
        )

    moderation_notes = []
    for idx, item in enumerate(_as_list(data.get("moderation_notes"))):
        note = _as_dict(item)
        moderation_notes.append(
            {
                "risk": _truncate(note.get("risk"), 120, f"Risk {idx + 1}"),
                "note": _truncate(note.get("note"), 800, "A moderation note was not provided."),
            }
        )

    if not rubric_breakdown:
        rubric_breakdown = [
            {
                "criterion": "Overall academic quality",
                "band": "Needs review",
                "justification": _truncate(
                    data.get("feedback_explanation"),
                    1200,
                    "Structured rubric evidence was not returned by the model.",
                ),
            }
        ]

    safety = _as_dict(data.get("safety"))

    return {
        "rubric_breakdown": rubric_breakdown,
        "feedback_explanation": _truncate(
            data.get("feedback_explanation"), 1600, "Detailed feedback explanation unavailable."
        ),
        "moderation_notes": moderation_notes,
        "safety": {
            "needs_review": _coerce_bool(safety.get("needs_review"), default=False),
            "reason": _truncate(safety.get("reason"), 2000, ""),
        },
    }


def _student_retrieval_is_weak(payload: StudentReportIn) -> bool:
    label = (payload.retrieval_confidence_label or "").strip().lower()
    safe_review = bool(payload.retrieval_safe_review)
    score = float(payload.retrieval_confidence_score or 0.0)

    return safe_review or label == "low" or score < 0.45


def _professor_retrieval_is_weak(payload: ProfessorReportIn) -> bool:
    label = (payload.retrieval_confidence_label or "").strip().lower()
    safe_review = bool(payload.retrieval_safe_review)
    score = float(payload.retrieval_confidence_score or 0.0)

    return safe_review or label == "low" or score < 0.45


def _student_rag_meta(payload: StudentReportIn) -> dict:
    return {
        "enabled": bool(payload.rag and payload.rag.enabled) or bool(payload.grounding_context),
        "confidence_score": float(payload.retrieval_confidence_score or 0.0),
        "confidence_label": payload.retrieval_confidence_label or "low",
        "safe_review": bool(payload.retrieval_safe_review),
        "citations": [c.model_dump() for c in (payload.grounding_citations or [])],
        "retrieved_chunks": list(payload.grounding_retrieved_chunks or []),
        "trace": payload.retrieval_trace or {},
    }


def _professor_rag_meta(payload: ProfessorReportIn) -> dict:
    return {
        "enabled": bool(payload.rag and payload.rag.enabled) or bool(payload.grounding_context),
        "confidence_score": float(payload.retrieval_confidence_score or 0.0),
        "confidence_label": payload.retrieval_confidence_label or "low",
        "safe_review": bool(payload.retrieval_safe_review),
        "citations": [c.model_dump() for c in (payload.grounding_citations or [])],
        "retrieved_chunks": list(payload.grounding_retrieved_chunks or []),
        "trace": payload.retrieval_trace or {},
    }


def _student_fix_target(payload: StudentReportIn) -> str:
    return "student_project_review" if (payload.analysis_type or "").strip().lower() == "student_project_review" else "student"


def _student_safe_fallback(model_used: str, safe_mode: bool, needs_review: bool) -> dict[str, Any]:
    """Schema-valid restricted student payload returned when both primary and fallback models
    fail JSON parsing after all retries. Prevents a hard 502 from propagating to the caller."""
    return {
        "summary": (
            "Automated feedback could not be generated for this submission. "
            "The language model did not return a parseable response after all retry attempts. "
            "Manual review is required."
        ),
        "issues": [
            {
                "title": "Automated generation failed",
                "evidence": (
                    "The LLM did not produce valid structured output. "
                    "This does not reflect the quality of the submission."
                ),
                "severity": "low",
            }
        ],
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
        "improvement_plan": [
            {
                "action": "Request manual feedback",
                "why": "Automated generation failed.",
                "how": "Contact your instructor for feedback on this submission.",
                "priority": 1,
            }
        ],
        "checklist": [
            {"item": "Await manual feedback from instructor.", "done": False}
        ],
        "confidence": {"mode": "restricted", "overall": 0.0},
        "model_agreement": {
            "ml_confidence": 0.0,
            "llm_confidence": 0.0,
            "final_confidence": 0.0,
        },
        "safety": {
            "needs_review": True,
            "reason": f"LLM generation failed after all retries (model={model_used}). Manual review required.",
        },
    }


def _professor_safe_fallback(model_used: str, needs_review: bool) -> dict[str, Any]:
    """Schema-valid restricted professor payload returned when both models fail JSON parsing."""
    return {
        "rubric_breakdown": [
            {
                "criterion": "Overall assessment",
                "band": "Needs review",
                "justification": (
                    "Automated rubric generation failed. "
                    "The LLM did not return a parseable response after all retry attempts. "
                    "Manual marking is required."
                ),
            }
        ],
        "feedback_explanation": (
            "Automated feedback could not be generated for this submission. "
            "The language model did not produce valid structured output after all retry attempts. "
            "Please review this submission manually."
        ),
        "moderation_notes": [
            {
                "risk": "Automated generation failed",
                "note": f"LLM (model={model_used}) did not return parseable JSON after all retries. Manual review required.",
            }
        ],
        "safety": {
            "needs_review": True,
            "reason": f"LLM generation failed after all retries (model={model_used}). Manual review required.",
        },
    }


async def _call_repair_model(repair_prompt: str) -> dict[str, Any]:
    """Call the repair/fallback model explicitly for JSON fix-up retries.

    When the primary model (gemma3) returns unparseable output, we must NOT
    retry via generate_with_fallback because that would try gemma first again.
    Instead we call the fallback model (mistral) directly using
    generate_with_specific_model.  If that path is unavailable (Anthropic
    provider has no such export), we fall back to generate_with_fallback as a
    safe default.
    """
    if _generate_with_specific_model is not None and _REPAIR_MODEL:
        try:
            return await _generate_with_specific_model(_REPAIR_MODEL, repair_prompt)
        except Exception as exc:
            logger.warning("repair model %s failed: %s — retrying via generate_with_fallback", _REPAIR_MODEL, exc)
    return await generate_with_fallback(repair_prompt)


async def _call_generate_model(
    prompt_payload: dict[str, Any],
    *,
    requested_model: str | None = None,
) -> dict[str, Any]:
    """Honor an exact model request when the backend asks for one.

    Phase 10 already owns the retry order in the backend pipeline, so a
    requested model here means "run this exact model", not "run the internal
    llm-service primary/fallback sequence again".
    """
    if requested_model and _generate_with_specific_model is not None:
        return await _generate_with_specific_model(requested_model, prompt_payload)
    return await generate_with_fallback(prompt_payload)


def _parse_llm_json(raw: str) -> Any:
    """Extract + cheap-repair + parse JSON from raw LLM output.

    Applies _extract_json_text first (balanced extractor, fence handling),
    then a cheap_repair pass on the extracted text before json.loads.
    Raises json.JSONDecodeError if neither step produces parseable JSON.
    """
    extracted = _extract_json_text(raw)
    try:
        return json.loads(extracted)
    except json.JSONDecodeError:
        pass
    repaired = _cheap_repair(extracted)
    return json.loads(repaired)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/llm/generate", response_model=PromptGenerateOut)
async def generate_prompt(
    payload: PromptGenerateIn,
    x_ai_secret: str | None = Header(default=None),
):
    """
    Protected prompt-through endpoint used by the Phase 10 LangChain factory.

    This keeps Ollama behind the existing llm-service boundary instead of
    allowing the backend to call Ollama directly.
    """
    _check_secret(x_ai_secret)

    # Append a compact JSON-only enforcer so the Phase 10 LangChain prompts
    # always arrive at Ollama with an unambiguous output constraint.
    enforced_prompt = (
        payload.prompt.rstrip()
        + "\n\n"
        + "CRITICAL: Your response must be a single valid JSON object. "
        + "Start with { and end with }. "
        + "No markdown fences, no prose before or after the JSON. "
        + "All keys must use double quotes. No trailing commas."
    )

    request_payload: dict[str, Any] = {
        "prompt": enforced_prompt,
        "stream": False,
        "format": "json",
    }
    options = dict(payload.options or {})
    if payload.temperature is not None:
        options["temperature"] = float(payload.temperature)
    if options:
        request_payload["options"] = options

    gen = await _call_generate_model(
        request_payload,
        requested_model=(payload.requested_model or "").strip() or None,
    )
    model_used = str(gen.get("model_used") or payload.requested_model or "")
    response_text = str(gen.get("response") or "")

    return JSONResponse(
        content={"response": response_text, "model_used": model_used},
        headers={"x-llm-model-used": model_used},
    )


@app.post("/llm/student/report", response_model=StudentReportOut)
async def student_report(
    payload: StudentReportIn,
    x_ai_secret: str | None = Header(default=None),
):
    _check_secret(x_ai_secret)

    combined = "\n".join(
        [
            payload.ingestion.text_content,
            payload.ingestion.ocr_text,
            payload.ingestion.audio_transcript,
        ]
    )
    _, injected, inj_reason = sanitize_input(combined, settings.max_input_chars)

    retrieval_weak = _student_retrieval_is_weak(payload)
    requested_mode = (payload.mode or "").strip().lower()

    safe_mode = (
        requested_mode == "restricted"
        or
        injected
        or (
            payload.ml.quality_band == "low"
            and payload.ml.confidence_0_to_4 <= 1
        )
    )
    needs_review = (
        requested_mode == "restricted"
        or injected
        or retrieval_weak
        or (payload.ml.confidence_0_to_4 <= 1)
    )

    logger.info(
        "student_report start requested_mode=%s analysis_type=%s safe_mode=%s needs_review=%s injected=%s retrieval_weak=%s quality_band=%s confidence_0_to_4=%s query=%r top_k=%s retrieved_chunks=%s citations=%s",
        requested_mode or "unset",
        (payload.analysis_type or "").strip().lower() or "unset",
        safe_mode,
        needs_review,
        injected,
        retrieval_weak,
        payload.ml.quality_band,
        payload.ml.confidence_0_to_4,
        payload.query or "",
        payload.top_k,
        len(payload.grounding_retrieved_chunks or []),
        len(payload.grounding_citations or []),
    )

    gen = await generate_with_fallback(student_prompt(payload, safe_mode=safe_mode))
    model_used = gen["model_used"]
    raw = gen["response"]
    for attempt in range(settings.max_retries + 1):
        try:
            # _parse_llm_json = extract (balanced + fence-aware) + cheap_repair + json.loads
            parsed = _parse_llm_json(raw)
            obj = _normalize_student_llm_json(parsed, safe_mode=safe_mode)
            out = StudentReportOut.model_validate(obj)

            response_body = out.model_dump()
            response_body.setdefault("safety", {})
            response_body["safety"]["needs_review"] = bool(
                response_body["safety"].get("needs_review") or needs_review
            )
            if response_body["safety"]["needs_review"] and not str(
                response_body["safety"].get("reason") or ""
            ).strip():
                reasons: list[str] = []
                if injected:
                    reasons.append("potential prompt-injection content was detected")
                if retrieval_weak:
                    reasons.append("grounding evidence was limited")
                if payload.ml.confidence_0_to_4 <= 1:
                    reasons.append("ML confidence was low")
                if requested_mode == "restricted":
                    reasons.append("restricted mode was requested")
                if reasons:
                    response_body["safety"]["reason"] = (
                        "Manual review recommended because " + ", ".join(reasons) + "."
                    )
            response_body["rag_meta"] = _student_rag_meta(payload)

            return JSONResponse(
                content=response_body,
                headers={
                    "x-llm-model-used": model_used,
                    "x-rag-enabled": str(response_body["rag_meta"]["enabled"]).lower(),
                    "x-rag-confidence-label": str(
                        response_body["rag_meta"]["confidence_label"]
                    ),
                },
            )

        except json.JSONDecodeError as exc:
            logger.warning(
                "student_report json_decode_error attempt=%s/%s model=%s safe_mode=%s injected=%s retrieval_weak=%s error=%s raw=%r",
                attempt + 1,
                settings.max_retries + 1,
                model_used,
                safe_mode,
                injected,
                retrieval_weak,
                exc,
                _clip(raw),
            )
            if attempt >= settings.max_retries:
                logger.error(
                    "student_report exhausted retries — returning safe fallback model=%s safe_mode=%s injected=%s retrieval_weak=%s reason=%s raw=%r",
                    model_used,
                    safe_mode,
                    injected,
                    retrieval_weak,
                    inj_reason,
                    _clip(raw),
                )
                fallback_body = _student_safe_fallback(model_used, safe_mode, needs_review)
                fallback_body["rag_meta"] = _student_rag_meta(payload)
                return JSONResponse(
                    content=fallback_body,
                    status_code=200,
                    headers={"x-llm-model-used": model_used, "x-llm-fallback": "true"},
                )

            # JSON failure = model failure: retry explicitly on the repair/fallback model
            gen = await _call_repair_model(
                fix_json_prompt(
                    raw,
                    _student_fix_target(payload),
                    forced_confidence_mode="restricted" if safe_mode else "normal",
                    forced_needs_review=needs_review,
                )
            )
            model_used = gen["model_used"]
            raw = gen["response"]
        except ValidationError as exc:
            logger.warning(
                "student_report validation_error attempt=%s/%s model=%s safe_mode=%s injected=%s retrieval_weak=%s errors=%s raw=%r",
                attempt + 1,
                settings.max_retries + 1,
                model_used,
                safe_mode,
                injected,
                retrieval_weak,
                exc.errors(),
                _clip(raw),
            )
            if attempt >= settings.max_retries:
                logger.error(
                    "student_report exhausted retries — returning safe fallback model=%s safe_mode=%s injected=%s retrieval_weak=%s reason=%s raw=%r",
                    model_used,
                    safe_mode,
                    injected,
                    retrieval_weak,
                    inj_reason,
                    _clip(raw),
                )
                fallback_body = _student_safe_fallback(model_used, safe_mode, needs_review)
                fallback_body["rag_meta"] = _student_rag_meta(payload)
                return JSONResponse(
                    content=fallback_body,
                    status_code=200,
                    headers={"x-llm-model-used": model_used, "x-llm-fallback": "true"},
                )

            # Validation failure after parsing = also route to the repair/fallback model
            gen = await _call_repair_model(
                fix_json_prompt(
                    raw,
                    _student_fix_target(payload),
                    forced_confidence_mode="restricted" if safe_mode else "normal",
                    forced_needs_review=needs_review,
                )
            )
            model_used = gen["model_used"]
            raw = gen["response"]


@app.post("/llm/professor/report", response_model=ProfessorReportOut)
async def professor_report(
    payload: ProfessorReportIn,
    x_ai_secret: str | None = Header(default=None),
):
    _check_secret(x_ai_secret)

    combined = "\n".join(
        [
            payload.ingestion.text_content,
            payload.ingestion.ocr_text,
            payload.ingestion.audio_transcript,
        ]
    )
    _, injected, inj_reason = sanitize_input(combined, settings.max_input_chars)

    retrieval_weak = _professor_retrieval_is_weak(payload)
    requested_mode = (payload.mode or "").strip().lower()

    needs_review = (
        requested_mode == "restricted"
        or
        injected
        or (payload.ml.moderation_consistency == "low")
        or retrieval_weak
    )

    logger.info(
        "professor_report start requested_mode=%s analysis_type=%s needs_review=%s injected=%s retrieval_weak=%s moderation_consistency=%s query=%r top_k=%s retrieved_chunks=%s citations=%s",
        requested_mode or "unset",
        (payload.analysis_type or "").strip().lower() or "unset",
        needs_review,
        injected,
        retrieval_weak,
        payload.ml.moderation_consistency,
        payload.query or "",
        payload.top_k,
        len(payload.grounding_retrieved_chunks or []),
        len(payload.grounding_citations or []),
    )

    gen = await generate_with_fallback(professor_prompt(payload, needs_review=needs_review))
    model_used = gen["model_used"]
    raw = gen["response"]

    for attempt in range(settings.max_retries + 1):
        try:
            # _parse_llm_json = extract (balanced + fence-aware) + cheap_repair + json.loads
            obj = _normalize_professor_output(_parse_llm_json(raw))
            out = ProfessorReportOut.model_validate(obj)

            response_body = out.model_dump()
            response_body["rag_meta"] = _professor_rag_meta(payload)

            return JSONResponse(
                content=response_body,
                headers={
                    "x-llm-model-used": model_used,
                    "x-rag-enabled": str(response_body["rag_meta"]["enabled"]).lower(),
                    "x-rag-confidence-label": str(
                        response_body["rag_meta"]["confidence_label"]
                    ),
                },
            )

        except json.JSONDecodeError as exc:
            logger.warning(
                "professor_report json_decode_error attempt=%s/%s model=%s needs_review=%s injected=%s retrieval_weak=%s error=%s raw=%r",
                attempt + 1,
                settings.max_retries + 1,
                model_used,
                needs_review,
                injected,
                retrieval_weak,
                exc,
                _clip(raw),
            )
            if attempt >= settings.max_retries:
                logger.error(
                    "professor_report exhausted retries — returning safe fallback model=%s needs_review=%s injected=%s retrieval_weak=%s reason=%s raw=%r",
                    model_used,
                    needs_review,
                    injected,
                    retrieval_weak,
                    inj_reason,
                    _clip(raw),
                )
                fallback_body = _professor_safe_fallback(model_used, needs_review)
                fallback_body["rag_meta"] = _professor_rag_meta(payload)
                return JSONResponse(
                    content=fallback_body,
                    status_code=200,
                    headers={"x-llm-model-used": model_used, "x-llm-fallback": "true"},
                )

            # JSON failure = model failure: retry explicitly on the repair/fallback model
            gen = await _call_repair_model(
                fix_json_prompt(
                    raw,
                    "professor",
                    forced_needs_review=needs_review,
                )
            )
            model_used = gen["model_used"]
            raw = gen["response"]
        except ValidationError as exc:
            logger.warning(
                "professor_report validation_error attempt=%s/%s model=%s needs_review=%s injected=%s retrieval_weak=%s errors=%s raw=%r",
                attempt + 1,
                settings.max_retries + 1,
                model_used,
                needs_review,
                injected,
                retrieval_weak,
                exc.errors(),
                _clip(raw),
            )
            if attempt >= settings.max_retries:
                logger.error(
                    "professor_report exhausted retries — returning safe fallback model=%s needs_review=%s injected=%s retrieval_weak=%s reason=%s raw=%r",
                    model_used,
                    needs_review,
                    injected,
                    retrieval_weak,
                    inj_reason,
                    _clip(raw),
                )
                fallback_body = _professor_safe_fallback(model_used, needs_review)
                fallback_body["rag_meta"] = _professor_rag_meta(payload)
                return JSONResponse(
                    content=fallback_body,
                    status_code=200,
                    headers={"x-llm-model-used": model_used, "x-llm-fallback": "true"},
                )

            # Validation failure after parsing = also route to the repair/fallback model
            gen = await _call_repair_model(
                fix_json_prompt(
                    raw,
                    "professor",
                    forced_needs_review=needs_review,
                )
            )
            model_used = gen["model_used"]
            raw = gen["response"]
