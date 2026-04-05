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
from app.prompts import student_prompt, professor_prompt, fix_json_prompt, student_feedback_style

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


def _active_primary_model() -> str:
    if _ACTIVE_PROVIDER == "anthropic":
        return str(settings.anthropic_primary_model or "")
    return str(settings.primary_model or "")


def _active_fallback_model() -> str:
    if _ACTIVE_PROVIDER == "anthropic":
        return str(settings.anthropic_fallback_model or "")
    return str(settings.fallback_model or "")


def _llm_response_headers(
    model_used: str,
    *,
    used_fallback: bool = False,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    headers = {
        "x-llm-model-used": str(model_used or ""),
        "x-llm-primary-model": _active_primary_model(),
        "x-llm-fallback-model": _active_fallback_model(),
    }
    if used_fallback:
        headers["x-llm-fallback"] = "true"
    if extra:
        headers.update(extra)
    return headers


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


def _first_present(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data and data.get(key) is not None:
            return data.get(key)
    return None


def _value_has_meaningful_content(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (int, float, bool)):
        return True
    if isinstance(value, list):
        return any(_value_has_meaningful_content(item) for item in value)
    if isinstance(value, dict):
        return any(_value_has_meaningful_content(item) for item in value.values())
    return bool(str(value).strip())


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


_STUDENT_PLACEHOLDER_SUMMARY = "Automated review generated with limited confidence."
_NOT_ASSESSED_PLACEHOLDER = "Not assessed."
_LOW_CONTENT_QUALITY_REASON = "low_content_quality"
_STUDENT_TOP_LEVEL_KEYS = {
    "summary",
    "issues",
    "strengths",
    "architecture_review",
    "implementation_review",
    "evaluation_review",
    "improvement_plan",
    "checklist",
    "confidence",
    "model_agreement",
    "safety",
    "rag_meta",
}


def _normalized_marker_text(value: Any) -> str:
    text = value if isinstance(value, str) else str(value or "")
    return re.sub(r"\s+", " ", text.strip()).casefold().rstrip(".!?")


def _is_student_placeholder_summary(value: Any) -> bool:
    normalized = _normalized_marker_text(value)
    placeholder = _normalized_marker_text(_STUDENT_PLACEHOLDER_SUMMARY)
    return normalized == placeholder or (
        "automated review generated" in normalized
        and "limited confidence" in normalized
    )


def _is_not_assessed_placeholder(value: Any) -> bool:
    return _normalized_marker_text(value) == _normalized_marker_text(_NOT_ASSESSED_PLACEHOLDER)


def _student_report_low_content_quality(report: Any) -> bool:
    data = _as_dict(report)
    if not data:
        return False

    summary_is_placeholder = _is_student_placeholder_summary(data.get("summary"))
    lists_are_empty = all(
        len(_as_list(data.get(key))) == 0
        for key in ("issues", "strengths", "improvement_plan", "checklist")
    )

    architecture_review = _as_dict(data.get("architecture_review"))
    implementation_review = _as_dict(data.get("implementation_review"))
    evaluation_review = _as_dict(data.get("evaluation_review"))

    architecture_is_placeholder = all(
        _is_not_assessed_placeholder(architecture_review.get(key))
        for key in ("overview", "backend", "frontend", "database", "security")
    )
    implementation_is_placeholder = (
        len(_as_list(implementation_review.get("features_built"))) == 0
        and _is_not_assessed_placeholder(implementation_review.get("technical_quality"))
        and _is_not_assessed_placeholder(implementation_review.get("integration_quality"))
    )
    evaluation_is_placeholder = all(
        _is_not_assessed_placeholder(evaluation_review.get(key))
        for key in ("testing_present", "limitations", "academic_quality")
    )

    return (
        summary_is_placeholder
        and lists_are_empty
        and architecture_is_placeholder
        and implementation_is_placeholder
        and evaluation_is_placeholder
    )


def _student_normalization_removed_content(parsed: Any, normalized: dict[str, Any]) -> bool:
    if not _student_report_low_content_quality(normalized):
        return False

    parsed_dict = _as_dict(parsed)
    if not parsed_dict:
        return False

    extra_keys = sorted(set(parsed_dict.keys()) - _STUDENT_TOP_LEVEL_KEYS)
    return any(_value_has_meaningful_content(parsed_dict.get(key)) for key in extra_keys)


def _student_debug_metrics(report: Any) -> dict[str, Any]:
    data = _as_dict(report)
    if not data:
        return {
            "summary_placeholder": False,
            "issues": 0,
            "strengths": 0,
            "improvements": 0,
            "checklist": 0,
            "low_content_quality": False,
        }
    return {
        "summary_placeholder": _is_student_placeholder_summary(data.get("summary")),
        "issues": len(_as_list(data.get("issues"))),
        "strengths": len(_as_list(data.get("strengths"))),
        "improvements": len(_as_list(data.get("improvement_plan"))),
        "checklist": len(_as_list(data.get("checklist"))),
        "low_content_quality": _student_report_low_content_quality(data),
    }


_PROJECT_TECH_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("Django", ("django",)),
    ("React", ("react", "next.js")),
    ("FastAPI", ("fastapi",)),
    ("Flask", ("flask",)),
    ("Chart.js", ("chart.js", "price chart")),
    ("Gemini AI", ("gemini", "ai assistant")),
    ("PostgreSQL", ("postgresql", "postgres")),
    ("SQLite", ("sqlite",)),
]

_PROJECT_FEATURE_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("BUY and SELL transaction management", ("buy and sell", "transaction management", "sell-quantity validation")),
    ("Portfolio analytics dashboard", ("portfolio analytics", "dashboard", "portfolio value")),
    ("Historical market data processing", ("historical price", "nasdaq 100", "market data")),
    ("Chart-based visualisation", ("chart.js", "price chart", "moving average")),
    ("User authentication", ("authentication", "login", "django's built-in framework")),
    ("AI assistant", ("ai assistant", "gemini", "ai-powered insights")),
    ("Scalability evaluation", ("scalability experiment", "render time", "query count")),
    ("Automated test evidence", ("unit test", "acceptance test", "functional acceptance test")),
]


def _sentence_split(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return []
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned)
        if sentence.strip()
    ]


def _unique_preserve(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        normalized = _normalized_marker_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(value)
    return output


def _extract_abstract_sentences(text: str) -> list[str]:
    source = text or ""
    upper = source.upper()
    start = upper.find("ABSTRACT")
    if start >= 0:
        segment = source[start + len("ABSTRACT") : start + len("ABSTRACT") + 4000]
    else:
        segment = source[:3000]
    return _sentence_split(segment)[:6]


def _label_hits(text: str, patterns: list[tuple[str, tuple[str, ...]]], *, limit: int = 6) -> list[str]:
    lowered = text.casefold()
    hits: list[str] = []
    for label, options in patterns:
        if any(option.casefold() in lowered for option in options):
            hits.append(label)
        if len(hits) >= limit:
            break
    return hits


def _has_any(text: str, *terms: str) -> bool:
    lowered = text.casefold()
    return any(term.casefold() in lowered for term in terms)


def _student_heuristic_report_from_payload(
    payload: StudentReportIn,
    *,
    safe_mode: bool,
    needs_review: bool,
) -> dict[str, Any] | None:
    text = str(payload.ingestion.text_content or "").strip()
    feedback_style = student_feedback_style(payload)
    if len(text) < (900 if feedback_style == "code" else 450):
        return None

    abstract_sentences = _extract_abstract_sentences(text)

    if feedback_style != "code":
        thesis_present = _has_any(text, "thesis", "this essay argues", "this paper argues", "argument", "research question")
        evidence_present = _has_any(text, "evidence", "study", "source", "citation", "reference", "literature")
        analysis_present = _has_any(text, "analysis", "critical", "evaluate", "however", "whereas", "compare", "limitation")
        structure_present = _has_any(text, "introduction", "conclusion", "paragraph", "section")
        referencing_present = _has_any(text, "references", "bibliography", "apa", "harvard", "citation")

        if len(abstract_sentences) < 2 and sum(
            1 for present in (thesis_present, evidence_present, analysis_present, structure_present) if present
        ) < 2:
            return None

        strengths: list[dict[str, Any]] = []
        if thesis_present:
            strengths.append(
                {
                    "title": "The submission has a discernible central focus",
                    "evidence": _truncate(
                        "The text signals a thesis, argument, or clear task focus, which gives the discussion a visible academic direction instead of feeling like disconnected notes.",
                        2000,
                        "The submission shows a visible central focus.",
                    ),
                }
            )
        if evidence_present:
            strengths.append(
                {
                    "title": "The discussion appears to use supporting material",
                    "evidence": _truncate(
                        "The submission refers to evidence, sources, studies, or cited material, which suggests the argument is at least partly grounded rather than purely asserted.",
                        2000,
                        "The submission appears to draw on supporting material.",
                    ),
                }
            )
        if structure_present and len(strengths) < 3:
            strengths.append(
                {
                    "title": "The writing shows some structural signposting",
                    "evidence": _truncate(
                        "Signals such as an introduction, conclusion, sections, or paragraph-level organisation suggest the work is trying to guide the reader through a coherent line of reasoning.",
                        2000,
                        "The writing shows some structural signposting.",
                    ),
                }
            )

        issues: list[dict[str, Any]] = []
        if not thesis_present:
            issues.append(
                {
                    "title": "The central argument needs sharper definition",
                    "evidence": _truncate(
                        "The submission discusses the topic, but the controlling thesis or exact position is not consistently explicit. That weakens academic direction because the reader cannot easily judge how each paragraph advances the main claim.",
                        2000,
                        "The central argument is not consistently explicit.",
                    ),
                    "severity": "high",
                }
            )
        if not analysis_present:
            issues.append(
                {
                    "title": "The discussion is more descriptive than critical",
                    "evidence": _truncate(
                        "The available text suggests explanation or topic coverage, but less explicit comparison, challenge, limitation analysis, or evaluative judgement. This matters academically because higher-quality essays need interpretation and critical positioning, not only description.",
                        2000,
                        "The discussion appears more descriptive than analytical.",
                    ),
                    "severity": "med",
                }
            )
        if not evidence_present:
            issues.append(
                {
                    "title": "Key claims need stronger supporting evidence",
                    "evidence": _truncate(
                        "The argument does not consistently show clear supporting sources, examples, or referenced material. That matters because unsupported claims reduce credibility and make the academic case less defensible.",
                        2000,
                        "Key claims need clearer supporting evidence.",
                    ),
                    "severity": "high",
                }
            )
        if not referencing_present and len(issues) < 3:
            issues.append(
                {
                    "title": "Referencing practice is not clearly evidenced",
                    "evidence": _truncate(
                        "The submission gives limited signs of a consistent referencing system or explicit citation trail. That matters because even strong ideas lose academic reliability when attribution is unclear.",
                        2000,
                        "Referencing practice is not clearly evidenced.",
                    ),
                    "severity": "med",
                }
            )
        if not structure_present and len(issues) < 3:
            issues.append(
                {
                    "title": "Structural signposting could be stronger",
                    "evidence": _truncate(
                        "The progression between parts of the argument is not always clearly signalled. This matters because weak structure makes it harder for the reader to follow the logic and evaluate the submission fairly.",
                        2000,
                        "Structural signposting could be stronger.",
                    ),
                    "severity": "med",
                }
            )
        if not issues:
            issues.append(
                {
                    "title": "Comparative analysis could be pushed further",
                    "evidence": _truncate(
                        "The submission appears coherent and evidence-aware, but stronger academic writing would still make the comparative reasoning and analytical judgments more explicit. That matters because higher-band work usually explains not just what the evidence says, but why one interpretation is more convincing than another.",
                        2000,
                        "Comparative analysis could be pushed further.",
                    ),
                    "severity": "med",
                }
            )
        if len(issues) == 1:
            issues.append(
                {
                    "title": "Source integration can be made more explicit",
                    "evidence": _truncate(
                        "Even where the submission appears to use evidence, the link between source material and the exact claim being supported can often be made clearer. This matters because explicit source integration improves academic credibility and makes the line of reasoning easier to defend.",
                        2000,
                        "Source integration can be made more explicit.",
                    ),
                    "severity": "med",
                }
            )
        issues = issues[:3]

        improvements = [
            {
                "action": "Clarify the thesis and paragraph purpose earlier",
                "why": "A sharper controlling argument makes the whole submission easier to evaluate and strengthens coherence.",
                "how": "State the core claim explicitly in the introduction and add topic sentences that show how each major paragraph advances it.",
                "priority": 1,
            },
            {
                "action": "Increase critical comparison and interpretation",
                "why": "Higher-quality academic writing needs analysis, not only topic coverage or description.",
                "how": "Add short comparative judgments, limitations, counterpoints, or implications after each major piece of evidence.",
                "priority": 2,
            },
            {
                "action": "Tighten evidence and referencing discipline",
                "why": "Clear attribution improves academic credibility and makes claims more defensible.",
                "how": "Check that each major claim is supported by a source, and keep in-text citations and the reference list consistent.",
                "priority": 3,
            },
        ]

        checklist = [
            {"item": "State the core thesis clearly in the introduction", "done": False},
            {"item": "Add critical comparison after major evidence points", "done": False},
            {"item": "Check citations and references for consistency", "done": False},
        ]

        summary_sentences = _unique_preserve(
            [
                _truncate(abstract_sentences[0] if abstract_sentences else "", 320, ""),
                _truncate(abstract_sentences[1] if len(abstract_sentences) > 1 else "", 320, ""),
                _truncate(
                    "The submission appears to pursue a recognisable academic argument rather than a loose collection of observations.",
                    320,
                    "",
                ),
                _truncate(
                    "The main improvements are stronger thesis control, deeper critical analysis, and clearer evidence handling.",
                    320,
                    "",
                ),
            ]
        )
        summary = " ".join(summary_sentences[:4]).strip() or "The submission provides enough evidence for a focused academic review."

        ml_confidence = max(0.0, min(1.0, float(payload.ml.confidence_0_to_4) / 4.0))
        heuristic_confidence = 0.46
        heuristic_confidence += 0.06 if thesis_present else 0.0
        heuristic_confidence += 0.06 if evidence_present else 0.0
        heuristic_confidence += 0.06 if analysis_present else 0.0
        heuristic_confidence += 0.04 if structure_present else 0.0
        heuristic_confidence += 0.04 if referencing_present else 0.0
        heuristic_confidence = max(0.42, min(0.72, heuristic_confidence))
        overall_confidence = min(0.72, round((heuristic_confidence + ml_confidence) / 2.0, 2))

        return {
            "summary": _truncate(summary, 1200, "The submission provides enough evidence for a focused academic review."),
            "issues": issues,
            "strengths": strengths[:3],
            "architecture_review": {
                "overview": "The written submission shows a discernible academic structure, though its argument could be signposted more sharply.",
                "backend": "The core supporting layer is the progression of claims and evidence rather than software implementation.",
                "frontend": "Clarity, paragraph flow, and presentation shape how effectively the argument reaches the reader.",
                "database": "The source base functions as the submission's evidence foundation and should be used more consistently where needed.",
                "security": "Academic integrity depends on cautious claims, accurate attribution, and avoiding unsupported assertions.",
            },
            "implementation_review": {
                "features_built": [item for item, present in [
                    ("Explicit thesis or task focus", thesis_present),
                    ("Source-based discussion", evidence_present),
                    ("Critical comparison or evaluation", analysis_present),
                    ("Structured sections or paragraph flow", structure_present),
                ] if present][:4],
                "technical_quality": "The submission is readable and academically oriented, but its strongest gains now come from sharper analysis and evidence control.",
                "integration_quality": "The overall quality depends on how well claims, evidence, structure, and conclusions are linked across the discussion.",
            },
            "evaluation_review": {
                "testing_present": "Direct empirical evaluation may not apply here, but the strength of the work still depends on how rigorously claims are supported and examined.",
                "limitations": "The current draft would benefit from clearer thesis framing, deeper analysis, and more explicit source support in weaker sections.",
                "academic_quality": "The work shows enough substance for targeted feedback, but the academic standard would rise with clearer critical positioning and evidence discipline.",
            },
            "improvement_plan": improvements[:3],
            "checklist": checklist[:3],
            "confidence": {
                "mode": "restricted" if safe_mode else "normal",
                "overall": 0.35 if safe_mode else overall_confidence,
            },
            "model_agreement": {
                "ml_confidence": ml_confidence,
                "llm_confidence": 0.35 if safe_mode else heuristic_confidence,
                "final_confidence": 0.35 if safe_mode else overall_confidence,
            },
            "safety": {
                "needs_review": bool(needs_review),
                "reason": "",
            },
        }

    features = _label_hits(text, _PROJECT_FEATURE_PATTERNS)
    tech_stack = _label_hits(text, _PROJECT_TECH_PATTERNS)
    if len(abstract_sentences) < 2 and len(features) < 2:
        return None

    testing_present = _has_any(
        text,
        "test",
        "testing",
        "unit test",
        "acceptance test",
        "evaluated",
        "evaluation",
    )
    security_present = _has_any(
        text,
        "authentication",
        "authorization",
        "security",
        "validation",
        "login",
    )
    scalability_present = _has_any(
        text,
        "scalability",
        "performance",
        "query count",
        "render time",
        "load",
    )
    rationale_present = _has_any(
        text,
        "trade-off",
        "tradeoff",
        "rationale",
        "justif",
        "because",
        "decision",
    )
    ai_present = _has_any(text, "gemini", "ai assistant", "ai-powered insights", "llm")

    strengths: list[dict[str, Any]] = []
    if features:
        strengths.append(
            {
                "title": "Implemented feature scope is concrete",
                "evidence": _truncate(
                    f"The submission explicitly describes {', '.join(features[:4])}, which gives clear evidence of a real implemented system rather than a speculative proposal.",
                    2000,
                    "The submission gives concrete evidence of implemented system features.",
                ),
            }
        )
    if testing_present:
        strengths.append(
            {
                "title": "Evaluation evidence goes beyond a simple feature list",
                "evidence": _truncate(
                    "The report discusses testing, evaluation, or performance evidence, which strengthens confidence that the project was exercised rather than only described at a high level.",
                    2000,
                    "The report provides some evaluation evidence.",
                ),
            }
        )
    if ai_present and len(strengths) < 3:
        strengths.append(
            {
                "title": "The project combines domain features with advanced functionality",
                "evidence": _truncate(
                    "The submission links core application workflows with analytics or AI-assisted functionality, suggesting a more ambitious integration scope than a basic CRUD project.",
                    2000,
                    "The project includes integrated advanced functionality.",
                ),
            }
        )

    issues: list[dict[str, Any]] = []
    if not rationale_present:
        issues.append(
            {
                "title": "Architecture decisions need stronger justification",
                "evidence": _truncate(
                    "The report names the main technologies and features, but it gives less explicit justification for why those design choices are the best fit for maintainability, scalability, or system boundaries.",
                    2000,
                    "Architecture rationale is under-explained.",
                ),
                "severity": "med",
            }
        )
    if testing_present:
        issues.append(
            {
                "title": "Testing breadth is still narrower than the overall system scope",
                "evidence": _truncate(
                    "The submission mentions testing and evaluation, but broader coverage for integration paths, failure handling, security edge cases, or long-running performance behaviour is not equally explicit.",
                    2000,
                    "Testing breadth is not fully clear for the whole system.",
                ),
                "severity": "med",
            }
        )
    if ai_present:
        issues.append(
            {
                "title": "AI integration risks and guardrails need clearer discussion",
                "evidence": _truncate(
                    "The project includes AI-assisted insight generation, but the report does not make the operational boundaries, failure cases, or trust limitations of that AI component equally explicit.",
                    2000,
                    "AI integration risks are under-explained.",
                ),
                "severity": "med",
            }
        )
    elif not security_present:
        issues.append(
            {
                "title": "Security discussion is thinner than the implementation scope",
                "evidence": _truncate(
                    "The submission focuses on functionality and evaluation, but it says less about access control, misuse handling, or defensive design considerations for a deployed system.",
                    2000,
                    "Security discussion is limited.",
                ),
                "severity": "med",
            }
        )
    if scalability_present and len(issues) < 3:
        issues.append(
            {
                "title": "Scalability evidence is useful but bounded",
                "evidence": _truncate(
                    "The report includes performance or scalability observations, yet those results appear scoped to the current implementation and would benefit from clearer boundaries, assumptions, and next-step optimisation priorities.",
                    2000,
                    "Scalability evidence remains bounded.",
                ),
                "severity": "low",
            }
        )
    issues = issues[:3]

    improvements: list[dict[str, Any]] = []
    improvements.append(
        {
            "action": "Explain major architecture decisions more explicitly",
            "why": "The project stack is concrete, but the rationale behind key design choices is not always fully justified.",
            "how": "Add a short architecture rationale section covering framework choice, data flow boundaries, AI integration boundaries, and the trade-offs that influenced those decisions.",
            "priority": 1,
        }
    )
    improvements.append(
        {
            "action": "Expand integration, failure-path, and edge-case testing",
            "why": "Current evaluation evidence is useful, but it does not fully cover the riskiest end-to-end paths or abnormal conditions.",
            "how": "Add tests for invalid transactions, AI-service failure handling, dashboard data edge cases, and performance under heavier portfolio sizes.",
            "priority": 2,
        }
    )
    if ai_present:
        improvements.append(
            {
                "action": "Document AI-assistant trust boundaries and fallback behaviour",
                "why": "AI features add value, but they also introduce uncertainty that should be managed explicitly.",
                "how": "Describe prompt/context boundaries, error handling, response limitations, and how the system behaves when AI output is unavailable or unreliable.",
                "priority": 3,
            }
        )
    elif not security_present:
        improvements.append(
            {
                "action": "Broaden the security review of the implemented system",
                "why": "The current report emphasises functionality more than defensive design or misuse handling.",
                "how": "Add a short section on access control, validation, abuse prevention, and any security assumptions that remain unresolved.",
                "priority": 3,
            }
        )

    checklist = [
        {"item": "Add explicit rationale for major framework and architecture choices", "done": False},
        {"item": "Add deeper end-to-end and edge-case tests", "done": False},
    ]
    if ai_present:
        checklist.append({"item": "Document AI failure handling and trust limitations", "done": False})
    elif not security_present:
        checklist.append({"item": "Document key security assumptions and gaps", "done": False})

    summary_sentences = _unique_preserve(
        [
            _truncate(abstract_sentences[0] if abstract_sentences else "", 320, ""),
            _truncate(abstract_sentences[1] if len(abstract_sentences) > 1 else "", 320, ""),
            _truncate(
                f"The submission gives clear evidence of {', '.join(features[:3]) or 'a real implemented system'}, and it appears technically substantial for a student software project.",
                320,
                "",
            ),
            _truncate(
                f"The main areas for improvement are stronger design justification, broader testing depth, and clearer discussion of {'AI integration limits' if ai_present else 'operational risks and limitations'}.",
                320,
                "",
            ),
        ]
    )
    summary = " ".join(summary_sentences[:4]).strip() or "The submission provides enough concrete evidence for a focused project review."

    architecture_overview = _truncate(
        f"The project appears to be a full-stack system built around {', '.join(tech_stack[:3]) or 'a web application stack'}, with evidence of concrete implementation and evaluation activity.",
        1200,
        "The project has a coherent implemented architecture.",
    )
    backend_text = _truncate(
        (
            "Backend evidence points to server-side application logic handling transactions, analytics, and service orchestration."
            if any(name in tech_stack for name in ("Django", "FastAPI", "Flask"))
            else "The backend structure is implied by the described workflows and data processing responsibilities."
        ),
        1200,
        "Backend evidence is limited.",
    )
    frontend_text = _truncate(
        (
            "Frontend evidence suggests dashboard-style user journeys, visual analytics, and interactive presentation of portfolio data."
            if _has_any(text, "dashboard", "chart", "chart.js", "interface", "ui")
            else "Frontend evidence is present but not deeply described."
        ),
        1200,
        "Frontend evidence is limited.",
    )
    database_text = _truncate(
        (
            "The project appears to rely on structured persistence for users, holdings, transactions, and historical market data."
            if _has_any(text, "database", "schema", "model", "transaction", "portfolio")
            else "Database design evidence is present only indirectly."
        ),
        1200,
        "Database evidence is limited.",
    )
    security_text = _truncate(
        (
            "Authentication and validation are concrete positives, but broader security controls and abuse handling deserve clearer discussion."
            if security_present
            else "Security implications are less explicit than the overall implementation scope."
        ),
        1200,
        "Security evidence is limited.",
    )

    evaluation_limitations = (
        "Evaluation evidence is present, but the report could still say more about failure paths, boundary conditions, and long-term operational risks."
        if testing_present
        else "Testing and evaluation evidence is less explicit than the implementation evidence."
    )
    academic_quality = (
        "The dissertation appears technically specific and evidence-rich, though parts of the design justification and critical reflection could be sharper."
        if abstract_sentences
        else "The submission appears technically grounded but would benefit from clearer academic framing."
    )

    ml_confidence = max(0.0, min(1.0, float(payload.ml.confidence_0_to_4) / 4.0))
    heuristic_confidence = 0.48
    heuristic_confidence += min(0.12, 0.03 * len(features))
    heuristic_confidence += 0.06 if testing_present else 0.0
    heuristic_confidence += 0.04 if len(abstract_sentences) >= 2 else 0.0
    heuristic_confidence += 0.04 if len(strengths) >= 2 else 0.0
    heuristic_confidence = max(0.42, min(0.72, heuristic_confidence))
    overall_confidence = min(0.7, round((heuristic_confidence + ml_confidence) / 2.0, 2))

    return {
        "summary": _truncate(summary, 1200, "The submission provides enough concrete evidence for a focused project review."),
        "issues": issues,
        "strengths": strengths[:3],
        "architecture_review": {
            "overview": architecture_overview,
            "backend": backend_text,
            "frontend": frontend_text,
            "database": database_text,
            "security": security_text,
        },
        "implementation_review": {
            "features_built": features[:6],
            "technical_quality": _truncate(
                "The implementation appears substantial and technically coherent, with multiple interacting project components rather than a minimal single-feature prototype.",
                1200,
                "Technical quality appears coherent.",
            ),
            "integration_quality": _truncate(
                "The report suggests meaningful integration between core workflows, analytics, visualisation, and supporting services, although some boundaries could be documented more explicitly.",
                1200,
                "Integration quality appears reasonable.",
            ),
        },
        "evaluation_review": {
            "testing_present": _truncate(
                "The submission includes concrete testing or evaluation evidence rather than only describing intended validation work."
                if testing_present
                else "Testing evidence is present only lightly in the available text.",
                1200,
                "Testing evidence is limited.",
            ),
            "limitations": _truncate(evaluation_limitations, 1200, "Evaluation limitations need clearer discussion."),
            "academic_quality": _truncate(academic_quality, 1200, "Academic quality appears reasonable."),
        },
        "improvement_plan": improvements[:3],
        "checklist": checklist[:3],
        "confidence": {
            "mode": "restricted" if safe_mode else "normal",
            "overall": 0.35 if safe_mode else overall_confidence,
        },
        "model_agreement": {
            "ml_confidence": ml_confidence,
            "llm_confidence": 0.35 if safe_mode else heuristic_confidence,
            "final_confidence": 0.35 if safe_mode else overall_confidence,
        },
        "safety": {
            "needs_review": bool(needs_review),
            "reason": "",
        },
    }


def _normalize_student_llm_json(obj: Any, *, safe_mode: bool) -> dict[str, Any]:
    data = _as_dict(obj).copy()
    data.pop("rag_meta", None)

    issues = []
    issues_source = _first_present(
        data,
        "issues",
        "weaknesses",
        "key_issues",
        "areas_for_improvement",
    )
    for idx, item in enumerate(_as_list(issues_source)):
        issue = _as_dict(item)
        issues.append(
            {
                "title": _truncate(issue.get("title"), 200, f"Issue {idx + 1}"),
                "evidence": _truncate(issue.get("evidence"), 2000, "Evidence was not provided."),
                "severity": _normalize_severity(issue.get("severity")),
            }
        )

    strengths = []
    strengths_source = _first_present(data, "strengths", "positives", "key_strengths")
    for idx, item in enumerate(_as_list(strengths_source)):
        strength = _as_dict(item)
        strengths.append(
            {
                "title": _truncate(strength.get("title"), 200, f"Strength {idx + 1}"),
                "evidence": _truncate(strength.get("evidence"), 2000, "Supporting evidence was not provided."),
            }
        )

    improvement_plan = []
    improvement_source = _first_present(
        data,
        "improvement_plan",
        "recommendations",
        "improvements",
        "next_steps",
        "actions",
    )
    for idx, item in enumerate(_as_list(improvement_source)):
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
    checklist_source = _first_present(data, "checklist", "action_items", "todo")
    for idx, item in enumerate(_as_list(checklist_source)):
        check = _as_dict(item)
        checklist.append(
            {
                "item": _truncate(check.get("item"), 200, f"Checklist item {idx + 1}"),
                "done": _coerce_bool(check.get("done"), default=False),
            }
        )

    architecture_review = _as_dict(
        _first_present(data, "architecture_review", "architecture", "system_architecture")
    )
    implementation_review = _as_dict(
        _first_present(data, "implementation_review", "implementation", "technical_review")
    )
    evaluation_review = _as_dict(
        _first_present(data, "evaluation_review", "evaluation", "testing_review")
    )
    confidence = _as_dict(data.get("confidence"))
    model_agreement = _as_dict(data.get("model_agreement"))
    safety = _as_dict(data.get("safety"))
    normalized_final_confidence = _coerce_float(model_agreement.get("final_confidence"), default=-1.0)
    normalized_overall = _coerce_float(confidence.get("overall"), default=-1.0)
    normalized_llm_confidence = _coerce_float(model_agreement.get("llm_confidence"), default=-1.0)
    normalized_ml_confidence = _coerce_float(model_agreement.get("ml_confidence"), default=-1.0)

    if normalized_overall < 0.0:
        normalized_overall = (
            normalized_final_confidence
            if normalized_final_confidence >= 0.0
            else (0.35 if safe_mode else 0.75)
        )
    if normalized_llm_confidence < 0.0:
        normalized_llm_confidence = normalized_overall
    if normalized_final_confidence < 0.0:
        normalized_final_confidence = normalized_llm_confidence
    if normalized_ml_confidence < 0.0:
        normalized_ml_confidence = 0.0

    return {
        "summary": _truncate(
            _first_present(data, "summary", "overview", "feedback_summary"),
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
            "ml_confidence": normalized_ml_confidence,
            "llm_confidence": normalized_llm_confidence,
            "final_confidence": normalized_final_confidence,
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
    return "student_project_review" if student_feedback_style(payload) == "code" else "student"


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


async def _call_content_retry_model(prompt_text: str) -> dict[str, Any]:
    """Retry the full original prompt on the fallback model after low-content output."""
    return await _call_generate_model(
        {
            "prompt": prompt_text,
            "stream": False,
            "format": "json",
        },
        requested_model=_REPAIR_MODEL or None,
    )


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

    try:
        gen = await _call_generate_model(
            request_payload,
            requested_model=(payload.requested_model or "").strip() or None,
        )
    except Exception as exc:
        requested = (payload.requested_model or "").strip() or "auto"
        logger.warning("llm generate failed requested_model=%s error=%s", requested, exc)
        raise HTTPException(
            status_code=502,
            detail=f"local generation failed requested_model={requested}: {type(exc).__name__}: {exc}",
        ) from exc
    model_used = str(gen.get("model_used") or payload.requested_model or "")
    response_text = str(gen.get("response") or "")

    return JSONResponse(
        content={"response": response_text, "model_used": model_used},
        headers=_llm_response_headers(
            model_used,
            used_fallback=(model_used == _active_fallback_model() and model_used != _active_primary_model()),
        ),
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
    feedback_style = student_feedback_style(payload)

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
        "student_report start requested_mode=%s analysis_type=%s feedback_style=%s safe_mode=%s needs_review=%s injected=%s retrieval_weak=%s quality_band=%s confidence_0_to_4=%s query=%r top_k=%s retrieved_chunks=%s citations=%s",
        requested_mode or "unset",
        (payload.analysis_type or "").strip().lower() or "unset",
        feedback_style,
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

    prompt_text, excerpt_cap, was_trimmed = student_prompt(payload, safe_mode=safe_mode)
    logger.info(
        "student_report prompt_stats prompt_chars=%s text_chars=%s excerpt_cap=%s trimmed=%s ocr_chars=%s audio_chars=%s rag_context_chars=%s rag_enabled=%s chunks=%s citations=%s",
        len(prompt_text),
        len(payload.ingestion.text_content or ""),
        excerpt_cap,
        was_trimmed,
        len(payload.ingestion.ocr_text or ""),
        len(payload.ingestion.audio_transcript or ""),
        len(payload.grounding_context or (payload.rag.context if payload.rag else "") or ""),
        bool(payload.rag and payload.rag.enabled) or bool(payload.grounding_context),
        len(payload.grounding_retrieved_chunks or []),
        len(payload.grounding_citations or []),
    )

    gen = await generate_with_fallback(prompt_text)
    model_used = gen["model_used"]
    raw = str(gen.get("response") or "")
    for attempt in range(settings.max_retries + 1):
        logger.info(
            "student_report raw_response attempt=%s/%s model=%s chars=%s preview=%r",
            attempt + 1,
            settings.max_retries + 1,
            model_used,
            len(raw),
            _clip(raw, 800),
        )
        try:
            # _parse_llm_json = extract (balanced + fence-aware) + cheap_repair + json.loads
            parsed = _parse_llm_json(raw)
            obj = _normalize_student_llm_json(parsed, safe_mode=safe_mode)
            low_content_quality = _student_report_low_content_quality(obj)
            normalization_removed_content = _student_normalization_removed_content(parsed, obj)
            metrics = _student_debug_metrics(obj)
            logger.info(
                "student_report normalization attempt=%s/%s model=%s parsed_keys=%s low_content=%s normalization_removed_content=%s summary_placeholder=%s issues=%s strengths=%s improvements=%s checklist=%s",
                attempt + 1,
                settings.max_retries + 1,
                model_used,
                sorted(_as_dict(parsed).keys()),
                metrics["low_content_quality"],
                normalization_removed_content,
                metrics["summary_placeholder"],
                metrics["issues"],
                metrics["strengths"],
                metrics["improvements"],
                metrics["checklist"],
            )
            if low_content_quality and attempt < settings.max_retries:
                logger.warning(
                    "student_report routing_to_retry failure=low_content_quality attempt=%s model=%s retry_model=%s normalization_removed_content=%s raw=%r",
                    attempt + 1,
                    model_used,
                    _REPAIR_MODEL or "fallback",
                    normalization_removed_content,
                    _clip(raw, 800),
                )
                gen = await _call_content_retry_model(prompt_text)
                model_used = gen["model_used"]
                raw = str(gen.get("response") or "")
                continue

            out = StudentReportOut.model_validate(obj)

            response_body = out.model_dump()
            response_body.setdefault("safety", {})
            low_content_quality = _student_report_low_content_quality(response_body)
            if low_content_quality:
                heuristic_body = _student_heuristic_report_from_payload(
                    payload,
                    safe_mode=safe_mode,
                    needs_review=needs_review,
                )
                if heuristic_body is not None:
                    response_body = heuristic_body
                    low_content_quality = False
                    logger.warning(
                        "student_report heuristic_recovery model=%s safe_mode=%s injected=%s retrieval_weak=%s",
                        model_used,
                        safe_mode,
                        injected,
                        retrieval_weak,
                    )
                else:
                    response_body["safety"]["needs_review"] = True
                    response_body["safety"]["reason"] = _LOW_CONTENT_QUALITY_REASON
                    response_body.setdefault("confidence", {})
                    response_body["confidence"]["overall"] = min(
                        _coerce_float(response_body["confidence"].get("overall"), default=0.35),
                        0.35,
                    )
                    logger.warning(
                        "student_report quality_gate=low_content_quality model=%s safe_mode=%s injected=%s retrieval_weak=%s",
                        model_used,
                        safe_mode,
                        injected,
                        retrieval_weak,
                    )
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
            if low_content_quality:
                response_body["rag_meta"]["quality_gate"] = _LOW_CONTENT_QUALITY_REASON
            elif response_body["summary"] != out.model_dump().get("summary"):
                response_body["rag_meta"]["recovery"] = "heuristic_student_builder"

            if attempt > 0:
                logger.info(
                    "student_report recovered attempt=%s/%s recovered_by=%s",
                    attempt + 1,
                    settings.max_retries + 1,
                    model_used,
                )
            return JSONResponse(
                content=response_body,
                headers=_llm_response_headers(
                    model_used,
                    used_fallback=(attempt > 0 or model_used == _active_fallback_model()),
                    extra={
                    "x-rag-enabled": str(response_body["rag_meta"]["enabled"]).lower(),
                    "x-rag-confidence-label": str(
                        response_body["rag_meta"]["confidence_label"]
                    ),
                    },
                ),
            )

        except json.JSONDecodeError as exc:
            logger.warning(
                "student_report failure=json_parse attempt=%s/%s model=%s safe_mode=%s injected=%s retrieval_weak=%s error=%s raw=%r",
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
                    "student_report failure=repair_exhausted model=%s safe_mode=%s injected=%s retrieval_weak=%s reason=%s raw=%r",
                    model_used,
                    safe_mode,
                    injected,
                    retrieval_weak,
                    inj_reason,
                    _clip(raw),
                )
                heuristic_body = _student_heuristic_report_from_payload(
                    payload,
                    safe_mode=safe_mode,
                    needs_review=needs_review,
                )
                if heuristic_body is not None:
                    heuristic_body["rag_meta"] = _student_rag_meta(payload)
                    heuristic_body["rag_meta"]["recovery"] = "heuristic_student_builder"
                    return JSONResponse(
                        content=heuristic_body,
                        status_code=200,
                        headers=_llm_response_headers(model_used, used_fallback=True),
                    )
                fallback_body = _student_safe_fallback(model_used, safe_mode, needs_review)
                fallback_body["rag_meta"] = _student_rag_meta(payload)
                return JSONResponse(
                    content=fallback_body,
                    status_code=200,
                    headers=_llm_response_headers(model_used, used_fallback=True),
                )

            # JSON failure = model failure: retry explicitly on the repair/fallback model
            logger.info(
                "student_report routing_to_repair failure=json_parse attempt=%s model=%s repair_model=%s",
                attempt + 1,
                model_used,
                _REPAIR_MODEL or "fallback",
            )
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
                "student_report failure=validation attempt=%s/%s model=%s safe_mode=%s injected=%s retrieval_weak=%s errors=%s raw=%r",
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
                    "student_report failure=repair_exhausted model=%s safe_mode=%s injected=%s retrieval_weak=%s reason=%s raw=%r",
                    model_used,
                    safe_mode,
                    injected,
                    retrieval_weak,
                    inj_reason,
                    _clip(raw),
                )
                heuristic_body = _student_heuristic_report_from_payload(
                    payload,
                    safe_mode=safe_mode,
                    needs_review=needs_review,
                )
                if heuristic_body is not None:
                    heuristic_body["rag_meta"] = _student_rag_meta(payload)
                    heuristic_body["rag_meta"]["recovery"] = "heuristic_student_builder"
                    return JSONResponse(
                        content=heuristic_body,
                        status_code=200,
                        headers=_llm_response_headers(model_used, used_fallback=True),
                    )
                fallback_body = _student_safe_fallback(model_used, safe_mode, needs_review)
                fallback_body["rag_meta"] = _student_rag_meta(payload)
                return JSONResponse(
                    content=fallback_body,
                    status_code=200,
                    headers=_llm_response_headers(model_used, used_fallback=True),
                )

            # Validation failure after parsing = also route to the repair/fallback model
            logger.info(
                "student_report routing_to_repair failure=validation attempt=%s model=%s repair_model=%s",
                attempt + 1,
                model_used,
                _REPAIR_MODEL or "fallback",
            )
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

    prof_prompt_text = professor_prompt(payload, needs_review=needs_review)
    text_chars = len(payload.ingestion.text_content or "")
    logger.info(
        "professor_report prompt_stats prompt_chars=%s text_chars=%s trimmed=%s ocr_chars=%s audio_chars=%s rag_context_chars=%s rag_enabled=%s chunks=%s citations=%s",
        len(prof_prompt_text),
        text_chars,
        text_chars > 5_000,
        len(payload.ingestion.ocr_text or ""),
        len(payload.ingestion.audio_transcript or ""),
        len(payload.grounding_context or (payload.rag.context if payload.rag else "") or ""),
        bool(payload.rag and payload.rag.enabled) or bool(payload.grounding_context),
        len(payload.grounding_retrieved_chunks or []),
        len(payload.grounding_citations or []),
    )
    gen = await generate_with_fallback(prof_prompt_text)
    model_used = gen["model_used"]
    raw = gen["response"]

    for attempt in range(settings.max_retries + 1):
        try:
            # _parse_llm_json = extract (balanced + fence-aware) + cheap_repair + json.loads
            obj = _normalize_professor_output(_parse_llm_json(raw))
            out = ProfessorReportOut.model_validate(obj)

            response_body = out.model_dump()
            response_body["rag_meta"] = _professor_rag_meta(payload)

            if attempt > 0:
                logger.info(
                    "professor_report recovered attempt=%s/%s recovered_by=%s",
                    attempt + 1,
                    settings.max_retries + 1,
                    model_used,
                )
            return JSONResponse(
                content=response_body,
                headers=_llm_response_headers(
                    model_used,
                    used_fallback=(attempt > 0 or model_used == _active_fallback_model()),
                    extra={
                    "x-rag-enabled": str(response_body["rag_meta"]["enabled"]).lower(),
                    "x-rag-confidence-label": str(
                        response_body["rag_meta"]["confidence_label"]
                    ),
                    },
                ),
            )

        except json.JSONDecodeError as exc:
            logger.warning(
                "professor_report failure=json_parse attempt=%s/%s model=%s needs_review=%s injected=%s retrieval_weak=%s error=%s raw=%r",
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
                    "professor_report failure=repair_exhausted model=%s needs_review=%s injected=%s retrieval_weak=%s reason=%s raw=%r",
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
                    headers=_llm_response_headers(model_used, used_fallback=True),
                )

            # JSON failure = model failure: retry explicitly on the repair/fallback model
            logger.info(
                "professor_report routing_to_repair failure=json_parse attempt=%s model=%s repair_model=%s",
                attempt + 1,
                model_used,
                _REPAIR_MODEL or "fallback",
            )
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
                "professor_report failure=validation attempt=%s/%s model=%s needs_review=%s injected=%s retrieval_weak=%s errors=%s raw=%r",
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
                    "professor_report failure=repair_exhausted model=%s needs_review=%s injected=%s retrieval_weak=%s reason=%s raw=%r",
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
                    headers=_llm_response_headers(model_used, used_fallback=True),
                )

            # Validation failure after parsing = also route to the repair/fallback model
            logger.info(
                "professor_report routing_to_repair failure=validation attempt=%s model=%s repair_model=%s",
                attempt + 1,
                model_used,
                _REPAIR_MODEL or "fallback",
            )
            gen = await _call_repair_model(
                fix_json_prompt(
                    raw,
                    "professor",
                    forced_needs_review=needs_review,
                )
            )
            model_used = gen["model_used"]
            raw = gen["response"]
