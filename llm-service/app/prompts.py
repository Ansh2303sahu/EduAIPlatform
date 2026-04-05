import json
from .schemas import StudentReportIn, ProfessorReportIn

_JSON_RULES = """
MANDATORY OUTPUT FORMAT
- Return ONLY a single valid JSON object â€” nothing else.
- Your response MUST begin with the character { and end with the character }.
- Do NOT write any text, explanation, or commentary before the opening {.
- Do NOT write any text, explanation, or commentary after the closing }.
- Do NOT wrap the JSON in markdown fences (no ```json, no ```).
- Use double quotes for ALL keys and ALL string values.
- Do NOT use single quotes anywhere.
- Do NOT use trailing commas (a comma before } or before ]).
- Do NOT omit required keys â€” use empty arrays [] or empty strings "" as defaults.
- Do NOT add keys that are not in the schema.
- Do NOT use Python-style True / False / None â€” use JSON true / false / null.
""".strip()


_STRICT_STUDENT_JSON_OUTPUT_RULES = """
STRICT OUTPUT RULES
- Return exactly one JSON object and nothing else.
- The FIRST character of your entire response must be {.
- The LAST character of your entire response must be }.
- The FIRST key inside the JSON must be "summary" â€” populate it before any other key.
- Do not add any prose, explanation, or preamble before the opening brace.
- Do not add any prose, explanation, or closing remark after the final brace.
- Do not wrap the JSON in markdown fences (no ```json blocks).
- If a value is uncertain, still return a valid JSON value â€” never return commentary in place of a value.
- Never omit the "summary" key â€” it is required and must contain at least 2 sentences.
- Severity values must be exactly "low", "med", or "high" â€” no other strings.
- If the schema contains "confidence", then "confidence.mode" must be exactly "normal" or "restricted".
- Use [] for empty arrays, never omit an array field entirely.
""".strip()


# Appended verbatim at the very end of every student and professor prompt, after
# all other instructions, so the model sees it immediately before generating.
_FINAL_JSON_REMINDER = (
    'Output only the JSON object. '
    'Write nothing before the opening { and nothing after the closing }. '
    'Do not use markdown fences.'
)


# Adaptive excerpt caps — thresholds are exported so main.py can log them.
# Normal: text_content ≤ LONG_INPUT_THRESHOLD  (no cap-reduction)
# Long:   text_content >  LONG_INPUT_THRESHOLD  (tighter cap + fewer snippets)
LONG_INPUT_THRESHOLD: int = 8_000

_EXCERPT_CODE_NORMAL:  int = 2_800
_EXCERPT_CODE_LONG:    int = 1_800
_EXCERPT_ESSAY_NORMAL: int = 2_200
_EXCERPT_ESSAY_LONG:   int = 1_400


def _compact_ingestion(ing) -> str:
    obj = {
        "text_content": (ing.text_content or "")[:5_000],
        "ocr_text": (ing.ocr_text or "")[:2_000],
        "audio_transcript": (ing.audio_transcript or "")[:2_000],
        "tables_json": ing.tables_json or {},
    }
    return json.dumps(obj, ensure_ascii=False)


def _safe_get(payload, key: str, default=None):
    try:
        return getattr(payload, key, default)
    except Exception:
        return default


def _field(source, key: str, default=None):
    if isinstance(source, dict):
        return source.get(key, default)
    try:
        return getattr(source, key, default)
    except Exception:
        return default


def build_rag_section(
    context: str | None,
    instruction: str | None,
    citations: list | None = None,
    retrieved_chunks: list | None = None,
    confidence_label: str | None = None,
    confidence_score: float | None = None,
    safe_review: bool | None = None,
) -> str:
    if not context:
        return """
ACADEMIC KNOWLEDGE BASE
No approved grounding context was provided.

IMPORTANT INSTRUCTIONS
- Do not invent policies, rubric rules, or academic guidance.
- Base your answer only on the submission content and ML signals.
- If certainty is limited, keep claims conservative.
""".strip()

    citation_lines = []
    for fallback_idx, c in enumerate(citations or [], start=1):
        idx = _field(c, "index", fallback_idx)
        title = _field(c, "title", "Untitled")
        section = _field(c, "section", "unknown")
        citation_lines.append(f"[{idx}] {title} | section: {section}")

    citation_block = "\n".join(citation_lines).strip()
    if not citation_block:
        citation_block = "No citation metadata provided."

    chunk_lines = []
    for idx, chunk in enumerate((retrieved_chunks or [])[:4], start=1):
        title = _field(chunk, "document_title", "Untitled")
        section = _field(chunk, "section", "unknown")
        category = _field(chunk, "category", "general")
        score = _field(chunk, "score", "unknown")
        chunk_lines.append(
            f"[Chunk {idx}] {title} | section: {section} | category: {category} | score: {score}"
        )

    chunk_block = "\n".join(chunk_lines).strip()
    if not chunk_block:
        chunk_block = "No retrieved chunk metadata provided."

    return f"""
ACADEMIC KNOWLEDGE BASE
The following information was retrieved from approved academic guidance documents.

Retrieved context:
{context}

Retrieved citations:
{citation_block}

Retrieved chunk signals:
{chunk_block}

Retrieval confidence:
- confidence_label: {confidence_label or "unknown"}
- confidence_score: {confidence_score if confidence_score is not None else 0.0}
- safe_review: {bool(safe_review)}

IMPORTANT INSTRUCTIONS
{instruction or "Use the retrieved context carefully and avoid unsupported claims."}

Use this knowledge when giving feedback.
When your output fields draw on the retrieved context, reference the source inline using its citation index, e.g. [1] or [2], matching the indices listed above.
Give extra weight to chunks that have a higher score or are marked as official sources.
Do NOT invent academic policies, rubric rules, marking rules, or citation guidance that are not supported by the retrieved context.
If retrieval confidence is low or safe_review is true, be more cautious and explicitly signal uncertainty in the safety.reason field.
""".strip()


def _student_standard_schema() -> dict:
    return {
        "summary": "string",
        "strengths": [
            {
                "title": "string",
                "evidence": "string"
            }
        ],
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
        "confidence": {
            "mode": "normal",
            "overall": 0.0
        },
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
def _student_project_schema() -> dict:
    return {
        "summary": "string",
        "issues": [
            {
                "title": "string",
                "evidence": "string",
                "severity": "low"
            }
        ],
        "strengths": [
            {
                "title": "string",
                "evidence": "string"
            }
        ],
        "architecture_review": {
            "overview": "string",
            "backend": "string",
            "frontend": "string",
            "database": "string",
            "security": "string"
        },
        "implementation_review": {
            "features_built": ["string"],
            "technical_quality": "string",
            "integration_quality": "string"
        },
        "evaluation_review": {
            "testing_present": "string",
            "limitations": "string",
            "academic_quality": "string"
        },
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
        "confidence": {
            "mode": "normal",
            "overall": 0.0
        },
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
def _collapse_ws(value: str | None) -> str:
    return " ".join((value or "").split())
def _clip_text(value: str | None, limit: int) -> str:
    text = _collapse_ws(value)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."
def _keyword_snippets(
    text: str | None,
    keyword_groups: list[tuple[str, tuple[str, ...]]],
    *,
    window: int = 220,
    max_snippets: int = 6,
) -> list[str]:
    source = _collapse_ws(text)
    lowered = source.casefold()
    snippets: list[str] = []
    seen: set[str] = set()
    for label, keywords in keyword_groups:
        hit = -1
        hit_len = 0
        for keyword in keywords:
            idx = lowered.find(keyword.casefold())
            if idx >= 0:
                hit = idx
                hit_len = len(keyword)
                break
        if hit < 0:
            continue
        start = max(0, hit - window)
        end = min(len(source), hit + hit_len + window)
        snippet = source[start:end].strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(source):
            snippet = snippet + "..."
        signature = snippet.casefold()
        if signature in seen:
            continue
        seen.add(signature)
        snippets.append(f"- {label}: {snippet[:560]}")
        if len(snippets) >= max_snippets:
            break
    return snippets
def _student_submission_evidence(
    payload: StudentReportIn,
    *,
    feedback_style: str,
) -> tuple[str, int, bool]:
    """Build the SUBMISSION EVIDENCE DIGEST block.

    Returns ``(digest_text, excerpt_cap_used, was_trimmed)`` so callers can
    log the trimming decision without rebuilding the digest.
    """
    text_content = payload.ingestion.text_content or ""
    is_long = len(text_content) > LONG_INPUT_THRESHOLD

    if feedback_style == "code":
        excerpt_cap = _EXCERPT_CODE_LONG if is_long else _EXCERPT_CODE_NORMAL
        max_snippets = 4 if is_long else 6
        keyword_groups = [
            ("stack", ("django", "react", "fastapi", "flask", "api", "web framework")),
            ("architecture", ("architecture", "backend", "frontend", "service", "dashboard")),
            ("data", ("database", "schema", "transaction", "portfolio", "model")),
            ("security", ("authentication", "authorization", "security", "login", "validation")),
            ("testing", ("test", "testing", "unit test", "acceptance", "evaluation")),
            ("analytics_or_ai", ("analytics", "gemini", "ai", "sharpe", "beta", "moving average")),
        ]
    else:
        excerpt_cap = _EXCERPT_ESSAY_LONG if is_long else _EXCERPT_ESSAY_NORMAL
        max_snippets = 4 if is_long else 5
        keyword_groups = [
            ("task_focus", ("objective", "question", "aim", "argument", "thesis")),
            ("evidence", ("evidence", "source", "reference", "citation", "study")),
            ("analysis", ("analysis", "evaluate", "comparison", "critical", "discussion")),
            ("structure", ("introduction", "conclusion", "section", "paragraph")),
            ("referencing", ("apa", "harvard", "references", "bibliography")),
        ]

    was_trimmed = len(text_content) > excerpt_cap
    snippets = _keyword_snippets(text_content, keyword_groups, max_snippets=max_snippets)
    ocr_cap = 400 if is_long else 600
    audio_cap = 400 if is_long else 600

    tables_json = payload.ingestion.tables_json or {}
    if isinstance(tables_json, dict):
        table_summary = f"dict keys={list(tables_json.keys())[:8]}"
    elif isinstance(tables_json, list):
        table_summary = f"list items={len(tables_json)}"
    else:
        table_summary = "none"

    lines = [
        "SUBMISSION EVIDENCE DIGEST",
        f"main_excerpt: {_clip_text(text_content, excerpt_cap)}",
        f"ocr_excerpt: {_clip_text(payload.ingestion.ocr_text, ocr_cap)}" if payload.ingestion.ocr_text else "ocr_excerpt: none",
        f"audio_excerpt: {_clip_text(payload.ingestion.audio_transcript, audio_cap)}" if payload.ingestion.audio_transcript else "audio_excerpt: none",
        f"table_signal: {table_summary}",
        "targeted_snippets:",
    ]
    if snippets:
        lines.extend(snippets)
    else:
        lines.append("- No targeted snippets extracted from the submission text.")
    return "\n".join(lines), excerpt_cap, was_trimmed
def _project_review_context(payload: StudentReportIn) -> str:
    tables_json = payload.ingestion.tables_json or {}
    if isinstance(tables_json, dict):
        table_signal = {
            "table_keys": list(tables_json.keys())[:12],
            "table_count_hint": len(tables_json),
        }
    elif isinstance(tables_json, list):
        table_signal = {"table_count_hint": len(tables_json)}
    else:
        table_signal = {"table_count_hint": 0}
    obj = {
        "submission_excerpt": _clip_text(payload.ingestion.text_content, 2200),
        "ocr_excerpt": _clip_text(payload.ingestion.ocr_text, 700),
        "audio_excerpt": _clip_text(payload.ingestion.audio_transcript, 700),
        "table_signal": table_signal,
        "ml_summary": {
            "feedback_category": payload.ml.feedback_category,
            "quality_band": payload.ml.quality_band,
            "confidence_0_to_4": payload.ml.confidence_0_to_4,
        },
        "analysis_focus": list((
            _safe_get(
                payload,
                "analysis_focus",
                [
                    "project aim",
                    "technical stack",
                    "architecture",
                    "implementation quality",
                    "security",
                    "testing",
                    "limitations",
                ],
            )
            or [
                "project aim",
                "technical stack",
                "architecture",
                "implementation quality",
                "security",
                "testing",
                "limitations",
            ]
        ))[:6],
        "mode": _safe_get(payload, "mode", "normal"),
        "submission_type": _safe_get(payload, "submission_type", ""),
        "safety_flags": _safe_get(payload, "safety_flags", {}) or {},
    }
    return json.dumps(obj, ensure_ascii=False, indent=2)
def _student_content_floor(feedback_style: str) -> str:
    if feedback_style == "code":
        return """
CONTENT FLOOR
- Do not return the placeholder phrases "Automated review generated with limited confidence." or "Not assessed." unless the submission truly provides no evidence for that exact field.
- If evidence is limited for a field, write a short evidence-gap explanation such as "Evidence is limited for backend structure because ..." rather than using "Not assessed." by itself.
- If the submission clearly names technologies, features, tests, dashboards, APIs, data models, or security controls, do not leave issues, improvement_plan, and checklist all empty.
- When the submission contains enough evidence, include at least 2 strengths, 2-3 issues, 2-3 improvement actions, and 3 checklist items.
- Each architecture_review, implementation_review, and evaluation_review field must contain a concrete observation tied to the submission or a specific evidence-gap explanation.
- If you identify implemented components, name them directly: frameworks, routes, pages, services, APIs, data models, dashboards, tests, workflows, or security controls.
- Do not invent web-app concerns for a local or desktop tool. Missing authentication is only an issue when the evidence shows accounts, roles, sensitive remote access, or an exposed multi-user workflow.
- If you produce substantive feedback, set model_agreement.llm_confidence and model_agreement.final_confidence to non-zero values that reflect how reliable the feedback is.
""".strip()
    return """
CONTENT FLOOR
- Do not return the placeholder phrase "Automated review generated with limited confidence." unless the submission truly provides almost no usable evidence.
- If the submission contains concrete topic evidence, do not leave issues, improvement_plan, and checklist all empty.
- When the submission contains enough evidence, include at least 2 strengths, 2-3 issues, 2-3 improvement actions, and 3 checklist items.
- Keep the summary concrete and tied to the actual submission rather than generic academic advice.
- Keep the language academic. Do not drift into software-review phrases such as implementation quality, separation of concerns, APIs, or authentication unless the submission is clearly a software build report.
- If you produce substantive feedback, set model_agreement.llm_confidence and model_agreement.final_confidence to non-zero values that reflect how reliable the feedback is.
""".strip()
def _keyword_score(text: str, terms: tuple[str, ...]) -> int:
    lowered = text.casefold()
    return sum(1 for term in terms if term.casefold() in lowered)
def student_feedback_style(payload: StudentReportIn, *, project_review: bool | None = None) -> str:
    analysis_type = str(_safe_get(payload, "analysis_type", "") or "").strip().lower()
    submission_type = str(_safe_get(payload, "submission_type", "") or "").strip().lower()
    if project_review is None:
        project_review = analysis_type == "student_project_review"
    if project_review or submission_type == "project":
        return "code"
    combined = " ".join(
        [
            str(payload.ingestion.text_content or ""),
            str(payload.ingestion.ocr_text or ""),
            str(payload.ingestion.audio_transcript or ""),
        ]
    )
    essay_terms = (
        "thesis",
        "argument",
        "literature",
        "reference",
        "references",
        "citation",
        "apa",
        "harvard",
        "essay",
        "conclusion",
        "paragraph",
        "critical analysis",
        "evidence",
        "risk assessment",
        "framework",
        "cia triad",
        "nist",
        "stride",
        "incident response",
        "mitigation",
        "policy",
    )
    code_terms = (
        "function",
        "abstract class",
        "subclass",
        "bug",
        "variable",
        "repository",
        "commit",
        "testing",
        "database",
        "sql",
        "exception",
        "stack trace",
        "module",
        "implemented in",
        "built using",
        "windows forms",
        "github",
        "source code",
        "codebase",
        "unit test",
        "integration test",
        "endpoint implementation",
        "api route",
    )
    essay_score = _keyword_score(combined, essay_terms)
    code_score = _keyword_score(combined, code_terms)
    if code_score >= max(2, essay_score + 1):
        return "code"
    return "essay"
def _student_style_block(feedback_style: str) -> str:
    if feedback_style == "code":
        return """
ASSESSOR STYLE
- Act like a senior software reviewer giving technically rigorous feedback on a student code or software submission.
- Focus on correctness, readability, modularity, maintainability, testing, error handling, security, performance, and documentation.
- Avoid vague praise such as "good project" unless you name the specific technical strength and why it matters.
ISSUE WRITING
- Make each issue title specific.
- In the evidence field, explain the technical weakness, why it matters, and the likely fix direction.
SEVERITY CALIBRATION
- Use "high" for correctness, security, data integrity, or failure-handling gaps that could seriously undermine trust in the system.
- Use "med" for testing, maintainability, modularity, separation-of-concerns, or performance gaps that materially limit quality.
- Use "low" only for polish, minor readability, naming, or documentation issues.
""".strip()
    return """
ASSESSOR STYLE
- Act like a strict university marker reviewing an essay or academic written submission.
- Focus on argument, structure, evidence use, critical analysis, referencing, clarity, and academic quality.
- Avoid generic praise; name the exact strength and why it is academically effective.
ISSUE WRITING
- Make each issue title specific.
- In the evidence field, explain the academic weakness, why it matters to the quality of the argument or grade, and the revision direction.
SEVERITY CALIBRATION
- Use "high" for serious thesis, structure, analysis, evidence, or referencing problems that weaken the submission substantially.
- Use "med" for partially developed analysis, weak paragraph logic, shallow evidence handling, or inconsistent academic clarity.
- Use "low" only for local phrasing, minor clarity, or small polish issues.
""".strip()


def student_prompt(payload: StudentReportIn, safe_mode: bool) -> tuple[str, int, bool]:
    """Build the student LLM prompt.

    Returns ``(prompt_text, excerpt_cap_used, was_trimmed)`` so callers can
    log the trimming decision without re-running the prompt builder.
    """
    analysis_type = str(_safe_get(payload, "analysis_type", "") or "").strip().lower()
    project_review = analysis_type == "student_project_review"
    feedback_style = student_feedback_style(payload, project_review=project_review)
    code_style = feedback_style == "code"
    schema = _student_project_schema() if code_style else _student_standard_schema()
    rag = _safe_get(payload, "rag", None)
    grounding_context = _safe_get(payload, "grounding_context", "")
    grounding_instruction = _safe_get(payload, "grounding_instruction", "")
    grounding_citations = _safe_get(payload, "grounding_citations", []) or []
    grounding_retrieved_chunks = _safe_get(payload, "grounding_retrieved_chunks", []) or []
    retrieval_confidence_score = _safe_get(payload, "retrieval_confidence_score", 0.0)
    retrieval_confidence_label = _safe_get(payload, "retrieval_confidence_label", "low")
    retrieval_safe_review = _safe_get(payload, "retrieval_safe_review", False)
    if isinstance(rag, dict):
        grounding_context = rag.get("context", grounding_context)
        grounding_instruction = rag.get("instruction", grounding_instruction)
        grounding_citations = rag.get("citations", grounding_citations)
        grounding_retrieved_chunks = rag.get("retrieved_chunks", grounding_retrieved_chunks)
        retrieval_confidence_score = rag.get("confidence_score", retrieval_confidence_score)
        retrieval_confidence_label = rag.get("confidence_label", retrieval_confidence_label)
        retrieval_safe_review = rag.get("safe_review", retrieval_safe_review)
    rag_section = build_rag_section(
        context=grounding_context,
        instruction=grounding_instruction,
        citations=grounding_citations,
        retrieved_chunks=grounding_retrieved_chunks,
        confidence_label=retrieval_confidence_label,
        confidence_score=retrieval_confidence_score,
        safe_review=retrieval_safe_review,
    )
    mode = (
        """
SAFE MODE:
- You MUST still populate every required JSON field with real content - empty strings and null values are not acceptable.
- Being cautious means qualifying your claims, not omitting them. Write feedback that is hedged but still specific and useful.
- Keep claims conservative and avoid strong assertions where evidence is limited.
- Do not grade.
- Set safety.needs_review to true.
- Explain the reason briefly in safety.reason (1-2 sentences).
- If the retrieved academic guidance is weak or uncertain, mention limited grounding in safety.reason.
- summary, issues, improvement_plan, and checklist must all contain substantive content.
"""
        if safe_mode
        else
        """
NORMAL MODE:
- Be specific and constructive.
- Do not grade.
- Set safety.needs_review to false unless there is a strong reason.
- Prefer advice that is supported by the retrieved academic guidance.
- If retrieval confidence is low, avoid overclaiming and use safety.reason to explain uncertainty when needed.
"""
    ).strip()

    # Build evidence digest once so both branches can use it and return trim metadata.
    evidence_block, excerpt_cap, was_trimmed = _student_submission_evidence(
        payload, feedback_style=feedback_style
    )

    if code_style:
        prompt_text = f"""
You are a senior university computing assessor with deep expertise in software engineering, code review, and academic project evaluation. Your feedback is technically precise, evidence-grounded, and directly useful to the student.
Your task: produce detailed technical feedback for the student software or code submission below.
{_JSON_RULES}
{_STRICT_STUDENT_JSON_OUTPUT_RULES}
You MUST return exactly one JSON object matching this schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}
Extra constraints:
- "summary" must be 2-4 sentences giving a concrete overview of the submission goal, overall technical quality, and the most important gap.
- "issues" must be an array of objects with keys: title, evidence, severity.
- "severity" must be exactly one of: "low", "med", "high".
- "strengths" must be an array of objects with keys: title, evidence.
- "architecture_review" must contain keys: overview, backend, frontend, database, security.
- "implementation_review" must contain keys: features_built, technical_quality, integration_quality.
- "features_built" must be a list of concrete implemented capabilities, modules, workflows, or endpoints when present.
- "evaluation_review" must contain keys: testing_present, limitations, academic_quality.
- "improvement_plan" must be an array of objects with keys: action, why, how, priority.
- "checklist" must be an array of objects with keys: item, done.
- "confidence" must contain keys: mode, overall.
- "confidence.mode" must be exactly "{'restricted' if safe_mode else 'normal'}".
- "confidence.overall" must be a number between 0.0 and 1.0.
- "model_agreement" values must be numbers between 0.0 and 1.0.
- "safety" must contain keys: needs_review, reason.
- No extra keys anywhere.
- Restricted mode does not change the schema. The JSON shape must stay identical to normal mode.
{_student_style_block("code")}
SOFTWARE REVIEW RULES
- Assess the submission as implemented technical work, not as a proposal.
- Name concrete technologies, modules, components, workflows, tests, APIs, data models, dashboards, or security controls when the evidence shows them.
- Distinguish clearly between implemented features, partial work, and future work.
- Keep each issue tied to a specific technical gap, risk, or maintenance consequence.
- Keep each improvement action concrete and immediately actionable.
- Use the evidence field to explain both the weakness and the likely fix direction where possible.
- Make architecture_review, implementation_review, and evaluation_review non-redundant.
- Do not invent details that are not supported by the submission or retrieved context.
- Do not criticise a desktop or single-user tool for lacking authentication unless the evidence clearly shows a multi-user, networked, or privileged-access requirement.
QUALITY TARGET
- The summary must identify what the submission is trying to build, the overall technical quality, and the most important gap.
- When the evidence supports it, include at least 2 strengths, 2-3 issues, 2-3 improvement actions, and 3 checklist items.
- CRITICAL: In every "evidence" field, name the specific component, module, feature, test, route, model, or omission from the submission that demonstrates the point. Never write an observation that could apply to any project.
- Keep evidence concise and specific. One strong sentence is better than two generic ones.
- Do not default every issue to low severity. Use the severity guidance above.
- If the submission clearly describes a real system, do not leave strengths, issues, improvement_plan, and checklist all empty.
{_student_content_floor("code")}
FORBIDDEN - never write these without a specific explanation tied to the submission:
- "improve the project" -> say what component needs changing and how
- "add more testing" -> say what is untested and what test type would address it
- "consider using X" -> say why X applies to this stack or use case
- "improve security" -> name the specific vulnerability or missing control
- "better documentation" -> say what is undocumented and why it matters for this system
{mode}
ML signals:
{json.dumps({
    "feedback_category": payload.ml.feedback_category,
    "quality_band": payload.ml.quality_band,
    "confidence_0_to_4": payload.ml.confidence_0_to_4
}, ensure_ascii=False, indent=2)}
{rag_section}
Submission evidence digest:
{evidence_block}
{_FINAL_JSON_REMINDER}
BEGIN:
""".strip()
        return prompt_text, excerpt_cap, was_trimmed
    prompt_text = f"""
You are a senior academic assessor at a university with extensive experience evaluating written academic work across disciplines. Your assessments are specific, evidence-grounded, and immediately actionable for the student.
Your task: produce detailed academic-style feedback on the student submission below. The feedback must be specific to this submission, not a generic template.
{_JSON_RULES}
{_STRICT_STUDENT_JSON_OUTPUT_RULES}
You MUST return exactly one JSON object matching this schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}
Extra constraints:
- "summary" must be a plain string giving a concrete overview of the task, overall academic quality, and the main gap (2-4 sentences minimum).
- "strengths" must be an array of objects with keys: title, evidence.
- "issues" must be an array of objects with keys: title, evidence, severity.
- "severity" must be exactly one of: "low", "med", "high".
- "improvement_plan" must be an array of objects with keys: action, why, how, priority.
- "priority" must be an integer from 1 to 10.
- "checklist" must be an array of objects with keys: item, done.
- "done" must be a boolean.
- "model_agreement" values must be numbers between 0.0 and 1.0.
- "confidence" must contain keys: mode, overall.
- "safety" must contain keys: needs_review, reason.
- No extra keys anywhere.
- Restricted mode does not change the schema. The JSON shape must stay identical to normal mode.
{_student_style_block("essay")}
REASONING GUIDANCE - think through these before writing the JSON:
1. What is the submission's apparent task, question, or objective?
2. Does the submission directly address that task, or does it drift?
3. What are the 3-5 most important specific weaknesses? What textual evidence supports each?
4. What are the most impactful improvements the student could make?
5. Is the argument or analysis backed by sources, reasoning, or examples, or is it asserted without support?
Focus on:
- how well the submission addresses its apparent task, question, or objective
- structure, coherence, paragraphing, and logical flow
- clarity of argument, explanation, or analysis
- use of evidence, examples, sources, or supporting detail
- critical thinking, depth of evaluation, and originality of reasoning
- academic writing quality, precision, and tone
- citation, referencing, or attribution quality when relevant
- methodology, technical substance, or evaluation detail if the work appears practical or research-based
Grounding rules:
- Use the retrieved academic guidance when identifying issues and improvement actions.
- Prefer evidence-backed advice over generic advice.
- Do not quote or reference academic rules that are absent from the retrieved context.
- The "evidence" field should refer to the submission content and may align with the retrieved guidance where relevant.
- Do not fabricate citations inside the JSON fields.
- Keep the tone academic rather than technical-review oriented unless the submission itself is explicitly a software implementation report.
- Do not use generic software-review language such as "implementation quality", "separation of concerns", or "authentication" for essay-style work unless the evidence clearly requires it.
Output quality requirements:
- Make the feedback specific to the actual submission, not a generic template.
- The summary must identify what the submission is trying to do, the overall academic quality, and the most important gap.
- When the evidence supports it, include at least 2 strengths, 2-3 issues, 2-3 improvement actions, and 3 checklist items.
- Each "evidence" field should be 1-2 precise sentences: identify the specific problem, explain why it matters academically, and imply the revision direction.
- CRITICAL: In every "evidence" field, anchor the claim to the actual submission — name the specific paragraph, section, phrase, or passage that demonstrates the issue or strength. Never write an observation that could apply to any submission.
- Keep checklist items short, practical, and directly actionable.
- Do not grade the work or invent a rubric unless one is clearly grounded in the provided context.
- Cover multiple dimensions where possible: task response, structure, evidence use, analysis depth, clarity, and referencing.
- Do not default every issue to low severity. Use the severity guidance above.
{_student_content_floor("essay")}
FORBIDDEN - never write these without a specific explanation tied to the submission:
- "improve clarity" -> say exactly what sentence or section is unclear and why
- "add more detail" -> say what specific detail or evidence is missing and where
- "consider using X" -> say why X is relevant to this particular submission
- "needs more analysis" -> say what analytical angle is absent and what it would reveal
- "good structure" -> say what specific structural feature works well and why
{mode}
ML signals:
{json.dumps({
    "feedback_category": payload.ml.feedback_category,
    "quality_band": payload.ml.quality_band,
    "confidence_0_to_4": payload.ml.confidence_0_to_4
}, ensure_ascii=False, indent=2)}
{rag_section}
Submission evidence digest:
{evidence_block}
{_FINAL_JSON_REMINDER}
BEGIN:
""".strip()
    return prompt_text, excerpt_cap, was_trimmed


def professor_prompt(payload: ProfessorReportIn, needs_review: bool) -> str:
    analysis_type = str(_safe_get(payload, "analysis_type", "") or "").strip().lower()
    project_review = analysis_type == "professor_project_review"
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

    rag = _safe_get(payload, "rag", None)
    grounding_context = _safe_get(payload, "grounding_context", "")
    grounding_instruction = _safe_get(payload, "grounding_instruction", "")
    grounding_citations = _safe_get(payload, "grounding_citations", []) or []
    grounding_retrieved_chunks = _safe_get(payload, "grounding_retrieved_chunks", []) or []
    retrieval_confidence_score = _safe_get(payload, "retrieval_confidence_score", 0.0)
    retrieval_confidence_label = _safe_get(payload, "retrieval_confidence_label", "low")
    retrieval_safe_review = _safe_get(payload, "retrieval_safe_review", False)

    if isinstance(rag, dict):
        grounding_context = rag.get("context", grounding_context)
        grounding_instruction = rag.get("instruction", grounding_instruction)
        grounding_citations = rag.get("citations", grounding_citations)
        grounding_retrieved_chunks = rag.get("retrieved_chunks", grounding_retrieved_chunks)
        retrieval_confidence_score = rag.get("confidence_score", retrieval_confidence_score)
        retrieval_confidence_label = rag.get("confidence_label", retrieval_confidence_label)
        retrieval_safe_review = rag.get("safe_review", retrieval_safe_review)

    rag_section = build_rag_section(
        context=grounding_context,
        instruction=grounding_instruction,
        citations=grounding_citations,
        retrieved_chunks=grounding_retrieved_chunks,
        confidence_label=retrieval_confidence_label,
        confidence_score=retrieval_confidence_score,
        safe_review=retrieval_safe_review,
    )

    effective_review_mode = bool(needs_review or retrieval_safe_review or str(retrieval_confidence_label).lower() == "low")

    mode = (
        """
REVIEW MODE:
- You MUST still populate every required JSON field with real content â€” empty strings and null values are not acceptable.
- rubric_breakdown, feedback_explanation, and moderation_notes must all contain substantive content even in review mode.
- Focus on moderation risks and uncertainty; qualify claims but do not omit them.
- Set safety.needs_review to true.
- Explain the reason briefly in safety.reason (1â€“2 sentences).
- If rubric grounding is weak, say that manual review is recommended â€” but still provide your best judgment.
- Do not overstate band certainty when the retrieved rubric/policy evidence is limited.
"""
        if effective_review_mode
        else
        """
NORMAL MODE:
- Be rubric-focused and moderation-safe.
- Set safety.needs_review to false unless there is a strong reason.
- Justify criterion-level decisions using the retrieved rubric/policy guidance where possible.
- Keep moderation notes precise and defensible.
"""
    ).strip()

    if project_review:
        return f"""
You are a senior university professor and external examiner with deep expertise in software engineering and computing project assessment. Your rubric judgments are known for being technically grounded, defensible under moderation, and specific to the actual project evidence.

Your task: produce detailed rubric-style, moderation-safe feedback for the student computing project submission below.

{_JSON_RULES}

You MUST return exactly one JSON object matching this schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}

Extra constraints:
- "rubric_breakdown" must be an array of objects with keys: criterion, band, justification.
- Include at least 4 rubric rows when the submission provides enough evidence.
- Use criterion names that fit project assessment, such as: project scope and aims, technical implementation, architecture quality, data design, testing and evaluation, security and usability, or academic quality.
- Each justification must be 2â€“4 sentences: one describing what the evidence shows, one explaining what the band boundary means, one noting any gaps or risks.
- "feedback_explanation" must be at least 3 substantive paragraphs: (1) overall technical and academic quality, (2) key strengths and weaknesses with specific evidence, (3) moderation considerations and recommendation for the marker.
- "moderation_notes" must be an array of objects with keys: risk, note.
- Include at least 2 moderation notes when uncertainty, inconsistency, or weak evidence is present.
- "safety" must contain keys: needs_review, reason.
- No extra keys anywhere.

REASONING GUIDANCE â€” think through these before writing the JSON:
1. What band does the project most clearly fit, based on the technical evidence?
2. Which criterion is the hardest to judge and why?
3. Are there any marking risks (e.g., over-claiming implementation, missing testing, unclear scope)?
4. What would a second marker need to verify?

Focus on:
- project aim and scope
- appropriateness of the technical stack
- backend architecture, services, routing, and data flow
- frontend/interface quality and interaction design
- database design, persistence, and data modelling
- authentication, authorization, and security controls
- external APIs, data ingestion, analytics, or AI features
- testing evidence, evaluation quality, limitations, and future improvements
- moderation safety and consistency of judgment

Grounding rules:
- Use the retrieved rubric, marking policy, moderation, and academic guidance when assigning criterion bands and writing justifications.
- Prefer official, policy-like, or rubric-aligned evidence where available.
- Do not invent rubric criteria, assessment rules, moderation policy, or project details not supported by the submission or retrieved sources.
- Keep justifications concrete and tied to the actual project artefacts, not generic computing advice.
- Distinguish clearly between implemented functionality, partially implemented work, and proposed future work.
- If evidence is weak, uncertain, or incomplete, keep the output useful but moderation-safe and explain the risk in moderation_notes and safety.reason.

Output quality requirements:
- Each rubric justification must cite a concrete technical component or a clear omission from the project.
- Prefer criterion-level comments that would help a marker defend the band to an external examiner.
- Keep moderation notes practical, specific, and oriented toward review risk.
- Use 4 to 6 rubric rows covering different dimensions â€” do not repeat the same issue with different wording.
- The feedback_explanation must not be a bullet list â€” write it as connected academic prose.

FORBIDDEN â€” never write these without specific project evidence:
- "the project could be improved" â†’ say what specifically is weak and why it affects the band
- "good implementation" â†’ name the component and what makes it technically sound
- "testing is present" â†’ name what is tested and what coverage or method is used
- "security is adequate" â†’ name the specific control (auth mechanism, input validation, etc.)

{mode}

ML signals:
{json.dumps({
    "rubric_band": payload.ml.rubric_band,
    "argument_depth": payload.ml.argument_depth,
    "moderation_consistency": payload.ml.moderation_consistency
}, ensure_ascii=False, indent=2)}

{rag_section}

Submission content:
{_compact_ingestion(payload.ingestion)}

{_FINAL_JSON_REMINDER}
BEGIN:
""".strip()

    return f"""
You are a senior university professor and experienced academic assessor. Your rubric judgments are specific, evidence-grounded, and written to survive moderation and external examination. You do not produce generic summaries â€” every claim is tied to something in the submission.

Your task: produce detailed, moderation-safe academic assessment feedback for the submission below.

{_JSON_RULES}

You MUST return exactly one JSON object matching this schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}

Extra constraints:
- "rubric_breakdown" must be an array of objects with keys: criterion, band, justification.
- Each justification must be 2â€“4 sentences: what the evidence shows, why it places the work at that band, and any moderation risk.
- "feedback_explanation" must be at least 3 substantive paragraphs written as academic prose (not bullet points): (1) overall argument, structure, and academic quality, (2) evidence use, critical analysis, and referencing, (3) moderation considerations and recommendation.
- "moderation_notes" must be an array of objects with keys: risk, note.
- "safety" must contain keys: needs_review, reason.
- No extra keys anywhere.

REASONING GUIDANCE â€” think through these before writing the JSON:
1. What is the submission's central argument or task â€” and does the work actually fulfil it?
2. Which criterion is the most uncertain and why?
3. What would a second marker need to scrutinise to agree with this judgment?
4. Is the band defensible if challenged at moderation?

Focus on:
- how well the submission meets plausible academic expectations for structure, argument, evidence, and clarity
- rubric-aligned strengths and weaknesses
- consistency and defensibility of any band-style judgments
- feedback phrasing that is academically appropriate and moderation-safe
- situations where the evidence is insufficient for a strong judgment
- whether the work demonstrates critical analysis, academic depth, and appropriate use of evidence
- whether referencing, structure, and coherence are strong enough to support the judgment

Grounding rules:
- Use the retrieved rubric/policy/moderation guidance when assigning criterion bands and justifications.
- Do not invent rubric criteria, marking policy rules, moderation procedures, or official guidance that are not supported by the retrieved context.
- Keep justifications aligned with the retrieved evidence and the submission content.
- If the retrieved grounding is weak, uncertain, or incomplete, set safety.needs_review to true and explain why.
- Do not fabricate citations inside the JSON fields.

Output quality requirements:
- Provide at least 3 rubric rows when the submission contains enough evidence â€” cover different dimensions (structure, argument, evidence, analysis, clarity, referencing).
- Make each justification specific and defensible â€” a second marker should be able to verify it from the submission.
- Avoid empty moderation notes; use them to explain real uncertainty, inconsistency, or moderation risk.
- Do not overstate certainty when the retrieved grounding is weak.

FORBIDDEN â€” never write these without a specific explanation:
- "good structure" â†’ say what structural feature is strong and why it works
- "needs more analysis" â†’ say what analytical angle is missing and what it would add
- "lacks evidence" â†’ say what specific claim is unsupported and what evidence would fix it
- "well-argued" â†’ say what makes the argument logically sound or persuasive
- "could be improved" â†’ say what exactly needs changing and how

{mode}

ML signals:
{json.dumps({
    "rubric_band": payload.ml.rubric_band,
    "argument_depth": payload.ml.argument_depth,
    "moderation_consistency": payload.ml.moderation_consistency
}, ensure_ascii=False, indent=2)}

{rag_section}

Submission content:
{_compact_ingestion(payload.ingestion)}

{_FINAL_JSON_REMINDER}
BEGIN:
""".strip()


def fix_json_prompt(
    bad_output: str,
    target: str,
    *,
    forced_confidence_mode: str | None = None,
    forced_needs_review: bool | None = None,
) -> str:
    if target == "student_project_review":
        schema = _student_project_schema()
    elif target == "student":
        schema = _student_standard_schema()
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

    extra_rules: list[str] = []
    if target == "student_project_review":
        extra_rules.extend(
            [
                '- Return exactly these top-level keys: "summary", "issues", "strengths", "architecture_review", "implementation_review", "evaluation_review", "improvement_plan", "checklist", "confidence", "model_agreement", "safety".',
                '- The response must start with { and end with }.',
                '- Output JSON only with no markdown and no commentary.',
                '- "architecture_review" must contain exactly: "overview", "backend", "frontend", "database", "security".',
                '- "implementation_review" must contain exactly: "features_built", "technical_quality", "integration_quality".',
                '- "evaluation_review" must contain exactly: "testing_present", "limitations", "academic_quality".',
                '- "confidence" must contain exactly: "mode", "overall".',
                '- "model_agreement" must contain exactly: "ml_confidence", "llm_confidence", "final_confidence".',
                '- "safety" must contain exactly: "needs_review", "reason".',
                '- Every issue severity must be exactly one of: "low", "med", "high".',
                '- Keep the restricted-mode schema identical to normal mode; only values may change.',
                '- Do not replace nested objects with strings.',
                '- Use [] for empty arrays instead of omitting the field.',
            ]
        )
    elif target == "student":
        extra_rules.extend(
            [
                '- Return exactly these top-level keys: "summary", "strengths", "issues", "improvement_plan", "checklist", "confidence", "model_agreement", "safety".',
                '- The response must start with { and end with }.',
                '- Output JSON only with no markdown and no commentary.',
                '- Every issue severity must be exactly one of: "low", "med", "high".',
                '- "confidence" must contain exactly: "mode", "overall".',
                '- "model_agreement" must contain exactly: "ml_confidence", "llm_confidence", "final_confidence".',
                '- "safety" must contain exactly: "needs_review", "reason".',
                '- Use [] for empty arrays instead of omitting the field.',
            ]
        )
    else:
        extra_rules.extend(
            [
                '- The response must start with { and end with }.',
                '- Output JSON only with no markdown and no commentary.',
            ]
        )
    if forced_confidence_mode:
        extra_rules.append(
            f'- If the schema contains "confidence", then "confidence.mode" must be exactly "{forced_confidence_mode}".'
        )
    if forced_needs_review is not None:
        extra_rules.append(
            f'- "safety.needs_review" must be {str(bool(forced_needs_review)).lower()}.'
        )

    extra_rules_block = "\n".join(extra_rules)

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
{extra_rules_block}

Broken output:
{bad_output}
""".strip()
