from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from typing import Any, Dict, Tuple

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.deps import CurrentUser, get_current_user

router = APIRouter(prefix="/phase7", tags=["phase7"])


# -------------------------
# Shared HTTP / Supabase helpers
# -------------------------
SUPABASE_TIMEOUT = httpx.Timeout(
    connect=15.0,
    read=120.0,
    write=30.0,
    pool=30.0,
)

LLM_TIMEOUT = httpx.Timeout(
    connect=15.0,
    read=240.0,
    write=30.0,
    pool=30.0,
)

AI_TIMEOUT = httpx.Timeout(
    connect=15.0,
    read=120.0,
    write=30.0,
    pool=30.0,
)


def _clean_base_url(value: str | None) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""

    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        raw = raw[1:-1].strip()

    raw = raw.rstrip("/")

    if raw.endswith("/rest/v1"):
        raw = raw[:-8].rstrip("/")

    return raw


def _require_supabase_config() -> None:
    if not _clean_base_url(settings.supabase_url):
        raise HTTPException(status_code=500, detail="SUPABASE_URL not configured")
    if not settings.supabase_service_role_key:
        raise HTTPException(status_code=500, detail="SUPABASE_SERVICE_ROLE_KEY not configured")


def _supabase_base() -> str:
    _require_supabase_config()
    return _clean_base_url(settings.supabase_url)


def _service_headers(*, prefer_return: bool = False) -> Dict[str, str]:
    key = settings.supabase_service_role_key
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation" if prefer_return else "return=minimal",
    }


async def _supabase_get(path: str, *, retries: int = 3) -> httpx.Response:
    url = f"{_supabase_base()}/rest/v1/{path}"
    last_exc: Exception | None = None
    backoff = 0.6

    for attempt in range(1, retries + 1):
        try:
            async with httpx.AsyncClient(timeout=SUPABASE_TIMEOUT) as client:
                r = await client.get(url, headers=_service_headers(prefer_return=False))
            return r
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            last_exc = e
            if attempt < retries:
                await asyncio.sleep(backoff)
                backoff *= 2

    raise HTTPException(
        status_code=504,
        detail=f"Supabase GET timeout/error after retries: path={path} error={type(last_exc).__name__}: {last_exc}",
    )


async def _supabase_post(table: str, payload: dict[str, Any], *, retries: int = 3) -> httpx.Response:
    url = f"{_supabase_base()}/rest/v1/{table}"
    last_exc: Exception | None = None
    backoff = 0.6

    for attempt in range(1, retries + 1):
        try:
            async with httpx.AsyncClient(timeout=SUPABASE_TIMEOUT) as client:
                r = await client.post(url, headers=_service_headers(prefer_return=True), json=payload)
            return r
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            last_exc = e
            if attempt < retries:
                await asyncio.sleep(backoff)
                backoff *= 2

    raise HTTPException(
        status_code=504,
        detail=f"Supabase POST timeout/error after retries: table={table} error={type(last_exc).__name__}: {last_exc}",
    )


async def _get_rows(path: str) -> list[dict[str, Any]]:
    r = await _supabase_get(path)
    if r.status_code >= 400:
        raise HTTPException(status_code=400, detail=f"Supabase fetch failed: {r.status_code} {r.text}")
    return r.json() or []


async def _post_row(table: str, payload: dict[str, Any]) -> dict[str, Any]:
    r = await _supabase_post(table, payload)
    if r.status_code >= 400:
        raise HTTPException(status_code=400, detail=f"Supabase insert failed: {r.status_code} {r.text}")

    rows = r.json() or []
    if not rows:
        raise HTTPException(status_code=500, detail="Supabase did not return created row")

    return rows[0]


# -------------------------
# Demo-grade rate limiting (per user)
# -------------------------
_RATE: Dict[str, list[float]] = {}
_RATE_MAX = 15
_RATE_WINDOW = 3600.0


def _rate_limit(user_id: str) -> None:
    now = time.time()
    ts = _RATE.get(user_id, [])
    ts = [x for x in ts if now - x < _RATE_WINDOW]

    if len(ts) >= _RATE_MAX:
        raise HTTPException(status_code=429, detail="Rate limit: too many report generations. Try later.")

    ts.append(now)
    _RATE[user_id] = ts


# -------------------------
# Prompt injection heuristic (demo)
# -------------------------
_INJECTION_PHRASES = [
    "ignore previous instructions",
    "system prompt",
    "developer message",
    "bypass",
    "jailbreak",
    "reveal hidden",
    "do not follow",
    "you are now",
]


def _detect_injection(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in _INJECTION_PHRASES)


def _safe_mode_student(ml: dict) -> dict:
    return {
        "summary": "The system could not confidently generate full feedback for this submission.",
        "issues": [
            "Some content triggered safety checks or low-confidence conditions.",
            "Please review the assignment manually and re-run after removing unrelated instructions.",
        ],
        "improvement_plan": [
            "Ensure the submission contains only academic content relevant to the task.",
            "Remove any instructions aimed at changing system behavior.",
        ],
        "checklist": [
            "Content is relevant to the assignment",
            "No prompt-injection style text",
            "Clear structure and citations (if required)",
        ],
        "confidence": {"mode": "safe", "ml_bucket_0_to_4": ml.get("confidence_0_to_4")},
        "safety": {"needs_review": True},
    }


def _safe_mode_professor(ml: dict) -> dict:
    return {
        "rubric_breakdown": [],
        "summary": "The system could not confidently generate a full rubric report for this submission.",
        "moderation_notes": [
            "Safety checks or low-confidence conditions were triggered.",
            "Please review manually and re-run after removing unrelated instructions.",
        ],
        "confidence": {"mode": "safe"},
        "safety": {"needs_review": True},
    }


# -------------------------
# Hashing / mapping helpers
# -------------------------
def _sha256_json(obj: Any) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _map_quality_band_from_confidence(c04: int) -> str:
    if c04 <= 1:
        return "low"
    if c04 == 2:
        return "med"
    return "high"


def _map_depth(label: str) -> str:
    m = {"shallow": "low", "basic": "med", "medium": "med", "developed": "high", "deep": "high"}
    return m.get((label or "").lower(), "med")


def _map_consistency(label: str) -> str:
    m = {"inconsistent": "low", "mixed": "med", "consistent": "high"}
    return m.get((label or "").lower(), "med")


def _agreement_score_student(conf_0_to_4: int, injected: bool, llm_ok: bool) -> float:
    base = {0: 0.15, 1: 0.30, 2: 0.55, 3: 0.75, 4: 0.90}.get(conf_0_to_4, 0.55)
    if injected:
        base -= 0.25
    if not llm_ok:
        base -= 0.20
    return float(max(0.0, min(1.0, base)))


def _agreement_score_professor(consistency: str, injected: bool, llm_ok: bool) -> float:
    base = {"low": 0.40, "med": 0.70, "high": 0.90}.get(consistency, 0.70)
    if injected:
        base -= 0.25
    if not llm_ok:
        base -= 0.20
    return float(max(0.0, min(1.0, base)))


# -------------------------
# Ownership gate
# -------------------------
async def _load_file(file_id: str, user: CurrentUser) -> dict[str, Any]:
    if user.role == "admin":
        rows = await _get_rows(
            f"files?id=eq.{file_id}&select=id,status,mime_type,submission_id,created_at,user_id&limit=1"
        )
    else:
        rows = await _get_rows(
            f"files?id=eq.{file_id}&user_id=eq.{user.id}&select=id,status,mime_type,submission_id,created_at,user_id&limit=1"
        )

    if not rows:
        raise HTTPException(status_code=404, detail="File not found")

    return rows[0]


# -------------------------
# Build ingestion bundle
# -------------------------
async def _build_ingestion_bundle(file_id: str, user: CurrentUser) -> Dict[str, Any]:
    text_rows = await _get_rows(
        f"extracted_text?file_id=eq.{file_id}&select=redacted_text&order=created_at.desc&limit=1"
        + (f"&user_id=eq.{user.id}" if user.role != "admin" else "")
    )
    text_content = (text_rows[0].get("redacted_text") if text_rows else "") or ""

    tr_rows = await _get_rows(
        f"transcripts?file_id=eq.{file_id}&select=redacted_transcript&order=created_at.desc&limit=1"
        + (f"&user_id=eq.{user.id}" if user.role != "admin" else "")
    )
    audio_transcript = (tr_rows[0].get("redacted_transcript") if tr_rows else "") or ""

    tables = await _get_rows(
        f"extracted_tables?file_id=eq.{file_id}"
        f"&select=table_index,sheet_name,columns,rows"
        f"&order=created_at.desc&limit=25"
        + (f"&user_id=eq.{user.id}" if user.role != "admin" else "")
    )

    return {
        "text_content": text_content,
        "ocr_text": "",
        "audio_transcript": audio_transcript,
        "tables_json": {"tables": tables},
    }


# -------------------------
# Call ai-service
# -------------------------
def _require_ai_service() -> None:
    if not settings.ai_service_url:
        raise HTTPException(status_code=500, detail="AI_SERVICE_URL not set")
    if not settings.ai_service_secret:
        raise HTTPException(status_code=500, detail="AI_SERVICE_SECRET not set")


def _ai_headers(user: CurrentUser) -> Dict[str, str]:
    return {
        "x-ai-secret": settings.ai_service_secret,
        "x-user-id": str(user.id),
        "x-role": str(user.role),
        "Content-Type": "application/json",
    }


async def _call_ai_student_multimodal(user: CurrentUser, ingestion: Dict[str, Any]) -> Dict[str, Any]:
    _require_ai_service()
    payload = {
        "text": ingestion.get("text_content", ""),
        "ocr": ingestion.get("ocr_text", ""),
        "audio": ingestion.get("audio_transcript", ""),
        "table": ingestion.get("tables_json", {}),
    }

    try:
        async with httpx.AsyncClient(timeout=AI_TIMEOUT) as client:
            r1 = await client.post(
                f"{settings.ai_service_url.rstrip('/')}/api/infer/student/feedback_multimodal",
                json=payload,
                headers=_ai_headers(user),
            )
            if r1.status_code >= 400:
                raise HTTPException(status_code=502, detail=f"ai-service feedback failed: {r1.text}")
            feedback = r1.json()

            r2 = await client.post(
                f"{settings.ai_service_url.rstrip('/')}/api/infer/student/confidence_multimodal",
                json=payload,
                headers=_ai_headers(user),
            )
            if r2.status_code >= 400:
                raise HTTPException(status_code=502, detail=f"ai-service confidence failed: {r2.text}")
            conf = r2.json()
    except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError) as e:
        raise HTTPException(status_code=504, detail=f"ai-service timeout/error: {type(e).__name__}: {e}") from e

    p = float((conf.get("prediction") or {}).get("confidence") or 0.0)
    if p < 0.35:
        c04 = 0
    elif p < 0.55:
        c04 = 1
    elif p < 0.70:
        c04 = 2
    elif p < 0.85:
        c04 = 3
    else:
        c04 = 4

    feedback_pred = feedback.get("prediction") or {}
    feedback_category = str(feedback_pred.get("label") or "other")

    return {
        "feedback_category": feedback_category,
        "quality_band": _map_quality_band_from_confidence(c04),
        "confidence_0_to_4": c04,
        "raw": {"feedback": feedback, "confidence": conf},
    }


async def _call_ai_professor_multimodal(user: CurrentUser, ingestion: Dict[str, Any]) -> Dict[str, Any]:
    _require_ai_service()
    payload = {
        "text": ingestion.get("text_content", ""),
        "ocr": ingestion.get("ocr_text", ""),
        "audio": ingestion.get("audio_transcript", ""),
        "table": ingestion.get("tables_json", {}),
    }

    try:
        async with httpx.AsyncClient(timeout=AI_TIMEOUT) as client:
            r = await client.post(
                f"{settings.ai_service_url.rstrip('/')}/api/infer/professor/multimodal/rubric-suite",
                json=payload,
                headers=_ai_headers(user),
            )
    except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError) as e:
        raise HTTPException(status_code=504, detail=f"ai-service timeout/error: {type(e).__name__}: {e}") from e

    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"ai-service rubric suite failed: {r.text}")

    out = r.json()
    preds = out.get("predictions") or {}

    rubric_band = (preds.get("rubric_band") or {}).get("label") or "adequate"
    depth_raw = (preds.get("argument_depth") or {}).get("label") or "medium"
    consistency_raw = (preds.get("moderation_consistency") or {}).get("label") or "mixed"

    return {
        "rubric_band": rubric_band,
        "argument_depth": _map_depth(depth_raw),
        "moderation_consistency": _map_consistency(consistency_raw),
        "raw": out,
        "raw_labels": {"argument_depth": depth_raw, "moderation_consistency": consistency_raw},
    }


# -------------------------
# Call llm-service
# -------------------------
def _require_llm() -> None:
    if not settings.llm_service_url:
        raise HTTPException(status_code=500, detail="LLM_SERVICE_URL not set")
    if not settings.llm_service_secret:
        raise HTTPException(status_code=500, detail="LLM_SERVICE_SECRET not set")


async def _call_llm(endpoint: str, payload: dict[str, Any]) -> Tuple[dict[str, Any], str]:
    _require_llm()

    try:
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            r = await client.post(
                f"{settings.llm_service_url.rstrip('/')}{endpoint}",
                json=payload,
                headers={"x-ai-secret": settings.llm_service_secret},
            )
    except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError) as e:
        raise HTTPException(status_code=504, detail=f"llm-service timeout/error: {type(e).__name__}: {e}") from e

    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"llm-service failed: {r.text}")

    model_used = r.headers.get("x-llm-model-used", "") or ""
    return r.json(), model_used


# -------------------------
# API
# -------------------------
class GenerateIn(BaseModel):
    file_id: str = Field(..., min_length=1)
    force: bool = False


@router.get("/latest/{role}/{file_id}")
async def latest(role: str, file_id: str, user: CurrentUser = Depends(get_current_user)):
    if role not in {"student", "professor"}:
        raise HTTPException(status_code=400, detail="Invalid role")

    await _load_file(file_id, user)

    rows = await _get_rows(
        f"ai_reports?file_id=eq.{file_id}&role=eq.{role}&select=*&order=created_at.desc&limit=1"
    )
    return {"found": bool(rows), "item": rows[0] if rows else None}


@router.post("/student/generate")
async def generate_student(body: GenerateIn, user: CurrentUser = Depends(get_current_user)):
    _rate_limit(str(user.id))

    if user.role not in {"student", "admin"}:
        raise HTTPException(status_code=403, detail="Student access required")

    request_id = str(uuid.uuid4())
    t_all = time.perf_counter()

    file_row = await _load_file(body.file_id, user)

    t0 = time.perf_counter()
    ingestion = await _build_ingestion_bundle(body.file_id, user)
    ingestion_ms = int((time.perf_counter() - t0) * 1000)

    input_hash = _sha256_json(ingestion)

    injected = _detect_injection(
        (ingestion.get("text_content") or "") + " " + (ingestion.get("audio_transcript") or "")
    )

    if not body.force:
        existing = await _get_rows(
            "ai_reports"
            f"?file_id=eq.{body.file_id}&role=eq.student&input_hash=eq.{input_hash}"
            "&select=*&order=created_at.desc&limit=1"
        )
        if existing:
            return {"cached": True, "request_id": request_id, "stored": existing[0]}

    t1 = time.perf_counter()
    ml = await _call_ai_student_multimodal(user, ingestion)
    ai_ms = int((time.perf_counter() - t1) * 1000)

    if injected or int(ml.get("confidence_0_to_4", 2)) <= 0:
        report = _safe_mode_student(ml)
        llm_ms = 0
        llm_model_used = "safe_mode"
    else:
        llm_payload = {
            "submission_id": str(file_row.get("submission_id") or ""),
            "ingestion": ingestion,
            "ml": {
                "feedback_category": ml["feedback_category"],
                "quality_band": ml["quality_band"],
                "confidence_0_to_4": ml["confidence_0_to_4"],
            },
        }

        t2 = time.perf_counter()
        report, llm_model_used = await _call_llm("/llm/student/report", llm_payload)
        llm_ms = int((time.perf_counter() - t2) * 1000)

    total_ms = int((time.perf_counter() - t_all) * 1000)
    llm_ok = isinstance(report, dict) and bool(report)
    agreement = _agreement_score_student(int(ml["confidence_0_to_4"]), injected, llm_ok)

    saved = await _post_row(
        "ai_reports",
        {
            "file_id": body.file_id,
            "submission_id": file_row.get("submission_id"),
            "role": "student",
            "report_json": report,
            "report_hash": _sha256_json(report),
            "prompt_hash": _sha256_json({"role": "student", "template": "v1"}),
            "input_hash": input_hash,
            "model_versions": {
                "request_id": request_id,
                "timings_ms": {
                    "ingestion": ingestion_ms,
                    "ai_service": ai_ms,
                    "llm_service": llm_ms,
                    "total": total_ms,
                },
                "llm_primary": "mistral",
                "llm_fallback": "phi3",
                "llm_model_used": llm_model_used or "unknown",
                "ml_models": {
                    "feedback": "student.feedback_classifier_multimodal.v1",
                    "confidence": "student.confidence_model_multimodal.v1",
                },
                "agreement": {
                    "final_confidence": agreement,
                    "ml_bucket_0_to_4": ml["confidence_0_to_4"],
                    "injected": injected,
                },
            },
            "needs_review": bool((report.get("safety") or {}).get("needs_review", False)) or injected,
        },
    )

    return {"cached": False, "request_id": request_id, "ml": ml, "report": report, "stored": saved}


@router.post("/professor/generate")
async def generate_professor(body: GenerateIn, user: CurrentUser = Depends(get_current_user)):
    _rate_limit(str(user.id))

    if user.role not in {"professor", "admin"}:
        raise HTTPException(status_code=403, detail="Professor access required")

    request_id = str(uuid.uuid4())
    t_all = time.perf_counter()

    file_row = await _load_file(body.file_id, user)

    t0 = time.perf_counter()
    ingestion = await _build_ingestion_bundle(body.file_id, user)
    ingestion_ms = int((time.perf_counter() - t0) * 1000)

    input_hash = _sha256_json(ingestion)

    injected = _detect_injection(
        (ingestion.get("text_content") or "") + " " + (ingestion.get("audio_transcript") or "")
    )

    if not body.force:
        existing = await _get_rows(
            "ai_reports"
            f"?file_id=eq.{body.file_id}&role=eq.professor&input_hash=eq.{input_hash}"
            "&select=*&order=created_at.desc&limit=1"
        )
        if existing:
            return {"cached": True, "request_id": request_id, "stored": existing[0]}

    t1 = time.perf_counter()
    ml = await _call_ai_professor_multimodal(user, ingestion)
    ai_ms = int((time.perf_counter() - t1) * 1000)

    if injected:
        report = _safe_mode_professor(ml)
        llm_ms = 0
        llm_model_used = "safe_mode"
    else:
        llm_payload = {
            "submission_id": str(file_row.get("submission_id") or ""),
            "ingestion": ingestion,
            "ml": {
                "rubric_band": ml["rubric_band"],
                "argument_depth": ml["argument_depth"],
                "moderation_consistency": ml["moderation_consistency"],
            },
        }

        t2 = time.perf_counter()
        report, llm_model_used = await _call_llm("/llm/professor/report", llm_payload)
        llm_ms = int((time.perf_counter() - t2) * 1000)

    total_ms = int((time.perf_counter() - t_all) * 1000)
    llm_ok = isinstance(report, dict) and bool(report)
    agreement = _agreement_score_professor(str(ml["moderation_consistency"]), injected, llm_ok)

    saved = await _post_row(
        "ai_reports",
        {
            "file_id": body.file_id,
            "submission_id": file_row.get("submission_id"),
            "role": "professor",
            "report_json": report,
            "report_hash": _sha256_json(report),
            "prompt_hash": _sha256_json({"role": "professor", "template": "v1"}),
            "input_hash": input_hash,
            "model_versions": {
                "request_id": request_id,
                "timings_ms": {
                    "ingestion": ingestion_ms,
                    "ai_service": ai_ms,
                    "llm_service": llm_ms,
                    "total": total_ms,
                },
                "llm_primary": "mistral",
                "llm_fallback": "phi3",
                "llm_model_used": llm_model_used or "unknown",
                "ml_models": {"rubric_suite": "professor.rubric_suite_multimodal.v1"},
                "agreement": {
                    "final_confidence": agreement,
                    "injected": injected,
                    "consistency": ml["moderation_consistency"],
                },
            },
            "needs_review": bool((report.get("safety") or {}).get("needs_review", False)) or injected,
        },
    )

    return {"cached": False, "request_id": request_id, "ml": ml, "report": report, "stored": saved}


@router.get("/history/{role}")
async def history(role: str, limit: int = 30, user: CurrentUser = Depends(get_current_user)):
    if role not in {"student", "professor"}:
        raise HTTPException(status_code=400, detail="Invalid role")
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="Invalid limit")

    select_cols = "id,file_id,submission_id,role,created_at,needs_review,model_versions"

    if user.role == "admin":
        rows = await _get_rows(
            f"ai_reports?role=eq.{role}&select={select_cols}&order=created_at.desc&limit={limit}"
        )
        return {"items": rows}

    files = await _get_rows(f"files?user_id=eq.{user.id}&select=id&limit=5000")
    allowed = {str(x["id"]) for x in files}

    if not allowed:
        return {"items": []}

    rows = await _get_rows(
        f"ai_reports?role=eq.{role}&select={select_cols}&order=created_at.desc&limit={min(limit * 5, 200)}"
    )
    rows = [r for r in rows if str(r.get("file_id")) in allowed][:limit]
    return {"items": rows}


@router.get("/compare/{role}")
async def compare(role: str, a: str, b: str, user: CurrentUser = Depends(get_current_user)):
    if role not in {"student", "professor"}:
        raise HTTPException(status_code=400, detail="Invalid role")

    rows = await _get_rows(f"ai_reports?id=in.({a},{b})&select=*")
    if len(rows) != 2:
        raise HTTPException(status_code=404, detail="Reports not found")

    by_id = {str(x.get("id")): x for x in rows}
    r1 = by_id.get(a)
    r2 = by_id.get(b)
    if not r1 or not r2:
        raise HTTPException(status_code=404, detail="Reports not found")

    if str(r1.get("role")) != role or str(r2.get("role")) != role:
        raise HTTPException(status_code=400, detail="Role mismatch")

    if user.role != "admin":
        fids = {str(r1.get("file_id")), str(r2.get("file_id"))}
        files = await _get_rows(f"files?user_id=eq.{user.id}&select=id&limit=5000")
        allowed = {str(x['id']) for x in files}
        if not fids.issubset(allowed):
            raise HTTPException(status_code=404, detail="Not found")

    j1 = r1.get("report_json") or {}
    j2 = r2.get("report_json") or {}

    def as_list(x):
        return [str(i) for i in (x or [])]

    issues1 = set(as_list(j1.get("issues")))
    issues2 = set(as_list(j2.get("issues")))
    chk1 = set(as_list(j1.get("checklist")))
    chk2 = set(as_list(j2.get("checklist")))

    return {
        "a": {"id": r1.get("id"), "created_at": r1.get("created_at"), "file_id": r1.get("file_id")},
        "b": {"id": r2.get("id"), "created_at": r2.get("created_at"), "file_id": r2.get("file_id")},
        "diff": {
            "issues_removed": sorted(list(issues1 - issues2)),
            "issues_added": sorted(list(issues2 - issues1)),
            "checklist_removed": sorted(list(chk1 - chk2)),
            "checklist_added": sorted(list(chk2 - chk1)),
            "summary_a": j1.get("summary") or "",
            "summary_b": j2.get("summary") or "",
        },
    }


@router.get("/professor/queue")
async def professor_queue(limit: int = 30, user: CurrentUser = Depends(get_current_user)):
    if user.role not in {"professor", "admin"}:
        raise HTTPException(status_code=403, detail="Professor access required")

    lim = min(max(limit, 1), 200)
    rows = await _get_rows(
        "ai_reports?role=eq.professor&select=id,file_id,created_at,needs_review,model_versions"
        "&order=needs_review.desc,created_at.desc"
        f"&limit={lim}"
    )
    return {"items": rows}