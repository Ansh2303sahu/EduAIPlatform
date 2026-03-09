from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError
import json
import re

from app.config import settings
from app.schemas import StudentReportIn, ProfessorReportIn, StudentReportOut, ProfessorReportOut
from app.security import sanitize_input
from app.ollama_client import generate_with_fallback
from app.prompts import student_prompt, professor_prompt, fix_json_prompt

app = FastAPI(title="llm-service", version="1.1")


def _check_secret(x_ai_secret: str | None):
    if not settings.service_secret:
        raise HTTPException(status_code=500, detail="LLM_SERVICE_SECRET not set")
    if not x_ai_secret or x_ai_secret != settings.service_secret:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _extract_json_text(raw: str) -> str:
    if not isinstance(raw, str):
        raw = str(raw)

    text = raw.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    start_obj = text.find("{")
    end_obj = text.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        return text[start_obj:end_obj + 1].strip()

    return text


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/llm/student/report", response_model=StudentReportOut)
async def student_report(payload: StudentReportIn, x_ai_secret: str | None = Header(default=None)):
    _check_secret(x_ai_secret)

    combined = "\n".join(
        [
            payload.ingestion.text_content,
            payload.ingestion.ocr_text,
            payload.ingestion.audio_transcript,
        ]
    )
    _, injected, inj_reason = sanitize_input(combined, settings.max_input_chars)

    safe_mode = injected or (payload.ml.quality_band == "low") or (payload.ml.confidence_0_to_4 <= 1)

    gen = await generate_with_fallback(student_prompt(payload, safe_mode=safe_mode))
    model_used = gen["model_used"]
    raw = gen["response"]

    for attempt in range(settings.max_retries + 1):
        try:
            cleaned = _extract_json_text(raw)
            obj = json.loads(cleaned)
            out = StudentReportOut.model_validate(obj)
            return JSONResponse(
                content=out.model_dump(),
                headers={"x-llm-model-used": model_used},
            )
        except (json.JSONDecodeError, ValidationError):
            if attempt >= settings.max_retries:
                raise HTTPException(
                    status_code=502,
                    detail=f"LLM invalid JSON; injected={injected}; reason={inj_reason}",
                )
            gen = await generate_with_fallback(fix_json_prompt(raw, "student"))
            model_used = gen["model_used"]
            raw = gen["response"]


@app.post("/llm/professor/report", response_model=ProfessorReportOut)
async def professor_report(payload: ProfessorReportIn, x_ai_secret: str | None = Header(default=None)):
    _check_secret(x_ai_secret)

    combined = "\n".join(
        [
            payload.ingestion.text_content,
            payload.ingestion.ocr_text,
            payload.ingestion.audio_transcript,
        ]
    )
    _, injected, inj_reason = sanitize_input(combined, settings.max_input_chars)

    needs_review = injected or (payload.ml.moderation_consistency == "low")

    gen = await generate_with_fallback(professor_prompt(payload, needs_review=needs_review))
    model_used = gen["model_used"]
    raw = gen["response"]

    for attempt in range(settings.max_retries + 1):
        try:
            cleaned = _extract_json_text(raw)
            obj = json.loads(cleaned)
            out = ProfessorReportOut.model_validate(obj)
            return JSONResponse(
                content=out.model_dump(),
                headers={"x-llm-model-used": model_used},
            )
        except (json.JSONDecodeError, ValidationError):
            if attempt >= settings.max_retries:
                raise HTTPException(
                    status_code=502,
                    detail=f"LLM invalid JSON; injected={injected}; reason={inj_reason}",
                )
            gen = await generate_with_fallback(fix_json_prompt(raw, "professor"))
            model_used = gen["model_used"]
            raw = gen["response"]
            cleaned = _extract_json_text(raw)
            obj = json.loads(cleaned)
