from __future__ import annotations

import hmac
import os
import tempfile
import threading
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, status, UploadFile, File

from app.api.routes import api_router  # ✅ your /api router

load_dotenv()

AI_SERVICE_SECRET = os.getenv("AI_SERVICE_SECRET", "change_me_dev_only")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_PATH", "").strip()
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", "30000000"))
MAX_SEGMENTS_RETURN = int(os.getenv("MAX_SEGMENTS_RETURN", "2000"))

ALLOWED_AUDIO_CONTENT_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/mp4",
    "audio/x-m4a",
    "audio/aac",
    "audio/flac",
    "audio/ogg",
    "audio/webm",
    "application/octet-stream",
}

app = FastAPI(title="EduAIPlatform AI Service")

# ✅ Mount Phase 6 APIs here
app.include_router(api_router, prefix="/api")


_model = None
_model_error: Optional[str] = None
_model_lock = threading.Lock()


def verify_secret(x_ai_service_secret: str) -> None:
    if not x_ai_service_secret or not hmac.compare_digest(x_ai_service_secret, AI_SERVICE_SECRET):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid AI service secret")


def _model_id() -> str:
    return WHISPER_MODEL_PATH if WHISPER_MODEL_PATH else WHISPER_MODEL


def _try_load_model() -> None:
    global _model, _model_error
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        _model = None
        _model_error = f"faster-whisper import failed: {e}"
        print(f"[ai-service] {_model_error}")
        return

    try:
        mid = _model_id()
        _model = WhisperModel(mid, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
        _model_error = None
        print(f"[ai-service] Whisper model loaded: {mid} ({WHISPER_DEVICE}/{WHISPER_COMPUTE_TYPE})")
    except Exception as e:
        _model = None
        _model_error = str(e)
        print(f"[ai-service] Whisper model load failed (service will stay up): {_model_error}")


def _get_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            _try_load_model()
    if _model is None:
        raise HTTPException(status_code=503, detail=f"Whisper model not ready: {_model_error}")
    return _model


@app.get("/health")
def health():
    return {
        "status": "ok",
        "phase6_router_loaded": True,
        "phase6_router_error": None,
        "whisper_loaded": _model is not None,
        "model_id": _model_id(),
        "model_error": _model_error,
    }


@app.on_event("startup")
def startup():
    _try_load_model()


@app.post("/v1/feedback")
async def feedback(payload: dict, x_ai_service_secret: str = Header(default="", alias="X-AI-SERVICE-SECRET")):
    verify_secret(x_ai_service_secret)

    ctx = payload.get("context", {})
    policy = ctx.get("policy", {})

    if not policy.get("no_external_browsing", True):
        raise HTTPException(status_code=400, detail="Policy violation: external browsing must be disabled")
    if not policy.get("no_cross_user_access", True):
        raise HTTPException(status_code=400, detail="Policy violation: cross-user access must be disabled")

    text = payload.get("input", {}).get("assignment_text", "")

    tips = []
    if len(text) < 200:
        tips.append("Your submission is quite short—consider expanding with evidence and examples.")
    if "conclusion" not in text.lower():
        tips.append("Consider adding a clear conclusion summarizing your key points.")
    if "reference" not in text.lower() and "bibliograph" not in text.lower():
        tips.append("If required, include references/citations to support your claims.")

    return {
        "policy_enforced": True,
        "feedback": {"summary": "Automated writing feedback (assistive, not a grade).", "tips": tips[:5]},
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    x_ai_service_secret: str = Header(default="", alias="X-AI-SERVICE-SECRET"),
):
    verify_secret(x_ai_service_secret)

    model = _get_model()

    ctype = (file.content_type or "application/octet-stream").lower()
    if ctype not in ALLOWED_AUDIO_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported audio content-type: {ctype}")

    blob = await file.read()
    if len(blob) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio too large")

    suffix = ""
    if file.filename and "." in file.filename:
        suffix = "." + file.filename.rsplit(".", 1)[-1].lower()

    with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
        tmp.write(blob)
        tmp.flush()

        segments, info = model.transcribe(tmp.name, beam_size=3, vad_filter=True)

        text_parts: List[str] = []
        seg_meta: List[Dict[str, Any]] = []

        for s in segments:
            t = (getattr(s, "text", "") or "").strip()
            if t:
                text_parts.append(t)
            seg_meta.append(
                {"start": float(getattr(s, "start", 0.0)), "end": float(getattr(s, "end", 0.0)), "text": t}
            )

        full_text = " ".join(text_parts).strip()

        return {
            "text": full_text,
            "meta": {
                "model": _model_id(),
                "device": WHISPER_DEVICE,
                "compute_type": WHISPER_COMPUTE_TYPE,
                "language": getattr(info, "language", None),
                "language_probability": float(getattr(info, "language_probability", 0.0) or 0.0),
                "bytes": len(blob),
                "filename": file.filename,
                "content_type": ctype,
                "segments_count": len(seg_meta),
            },
            "segments": seg_meta[:MAX_SEGMENTS_RETURN],
        }