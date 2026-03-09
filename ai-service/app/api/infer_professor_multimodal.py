from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, Header
from pydantic import BaseModel, Field
from transformers import DistilBertTokenizerFast

from app.core.security import Roles, require_role, require_service_secret
from app.core.rate_limit import enforce_rate_limit
from app.services.audit_log import audit_log
from app.registry.model_registry import get_model_path
from app.registry.filesystem_store import load_metadata, model_dir
from app.registry.loader import load_onnx_session

router = APIRouter(prefix="/infer/professor", tags=["infer-professor"])
_CACHE: Dict[str, Any] = {}


# ----------------------------
# Schemas
# ----------------------------
class TextIn(BaseModel):
    text: str = Field(..., min_length=1)


TableType = Union[Dict[str, Any], List[Any], str, None]


class MultimodalIn(BaseModel):
    text: str = Field(default="")
    ocr: str = Field(default="")
    audio: str = Field(default="")
    table: TableType = Field(default=None)


# ----------------------------
# Utils
# ----------------------------
def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def _uncertain(conf: float, thr: float) -> Dict[str, Any]:
    return {"uncertain": bool(conf < thr), "threshold": thr, "reason": ("low_confidence" if conf < thr else "ok")}


def _read_json(path: Path) -> Optional[dict]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def _load_head_temperatures(d: Path) -> Optional[Dict[str, float]]:
    """
    Optional per-head calibration:
      <model_dir>/temperatures.json
      { "rubric_band": 1.1, "argument_depth": 1.3, "moderation_consistency": 1.2 }
    """
    obj = _read_json(d / "temperatures.json")
    if not obj:
        return None
    out: Dict[str, float] = {}
    for k, v in obj.items():
        try:
            out[k] = max(float(v), 1e-6)
        except Exception:
            continue
    return out or None


# ----------------------------
# Existing text-only professor model
# ----------------------------
def _load_rubric_textonly():
    key = "professor.rubric_band_predictor.v1"
    if key in _CACHE:
        return _CACHE[key]

    onnx_path = get_model_path("professor", "rubric_band_predictor", "v1")
    sess = load_onnx_session(onnx_path)
    tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased", local_files_only=True)
    meta = load_metadata(model_dir("professor", "rubric_band_predictor", "v1"))

    _CACHE[key] = (sess, tok, meta)
    return _CACHE[key]


@router.post("/rubric-band")
def rubric_band(
    body: TextIn,
    background_tasks: BackgroundTasks,
    x_user_id: str = Header(default="", alias="X-User-Id"),
    x_role: str = Header(default="", alias="X-Role"),
    x_forwarded_for: str = Header(default="", alias="X-Forwarded-For"),
    user_agent: str = Header(default="", alias="User-Agent"),
    _secret=Depends(require_service_secret),
    _role=Depends(require_role({Roles.PROFESSOR, Roles.ADMIN})),
):
    ip = (x_forwarded_for.split(",")[0].strip() if x_forwarded_for else None) or None
    user_id = (x_user_id.strip() or None)

    enforce_rate_limit(route="infer.professor.rubric_band", per_minute=120, role=x_role, user_id=user_id, ip=ip)

    t0 = time.time()
    sess, tok, meta = _load_rubric_textonly()

    enc = tok(body.text, truncation=True, padding="max_length", max_length=256, return_tensors="np")
    inputs = {"input_ids": enc["input_ids"].astype(np.int64), "attention_mask": enc["attention_mask"].astype(np.int64)}

    logits = sess.run(None, inputs)[0]
    probs = _softmax(logits)
    pred = int(np.argmax(probs, axis=1)[0])
    conf = float(np.max(probs, axis=1)[0])

    label_map = meta.get("label_map", {"0": "ineffective", "1": "adequate", "2": "effective"})
    label = label_map.get(str(pred), f"label_{pred}")

    latency_ms = int((time.time() - t0) * 1000)

    background_tasks.add_task(
        audit_log,
        actor_user_id=user_id,
        action="infer.professor.rubric_band_predictor",
        metadata={"version": "v1", "input_len": len(body.text), "latency_ms": latency_ms, "confidence": conf},
        ip=ip,
        user_agent=user_agent,
    )

    return {
        "model": "professor.rubric_band_predictor",
        "version": "v1",
        "prediction": {"label_id": pred, "label": label, "confidence": conf},
        **_uncertain(conf, thr=0.60),
    }


# ----------------------------
# Multimodal helpers (MiniLM + table features)
# ----------------------------
def _get_embedder():
    key = "embedder.minilm"
    if key in _CACHE:
        return _CACHE[key]
    from sentence_transformers import SentenceTransformer

    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    _CACHE[key] = emb
    return emb


def _embed_384(embedder, s: str) -> np.ndarray:
    s = (s or "").strip()
    if not s:
        return np.zeros((1, 384), dtype=np.float32)
    v = embedder.encode([s], normalize_embeddings=True)
    v = np.asarray(v, dtype=np.float32)
    return v.reshape(1, 384)


def _extract_numbers(obj: Any, out: List[float]) -> None:
    if obj is None:
        return
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if np.isfinite(obj):
            out.append(float(obj))
        return
    if isinstance(obj, str):
        try:
            v = float(obj.strip())
            if np.isfinite(v):
                out.append(v)
        except Exception:
            return
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _extract_numbers(v, out)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            _extract_numbers(v, out)
        return


def _table_features_64(table: TableType) -> np.ndarray:
    nums: List[float] = []
    _extract_numbers(table, nums)
    if not nums:
        return np.zeros((1, 64), dtype=np.float32)

    x = np.asarray(nums, dtype=np.float32)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros((1, 64), dtype=np.float32)

    feats = [
        float(x.size),
        float(np.mean(x)),
        float(np.std(x)),
        float(np.min(x)),
        float(np.max(x)),
        float(np.sum(x)),
        float(np.median(x)),
        float(np.percentile(x, 25)),
        float(np.percentile(x, 75)),
    ]
    vec = np.asarray(feats, dtype=np.float32)
    reps = int(np.ceil(64 / vec.size))
    vec = np.tile(vec, reps)[:64]
    return vec.reshape(1, 64).astype(np.float32)


def _mask(text: str, ocr: str, audio: str, table: TableType) -> np.ndarray:
    return np.asarray(
        [[int(bool((text or "").strip())), int(bool((ocr or "").strip())), int(bool((audio or "").strip())), int(table is not None)]],
        dtype=np.int64,
    )


# ----------------------------
# Multitask multimodal ONNX loader
# ----------------------------
def _load_rubric_suite_multimodal():
    key = "professor.rubric_suite_multimodal.v1"
    if key in _CACHE:
        return _CACHE[key]

    onnx_path = get_model_path("professor", "rubric_suite_multimodal", "v1")  # expects model.onnx
    sess = load_onnx_session(onnx_path)

    d = model_dir("professor", "rubric_suite_multimodal", "v1")
    meta = load_metadata(d)
    head_temps = _load_head_temperatures(d)

    # Determine output order:
    # - if metadata provides head_order use it
    # - else fall back to ONNX output names (logits_<head>)
    head_order = meta.get("head_order")
    if not head_order:
        outs = [o.name for o in sess.get_outputs()]
        # Prefer logits_<name> naming if present
        parsed = []
        for n in outs:
            if n.startswith("logits_"):
                parsed.append(n.replace("logits_", "", 1))
        head_order = parsed or ["rubric_band", "argument_depth", "moderation_consistency"]

    _CACHE[key] = (sess, meta, head_temps, head_order)
    return _CACHE[key]


@router.post("/multimodal/rubric-suite")
def rubric_suite_multimodal(
    body: MultimodalIn,
    background_tasks: BackgroundTasks,
    x_user_id: str = Header(default="", alias="x-user-id"),
    x_role: str = Header(default="", alias="x-role"),
    x_forwarded_for: str = Header(default="", alias="x-forwarded-for"),
    user_agent: str = Header(default="", alias="user-agent"),
    _secret=Depends(require_service_secret),
    _role=Depends(require_role({Roles.PROFESSOR, Roles.ADMIN})),
):
    ip = (x_forwarded_for.split(",")[0].strip() if x_forwarded_for else None) or None
    user_id = (x_user_id.strip() or None)

    enforce_rate_limit(route="infer.professor.rubric_suite_multimodal", per_minute=120, role=x_role, user_id=user_id, ip=ip)

    t0 = time.time()

    sess, meta, head_temps, head_order = _load_rubric_suite_multimodal()
    embedder = _get_embedder()

    text_emb = _embed_384(embedder, body.text)
    ocr_emb = _embed_384(embedder, body.ocr)
    audio_emb = _embed_384(embedder, body.audio)
    table_emb = _table_features_64(body.table)
    m = _mask(body.text, body.ocr, body.audio, body.table)

    inputs = {
        "text_emb": text_emb.astype(np.float32),
        "ocr_emb": ocr_emb.astype(np.float32),
        "audio_emb": audio_emb.astype(np.float32),
        "table_emb": table_emb.astype(np.float32),
        "mask": m.astype(np.int64),  # IMPORTANT: int64, not bool
    }

    outs = sess.run(None, inputs)

    label_maps = meta.get("label_maps", {}) or {}
    predictions: Dict[str, Any] = {}
    any_uncertain = False

    for i, head in enumerate(head_order):
        if i >= len(outs):
            break
        logits = np.asarray(outs[i], dtype=np.float32)

        temp = float(meta.get("temperature", 1.0))
        if head_temps and head in head_temps:
            temp = float(head_temps[head])

        logits = logits / max(temp, 1e-6)
        probs = _softmax(logits)
        pred = int(np.argmax(probs, axis=1)[0])
        conf = float(np.max(probs, axis=1)[0])

        thr = 0.60 if head == "rubric_band" else 0.55
        u = _uncertain(conf, thr)
        any_uncertain = any_uncertain or bool(u["uncertain"])

        lm = label_maps.get(head, {})
        label = lm.get(str(pred), f"label_{pred}")

        predictions[head] = {
            "label_id": pred,
            "label": label,
            "confidence": conf,
            "temperature": temp,
            **u,
        }

    latency_ms = int((time.time() - t0) * 1000)

    background_tasks.add_task(
        audit_log,
        actor_user_id=user_id,
        action="infer.professor.rubric_suite_multimodal",
        metadata={"version": "v1", "latency_ms": latency_ms, "mask": m[0].tolist(), "uncertain": any_uncertain},
        ip=ip,
        user_agent=user_agent,
    )

    return {
        "model": "professor.rubric_suite_multimodal",
        "version": "v1",
        "predictions": predictions,
        "uncertain": any_uncertain,
        "modalities_used": {"text": bool(m[0, 0]), "ocr": bool(m[0, 1]), "audio": bool(m[0, 2]), "table": bool(m[0, 3])},
    }