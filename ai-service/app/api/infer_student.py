from __future__ import annotations

import json
import time
import os
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Union

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
from app.registry.model_registry import load_registered_model
from app.registry.onnx_multimodal import onnx_predict_multimodal

router = APIRouter(prefix="/infer/student", tags=["infer-student"])

_CACHE: Dict[str, Any] = {}


# ----------------------------
# Request schemas
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


def _read_temperature(model_folder: Path, model_name: str) -> float:
    candidates = [
        model_folder / f"{model_name}.temperature.json",
        model_folder / "temperature.json",
        model_folder / "model.temperature.json",
    ]
    for p in candidates:
        try:
            if p.exists():
                obj = json.loads(p.read_text(encoding="utf-8"))
                if "temperature" in obj:
                    t = float(obj["temperature"])
                    return max(t, 1e-6)
        except Exception:
            pass
    return 1.0


def _ensure_2d_float32(x: Any, dim: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != dim:
        raise ValueError(f"Expected shape [B,{dim}] but got {arr.shape}")
    return arr


def _mask_for_modalities(text: str, ocr: str, audio: str, table: TableType) -> np.ndarray:
    # mask order: [text, ocr, audio, table]
    t_ok = bool((text or "").strip())
    o_ok = bool((ocr or "").strip())
    a_ok = bool((audio or "").strip())
    tb_ok = table is not None and (str(table).strip() != "")
    # IMPORTANT: return BOOL mask for ONNX (model expects tensor(bool))
    return np.asarray([[t_ok, o_ok, a_ok, tb_ok]], dtype=np.bool_)


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

    if len(nums) == 0:
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
    ]

    for q in (1, 5, 10, 25, 75, 90, 95, 99):
        feats.append(float(np.percentile(x, q)))

    centered = x - np.mean(x)
    m2 = float(np.mean(centered**2)) if x.size else 0.0
    m3 = float(np.mean(centered**3)) if x.size else 0.0
    m4 = float(np.mean(centered**4)) if x.size else 0.0
    feats.extend([m2, m3, m4])

    vec = np.asarray(feats, dtype=np.float32)
    if vec.size < 64:
        reps = int(np.ceil(64 / vec.size))
        vec = np.tile(vec, reps)[:64]
    else:
        vec = vec[:64]

    return vec.reshape(1, 64).astype(np.float32)


# ----------------------------
# Label resolution (NEW)
# ----------------------------
@lru_cache(maxsize=128)
def _load_labels_json(path_str: str) -> Dict[int, str]:
    """
    Loads id->label mapping from labels.json (preferred).
    Accepts keys like "0" or 0.
    """
    p = Path(path_str)
    if not p.exists():
        return {}

    try:
        raw = json.loads(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

    out: Dict[int, str] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                continue
    return out


def _resolve_label(*, pred_id: int, meta: dict, labels_path: Path | None = None) -> str:
    """
    Resolve label by priority:
      1) labels.json id->name
      2) metadata.json label_map (string keys)
      3) fallback label_{id}
    """
    if labels_path is not None:
        id2label = _load_labels_json(str(labels_path))
        if pred_id in id2label:
            return id2label[pred_id]

    label_map = meta.get("label_map", {}) if isinstance(meta, dict) else {}
    if isinstance(label_map, dict):
        v = label_map.get(str(pred_id))
        if v:
            return str(v)

    return f"label_{pred_id}"


# ----------------------------
# Legacy (text-only) feedback_classifier v1
# ----------------------------
def _load_feedback():
    key = "student.feedback_classifier.v1"
    if key in _CACHE:
        return _CACHE[key]

    onnx_path = get_model_path("student", "feedback_classifier", "v1")
    sess = load_onnx_session(onnx_path)
    tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased", local_files_only=True)
    meta = load_metadata(model_dir("student", "feedback_classifier", "v1"))

    _CACHE[key] = (sess, tok, meta)
    return _CACHE[key]


@router.post("/feedback")
def feedback_classification(
    body: TextIn,
    background_tasks: BackgroundTasks,
    x_user_id: str = Header(default="", alias="x-user-id"),
    x_role: str = Header(default="", alias="x-role"),
    x_forwarded_for: str = Header(default="", alias="x-forwarded-for"),
    user_agent: str = Header(default="", alias="user-agent"),
    _secret=Depends(require_service_secret),
    _role=Depends(require_role({Roles.STUDENT, Roles.ADMIN})),
):
    ip = (x_forwarded_for.split(",")[0].strip() if x_forwarded_for else None) or None
    user_id = (x_user_id.strip() or None)

    enforce_rate_limit(route="infer.student.feedback", per_minute=60, role=x_role, user_id=user_id, ip=ip)

    t0 = time.time()

    sess, tok, meta = _load_feedback()
    enc = tok(body.text, truncation=True, padding="max_length", max_length=256, return_tensors="np")
    inputs = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }
    logits = sess.run(None, inputs)[0]
    probs = _softmax(logits)
    pred = int(np.argmax(probs, axis=1)[0])
    conf = float(np.max(probs, axis=1)[0])

    # legacy: only metadata label_map exists
    label = _resolve_label(pred_id=pred, meta=meta, labels_path=None)

    latency_ms = int((time.time() - t0) * 1000)

    background_tasks.add_task(
        audit_log,
        actor_user_id=user_id,
        action="infer.student.feedback_classifier",
        ip=ip,
        user_agent=user_agent,
        metadata={"version": "v1", "input_len": len(body.text), "latency_ms": latency_ms, "confidence": conf},
    )

    return {
        "model": "student.feedback_classifier",
        "version": "v1",
        "prediction": {"label_id": pred, "label": label, "confidence": conf},
    }


# ----------------------------
# NEW: Multimodal feedback_classifier_multimodal v1
# ----------------------------
def _get_embedder():
    key = "embedder.all-MiniLM-L6-v2"
    if key in _CACHE:
        return _CACHE[key]

    sentence_transformers = import_module("sentence_transformers")
    SentenceTransformer = sentence_transformers.SentenceTransformer

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    _CACHE[key] = embedder
    return embedder


def _embed_384(embedder, s: str) -> np.ndarray:
    s = (s or "").strip()
    if not s:
        return np.zeros((1, 384), dtype=np.float32)
    v = embedder.encode([s], normalize_embeddings=True)
    return _ensure_2d_float32(v, 384)


def _load_feedback_multimodal():
    key = "student.feedback_classifier_multimodal.v1"
    if key in _CACHE:
        return _CACHE[key]

    model_name = "feedback_classifier_multimodal"
    version = "v1"

    onnx_path = get_model_path("student", model_name, version)
    sess = load_onnx_session(onnx_path)

    d = model_dir("student", model_name, version)
    meta = load_metadata(d)
    temperature = _read_temperature(d, model_name)

    # preferred labels.json path
    labels_path = d / "labels.json"

    _CACHE[key] = (sess, meta, temperature, labels_path)
    return _CACHE[key]


def _load_confidence_multimodal():
    key = "student.confidence_model_multimodal.v1"
    if key in _CACHE:
        return _CACHE[key]

    bundle = load_registered_model(
        role="student",
        model_name="confidence_model_multimodal",
        version="v1",
        filename="model.onnx",
    )
    _CACHE[key] = bundle
    return bundle


def build_multimodal_inputs(body: MultimodalIn):
    embedder = _get_embedder()

    text_emb = _embed_384(embedder, body.text)
    ocr_emb = _embed_384(embedder, body.ocr)
    audio_emb = _embed_384(embedder, body.audio)
    table_emb = _table_features_64(body.table)

    mask = _mask_for_modalities(body.text, body.ocr, body.audio, body.table)

    modalities_used = {
        "text": bool(mask[0, 0]),
        "ocr": bool(mask[0, 1]),
        "audio": bool(mask[0, 2]),
        "table": bool(mask[0, 3]),
    }

    return text_emb, ocr_emb, audio_emb, table_emb, mask, modalities_used


@router.post("/feedback_multimodal")
def feedback_classification_multimodal(
    body: MultimodalIn,
    background_tasks: BackgroundTasks,
    x_user_id: str = Header(default="", alias="x-user-id"),
    x_role: str = Header(default="", alias="x-role"),
    x_forwarded_for: str = Header(default="", alias="x-forwarded-for"),
    user_agent: str = Header(default="", alias="user-agent"),
    _secret=Depends(require_service_secret),
    _role=Depends(require_role({Roles.STUDENT, Roles.ADMIN})),
):
    ip = (x_forwarded_for.split(",")[0].strip() if x_forwarded_for else None) or None
    user_id = (x_user_id.strip() or None)

    enforce_rate_limit(route="infer.student.feedback_multimodal", per_minute=60, role=x_role, user_id=user_id, ip=ip)

    t0 = time.time()

    sess, meta, temperature, labels_path = _load_feedback_multimodal()

    text_emb, ocr_emb, audio_emb, table_emb, mask, modalities_used = build_multimodal_inputs(body)

    inputs = {
        "text_emb": text_emb.astype(np.float32),
        "ocr_emb": ocr_emb.astype(np.float32),
        "audio_emb": audio_emb.astype(np.float32),
        "table_emb": table_emb.astype(np.float32),
        "mask": mask.astype(np.bool_),  # ✅ FIX: bool mask
    }

    logits = sess.run(None, inputs)[0].astype(np.float32)
    logits = logits / max(float(temperature), 1e-6)

    probs = _softmax(logits)
    pred = int(np.argmax(probs, axis=1)[0])
    conf = float(np.max(probs, axis=1)[0])

    # ✅ FIX: use labels.json id->label (preferred), then fallback
    label = _resolve_label(pred_id=pred, meta=meta, labels_path=labels_path)

    latency_ms = int((time.time() - t0) * 1000)

    background_tasks.add_task(
        audit_log,
        actor_user_id=user_id,
        action="infer.student.feedback_classifier_multimodal",
        ip=ip,
        user_agent=user_agent,
        metadata={
            "version": "v1",
            "latency_ms": latency_ms,
            "confidence": conf,
            "mask": [bool(x) for x in mask[0].tolist()],
            "temperature": float(temperature),
        },
    )

    return {
        "model": "student.feedback_classifier_multimodal",
        "version": "v1",
        "prediction": {"label_id": pred, "label": label, "confidence": conf},
        "modalities_used": modalities_used,
    }


@router.post("/confidence_multimodal")
def confidence_multimodal(
    body: MultimodalIn,
    background_tasks: BackgroundTasks,
    x_user_id: str = Header(default="", alias="X-User-Id"),
    x_role: str = Header(default="", alias="X-Role"),
    x_forwarded_for: str = Header(default="", alias="X-Forwarded-For"),
    user_agent: str = Header(default="", alias="User-Agent"),
    _secret=Depends(require_service_secret),
    _role=Depends(require_role({Roles.STUDENT, Roles.ADMIN})),
):
    ip = (x_forwarded_for.split(",")[0].strip() if x_forwarded_for else None) or None
    user_id = (x_user_id.strip() or None)

    enforce_rate_limit(route="infer.student.confidence_multimodal", per_minute=60, role=x_role, user_id=user_id, ip=ip)

    t0 = time.time()
    bundle = _load_confidence_multimodal()

    text_emb, ocr_emb, audio_emb, table_emb, mask, modalities_used = build_multimodal_inputs(body)

    pred, probs = onnx_predict_multimodal(
        bundle.model,
        text_emb=text_emb,
        ocr_emb=ocr_emb,
        audio_emb=audio_emb,
        table_emb=table_emb,
        mask=mask,  # already bool
        temperature=bundle.temperature,
    )

    conf = float(probs[pred])
    latency_ms = int((time.time() - t0) * 1000)

    background_tasks.add_task(
        audit_log,
        actor_user_id=user_id,
        action="infer.student.confidence_model_multimodal",
        ip=ip,
        user_agent=user_agent,
        metadata={"version": "v1", "latency_ms": latency_ms, "confidence": conf},
    )

    return {
        "model": "student.confidence_model_multimodal",
        "version": "v1",
        "prediction": {"label_id": pred, "label": str(pred), "confidence": conf},
        "modalities_used": modalities_used,
    }