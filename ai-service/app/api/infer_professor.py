from __future__ import annotations

import json
import time
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Optional, Union, List

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, Header
from pydantic import BaseModel, Field
from transformers import DistilBertTokenizerFast

from app.core.security import Roles, require_role, require_service_secret
from app.core.rate_limit import enforce_rate_limit
from app.services.audit_log import audit_log
from app.registry.model_registry import get_model_path, load_registered_model
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


def _read_json(path: Path) -> Optional[dict]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def _load_head_temperatures(d: Path) -> Optional[Dict[str, float]]:
    """
    Load per-head temperatures:
      - temperatures.json  (recommended)
      - fallback to None if not found/invalid
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


def _uncertainty(conf: float, threshold: float) -> Dict[str, Any]:
    if conf < threshold:
        return {"uncertain": True, "reason": "low_confidence", "threshold": threshold}
    return {"uncertain": False, "reason": "ok", "threshold": threshold}


# ----------------------------
# Existing: text-only rubric band predictor
# ----------------------------
def _load_rubric_band_textonly():
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

    enforce_rate_limit(
        route="infer.professor.rubric_band",
        per_minute=120,
        role=x_role,
        user_id=user_id,
        ip=ip,
    )

    t0 = time.time()

    sess, tok, meta = _load_rubric_band_textonly()
    enc = tok(
        body.text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="np",
    )
    inputs = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }
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
        ip=ip,
        user_agent=user_agent,
        metadata={"version": "v1", "input_len": len(body.text), "latency_ms": latency_ms, "confidence": conf},
    )

    return {
        "model": "professor.rubric_band_predictor",
        "version": "v1",
        "prediction": {"label_id": pred, "label": label, "confidence": conf},
        **_uncertainty(conf, threshold=0.60),
    }


# ----------------------------
# NEW: multimodal multitask rubric suite (ONNX)
# Endpoint: /infer/professor/multimodal/rubric-suite
# ----------------------------
def _get_embedder():
    key = "embedder.all-MiniLM-L6-v2"
    if key in _CACHE:
        return _CACHE[key]
    sentence_transformers = import_module("sentence_transformers")
    SentenceTransformer = sentence_transformers.SentenceTransformer

    # Keep CPU-safe inside container. You can switch to "cuda" later if you install torch+cuda in ai-service.
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    _CACHE[key] = emb
    return emb


def _embed_384(embedder, s: str) -> np.ndarray:
    s = (s or "").strip()
    if not s:
        return np.zeros((1, 384), dtype=np.float32)
    v = embedder.encode([s], normalize_embeddings=True)
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    return v.astype(np.float32)


def _mask_for_modalities(text: str, ocr: str, audio: str, table: TableType) -> np.ndarray:
    t_ok = bool((text or "").strip())
    o_ok = bool((ocr or "").strip())
    a_ok = bool((audio or "").strip())
    tb_ok = table is not None and (str(table).strip() != "")
    # ✅ your ONNX expects BOOL mask
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


def _load_rubric_suite():
    """
    Loads:
      - ONNX session
      - metadata.json
      - per-head temperatures.json (optional)
      - single temperature fallback (bundle.temperature)
    """
    key = "professor.rubric_suite_multimodal.v1"
    if key in _CACHE:
        return _CACHE[key]

    bundle = load_registered_model(
        role="professor",
        model_name="rubric_suite_multimodal",
        version="v1",
        filename="model.onnx",
    )
    d = model_dir("professor", "rubric_suite_multimodal", "v1")
    head_temps = _load_head_temperatures(d)

    _CACHE[key] = (bundle, head_temps)
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

    enforce_rate_limit(
        route="infer.professor.rubric_suite_multimodal",
        per_minute=120,
        role=x_role,
        user_id=user_id,
        ip=ip,
    )

    t0 = time.time()

    (bundle, head_temps) = _load_rubric_suite()
    sess = bundle.model
    meta = bundle.metadata or {}
    fallback_temp = float(bundle.temperature or 1.0)

    embedder = _get_embedder()

    text_emb = _embed_384(embedder, body.text)
    ocr_emb = _embed_384(embedder, body.ocr)
    audio_emb = _embed_384(embedder, body.audio)
    table_emb = _table_features_64(body.table)
    mask = _mask_for_modalities(body.text, body.ocr, body.audio, body.table)

    inputs = {
    "text_emb": text_emb.astype(np.float32),
    "ocr_emb": ocr_emb.astype(np.float32),
    "audio_emb": audio_emb.astype(np.float32),
    "table_emb": table_emb.astype(np.float32),
    # ✅ MUST be bool because ONNX expects tensor(bool)
    "mask": mask.astype(np.bool_),
}

    # Multi-output ONNX: expect logits per head
    out = sess.run(None, inputs)

    # We support two layouts:
    # 1) Named outputs in fixed order [rubric_band, argument_depth, moderation_consistency]
    # 2) If metadata has "head_order", use it.
    head_order = meta.get("head_order") or ["rubric_band", "argument_depth", "moderation_consistency"]

    predictions: Dict[str, Any] = {}
    any_uncertain = False

    for i, head_name in enumerate(head_order):
        if i >= len(out):
            continue

        logits = np.asarray(out[i], dtype=np.float32)

        t = fallback_temp
        if head_temps and head_name in head_temps:
            t = float(head_temps[head_name])

        logits = logits / max(t, 1e-6)

        probs = _softmax(logits)
        pred = int(np.argmax(probs, axis=1)[0])
        conf = float(np.max(probs, axis=1)[0])

        # Per-head label maps (optional)
        label_maps = meta.get("label_maps", {}) or {}
        lm = label_maps.get(head_name, {})
        label = lm.get(str(pred), f"label_{pred}")

        # thresholds: stricter for primary head
        thr = 0.60 if head_name == "rubric_band" else 0.55
        u = _uncertainty(conf, threshold=thr)
        any_uncertain = any_uncertain or bool(u["uncertain"])

        predictions[head_name] = {
            "label_id": pred,
            "label": label,
            "confidence": conf,
            "temperature": t,
            "uncertain": u["uncertain"],
            "reason": u["reason"],
        }

    latency_ms = int((time.time() - t0) * 1000)

    background_tasks.add_task(
        audit_log,
        actor_user_id=user_id,
        action="infer.professor.rubric_suite_multimodal",
        ip=ip,
        user_agent=user_agent,
        metadata={
            "version": "v1",
            "latency_ms": latency_ms,
            "mask": [bool(x) for x in mask[0].tolist()],
            "uncertain": any_uncertain,
            "model_path": str(bundle.artifact_path),
        },
    )

    return {
        "model": "professor.rubric_suite_multimodal",
        "version": "v1",
        "predictions": predictions,
        "uncertain": any_uncertain,
        "modalities_used": {
            "text": bool(mask[0, 0]),
            "ocr": bool(mask[0, 1]),
            "audio": bool(mask[0, 2]),
            "table": bool(mask[0, 3]),
        },
    }
