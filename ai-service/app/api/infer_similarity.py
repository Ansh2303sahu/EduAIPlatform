from __future__ import annotations

import time
from typing import Any, Dict, List, Union

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, Header
from pydantic import BaseModel, Field
from transformers import DistilBertTokenizerFast

from app.core.security import Roles, require_role, require_service_secret
from app.core.rate_limit import enforce_rate_limit
from app.registry.model_registry import get_model_path
from app.registry.filesystem_store import load_metadata, model_dir
from app.registry.loader import load_onnx_session
from app.services.audit_log import audit_log
from app.similarity.embeddings import embed_similarity

router = APIRouter(prefix="/infer/similarity", tags=["infer-similarity"])

_CACHE: Dict[str, Any] = {}


# ----------------------------
# Schemas
# ----------------------------
class SimilarityIn(BaseModel):
    text_a: str = Field(..., min_length=1)
    text_b: str = Field(..., min_length=1)


TableType = Union[Dict[str, Any], List[Any], str, None]


class MultimodalSimilarityIn(BaseModel):
    text_a: str = Field(default="")
    text_b: str = Field(default="")
    audio_a: str = Field(default="")
    audio_b: str = Field(default="")
    table_a: TableType = Field(default=None)
    table_b: TableType = Field(default=None)


# ----------------------------
# Utils
# ----------------------------
def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    na = float(np.linalg.norm(a) + 1e-12)
    nb = float(np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / (na * nb))


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
        return np.zeros((64,), dtype=np.float32)

    x = np.asarray(nums, dtype=np.float32)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros((64,), dtype=np.float32)

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

    return vec.astype(np.float32)


def _get_embedder():
    key = "embedder.all-MiniLM-L6-v2"
    if key in _CACHE:
        return _CACHE[key]
    from sentence_transformers import SentenceTransformer

    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    _CACHE[key] = emb
    return emb


def _embed(embedder, s: str) -> np.ndarray:
    s = (s or "").strip()
    if not s:
        return np.zeros((384,), dtype=np.float32)
    v = embedder.encode([s], normalize_embeddings=True)
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    return v


# ----------------------------
# Existing ONNX similarity models
# ----------------------------
def _load_quora():
    key = "similarity.paraphrase_similarity.v1"
    if key in _CACHE:
        return _CACHE[key]

    model_path = get_model_path("similarity", "paraphrase_similarity", "v1")
    session = load_onnx_session(model_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased", local_files_only=True)
    meta = load_metadata(model_dir("similarity", "paraphrase_similarity", "v1"))

    _CACHE[key] = (session, tokenizer, meta)
    return _CACHE[key]


def _load_patent():
    key = "similarity.patent_phrase_match.v1"
    if key in _CACHE:
        return _CACHE[key]

    path = get_model_path("similarity", "patent_phrase_match", "v1")
    sess = load_onnx_session(path)
    tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased", local_files_only=True)
    meta = load_metadata(model_dir("similarity", "patent_phrase_match", "v1"))

    _CACHE[key] = (sess, tok, meta)
    return _CACHE[key]


@router.post("/paraphrase")
def paraphrase(
    body: SimilarityIn,
    background_tasks: BackgroundTasks,
    x_user_id: str = Header(default="", alias="X-User-Id"),
    x_role: str = Header(default="", alias="X-Role"),
    user_agent: str = Header(default="", alias="User-Agent"),
    x_forwarded_for: str = Header(default="", alias="X-Forwarded-For"),
    _secret=Depends(require_service_secret),
    _role=Depends(require_role({Roles.STUDENT, Roles.PROFESSOR, Roles.ADMIN})),
):
    ip = (x_forwarded_for.split(",")[0].strip() if x_forwarded_for else None) or None
    user_id = (x_user_id.strip() or None)

    enforce_rate_limit(route="infer.similarity.paraphrase", per_minute=90, role=x_role, user_id=user_id, ip=ip)

    t0 = time.time()
    session, tokenizer, meta = _load_quora()

    enc = tokenizer(
        body.text_a,
        body.text_b,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="np",
    )

    inputs = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }

    logits = session.run(None, inputs)[0]
    probs = _softmax(logits)

    pred = int(np.argmax(probs, axis=1)[0])
    duplicate_prob = float(probs[0, 1])

    label_map = meta.get("label_map", {"0": "not_duplicate", "1": "duplicate"})
    label = label_map.get(str(pred), f"label_{pred}")

    latency_ms = int((time.time() - t0) * 1000)

    background_tasks.add_task(
        audit_log,
        actor_user_id=user_id,
        action="infer.similarity.paraphrase_similarity",
        metadata={
            "version": "v1",
            "input_len_a": len(body.text_a),
            "input_len_b": len(body.text_b),
            "latency_ms": latency_ms,
        },
        ip=ip,
        user_agent=user_agent,
    )

    return {
        "model": "similarity.paraphrase_similarity",
        "version": "v1",
        "prediction": {"label_id": pred, "label": label, "duplicate_prob": duplicate_prob},
    }


@router.post("/patent")
def patent_phrase_similarity(
    body: SimilarityIn,
    background_tasks: BackgroundTasks,
    x_user_id: str = Header(default="", alias="X-User-Id"),
    x_role: str = Header(default="", alias="X-Role"),
    user_agent: str = Header(default="", alias="User-Agent"),
    x_forwarded_for: str = Header(default="", alias="X-Forwarded-For"),
    _secret=Depends(require_service_secret),
    _role=Depends(require_role({Roles.STUDENT, Roles.PROFESSOR, Roles.ADMIN})),
):
    ip = (x_forwarded_for.split(",")[0].strip() if x_forwarded_for else None) or None
    user_id = (x_user_id.strip() or None)

    enforce_rate_limit(route="infer.similarity.patent", per_minute=90, role=x_role, user_id=user_id, ip=ip)

    t0 = time.time()
    sess, tok, meta = _load_patent()

    enc = tok(
        body.text_a,
        body.text_b,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="np",
    )
    inputs = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }

    out = sess.run(None, inputs)
    score = float(np.squeeze(out[0]))

    threshold = float(meta.get("threshold", 0.75))
    label = "match" if score >= threshold else "no_match"

    latency_ms = int((time.time() - t0) * 1000)

    background_tasks.add_task(
        audit_log,
        actor_user_id=user_id,
        action="infer.similarity.patent_phrase_match",
        metadata={"version": "v1", "score": score, "threshold": threshold, "latency_ms": latency_ms},
        ip=ip,
        user_agent=user_agent,
    )

    return {
        "model": "similarity.patent_phrase_match",
        "version": "v1",
        "prediction": {"score_0_to_1": score, "threshold": threshold, "label": label},
    }


@router.post("/embedding")
def embedding_similarity_endpoint(
    body: SimilarityIn,
    background_tasks: BackgroundTasks,
    x_user_id: str = Header(default="", alias="X-User-Id"),
    x_role: str = Header(default="", alias="X-Role"),
    user_agent: str = Header(default="", alias="User-Agent"),
    x_forwarded_for: str = Header(default="", alias="X-Forwarded-For"),
    _secret=Depends(require_service_secret),
    _role=Depends(require_role({Roles.STUDENT, Roles.PROFESSOR, Roles.ADMIN})),
):
    ip = (x_forwarded_for.split(",")[0].strip() if x_forwarded_for else None) or None
    user_id = (x_user_id.strip() or None)

    enforce_rate_limit(route="infer.similarity.embedding", per_minute=90, role=x_role, user_id=user_id, ip=ip)

    t0 = time.time()
    score = float(embed_similarity(body.text_a, body.text_b))
    threshold = 0.78
    label = "match" if score >= threshold else "not_match"
    latency_ms = int((time.time() - t0) * 1000)

    background_tasks.add_task(
        audit_log,
        actor_user_id=user_id,
        action="infer.similarity.embedding_baseline",
        metadata={"version": "v1", "score": score, "threshold": threshold, "latency_ms": latency_ms},
        ip=ip,
        user_agent=user_agent,
    )

    return {
        "model": "similarity.embedding_baseline",
        "version": "v1",
        "prediction": {"score_0_to_1": score, "threshold": threshold, "label": label},
    }


# ----------------------------
# Multimodal similarity (no training required)
# Endpoint: /infer/similarity/multimodal
# ----------------------------
@router.post("/multimodal")
def multimodal_similarity(
    body: MultimodalSimilarityIn,
    background_tasks: BackgroundTasks,
    x_user_id: str = Header(default="", alias="x-user-id"),
    x_role: str = Header(default="", alias="x-role"),
    user_agent: str = Header(default="", alias="user-agent"),
    x_forwarded_for: str = Header(default="", alias="x-forwarded-for"),
    _secret=Depends(require_service_secret),
    _role=Depends(require_role({Roles.STUDENT, Roles.PROFESSOR, Roles.ADMIN})),
):
    ip = (x_forwarded_for.split(",")[0].strip() if x_forwarded_for else None) or None
    user_id = (x_user_id.strip() or None)

    enforce_rate_limit(route="infer.similarity.multimodal", per_minute=90, role=x_role, user_id=user_id, ip=ip)

    t0 = time.time()

    embedder = _get_embedder()

    parts: Dict[str, float] = {}
    weights: Dict[str, float] = {}

    # Base weights (renormalized based on provided modalities)
    if (body.text_a or "").strip() and (body.text_b or "").strip():
        ta = _embed(embedder, body.text_a)
        tb = _embed(embedder, body.text_b)
        parts["text"] = _cos(ta, tb)
        weights["text"] = 0.6

    if (body.audio_a or "").strip() and (body.audio_b or "").strip():
        aa = _embed(embedder, body.audio_a)
        ab = _embed(embedder, body.audio_b)
        parts["audio"] = _cos(aa, ab)
        weights["audio"] = 0.2

    if body.table_a is not None and body.table_b is not None:
        fa = _table_features_64(body.table_a)
        fb = _table_features_64(body.table_b)
        parts["table"] = _cos(fa, fb)
        weights["table"] = 0.2

    if not parts:
        return {
            "model": "similarity.multimodal_v1",
            "version": "v1",
            "error": "No comparable modalities provided (need text_a/text_b and/or audio_a/audio_b and/or table_a/table_b).",
            "modalities_used": {"text": False, "audio": False, "table": False},
        }

    # renormalize weights over present modalities
    wsum = float(sum(weights.values()))
    norm_weights = {k: float(weights[k] / wsum) for k in parts.keys()}

    score = 0.0
    for k, v in parts.items():
        score += norm_weights[k] * v

    # cosine [-1,1] -> [0,1]
    similarity_0_1 = float((score + 1.0) / 2.0)

    # industry-grade decision helpers
    threshold = 0.78
    margin = 0.03
    label = "match" if similarity_0_1 >= threshold else "not_match"
    uncertain = abs(similarity_0_1 - threshold) < margin

    latency_ms = int((time.time() - t0) * 1000)

    background_tasks.add_task(
        audit_log,
        actor_user_id=user_id,
        action="infer.similarity.multimodal",
        metadata={
            "version": "v1",
            "similarity_0_to_1": similarity_0_1,
            "label": label,
            "threshold": threshold,
            "uncertain": uncertain,
            "breakdown": parts,
            "weights_used": norm_weights,
            "latency_ms": latency_ms,
        },
        ip=ip,
        user_agent=user_agent,
    )

    return {
        "model": "similarity.multimodal_v1",
        "version": "v1",
        "similarity_0_to_1": similarity_0_1,
        "threshold": threshold,
        "label": label,
        "uncertain": uncertain,
        "breakdown": parts,
        "weights_used": norm_weights,
        "modalities_used": {
            "text": "text" in parts,
            "audio": "audio" in parts,
            "table": "table" in parts,
        },
    }