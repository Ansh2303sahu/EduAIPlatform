from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import httpx

# NOTE:
# - ModelBundle + load_model_by_extension come from your app/registry/loader.py
# - Ensure loader.py returns a ModelBundle (recommended). If your loader.py currently
#   returns a raw session/model, update loader.py accordingly.
from app.registry.loader import ModelBundle, load_model_by_extension

# Root folder where models are stored in-container
# Your project stores models under /app/app/models (based on your screenshot),
# so default to that. You can override with MODELS_DIR env var.
# Prefer MODELS_DIR if set, otherwise pick the correct default for your repo mount
MODELS_ROOT = Path(os.getenv("MODELS_DIR") or "/app/app/models")


def get_model_dir(role: str, model_name: str, version: str) -> Path:
    return MODELS_ROOT / role / model_name / version


def get_model_path(role: str, model_name: str, version: str, filename: str = "model.onnx") -> Path:
    """
    Used by infer_* routers to locate model artifacts on disk.
    Default filename is model.onnx (and model.onnx.data lives alongside it).
    """
    return get_model_dir(role, model_name, version) / filename


def ensure_model_dir(*, role: str, model_name: str, version: str) -> Path:
    p = get_model_dir(role, model_name, version)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_metadata(model_dir: Path, metadata: Dict[str, Any]) -> Path:
    meta_path = model_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return meta_path


def _supabase_url() -> Optional[str]:
    v = os.getenv("SUPABASE_URL", "").strip()
    return v or None


def _supabase_key() -> Optional[str]:
    v = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    return v or None


def _headers() -> dict[str, str]:
    key = _supabase_key() or ""
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=representation",
    }


def _upsert_model_registry(payload: dict[str, Any]) -> None:
    base = _supabase_url()
    key = _supabase_key()
    if not base or not key:
        # Supabase not configured - allow local filesystem-only usage
        return

    # on_conflict must match your unique index: (role, model_name, version)
    url = f"{base.rstrip('/')}/rest/v1/model_registry?on_conflict=role,model_name,version"
    timeout = httpx.Timeout(30.0, connect=30.0, read=30.0, write=30.0)

    r = httpx.post(url, headers=_headers(), json=payload, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"model_registry upsert failed ({r.status_code}): {r.text}")


def _copy_artifact(src: Path, dst_dir: Path) -> Path:
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"Artifact not found: {src}")
    dst = dst_dir / src.name
    dst.write_bytes(src.read_bytes())
    return dst


def register_model(
    *,
    role: str,
    model_name: str,
    version: str,
    artifact_path: Path,
    metadata: Dict[str, Any],
    extra_artifacts: Sequence[Path] = (),
) -> Path:
    """
    1) Copy artifacts + write metadata.json to filesystem:
         /models/<role>/<model_name>/<version>/
    2) Upsert registry row to Supabase: public.model_registry

    Notes:
    - For ONNX with external data, pass both:
        artifact_path = model.onnx
        extra_artifacts = [model.onnx.data, temperature.json, ...]
    - We store only one artifact_path column in the table; all other paths go into metadata.
    """
    model_dir = ensure_model_dir(role=role, model_name=model_name, version=version)

    main_dst = _copy_artifact(Path(artifact_path), model_dir)

    extras: list[str] = []
    for p in extra_artifacts:
        dst = _copy_artifact(Path(p), model_dir)
        extras.append(str(dst))

    # Enrich metadata with what we actually saved
    md = dict(metadata or {})
    md.setdefault("role", role)
    md.setdefault("model_name", model_name)
    md.setdefault("version", version)
    md["artifact_path"] = str(main_dst)
    if extras:
        md["extra_artifacts"] = extras

    save_metadata(model_dir, md)

    payload = {
        "role": role,
        "model_name": model_name,
        "version": version,
        "artifact_path": str(main_dst),
        "dataset_version": md.get("dataset_version") or md.get("dataset_path"),
        "metrics": md.get("metrics", {}) or {},
        "metadata": md,
    }

    try:
        _upsert_model_registry(payload)
    except Exception:
        # don't break training if Supabase has a temporary blip
        pass

    return model_dir


def register_multimodal_onnx(
    *,
    role: str,
    model_name: str,
    version: str,
    onnx_path: Path,
    dataset_version: str,
    temperature_json_path: Optional[Path] = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    fusion_type: str = "attn+gated_concat",
    modalities: Optional[list[str]] = None,
    metrics: Optional[dict[str, Any]] = None,
) -> Path:
    """
    Convenience wrapper for multimodal ONNX artifacts:
      - model.onnx
      - model.onnx.data (if present)
      - temperature.json (optional)
    """
    modalities = modalities or ["text", "ocr", "audio", "table"]

    extra: list[Path] = []

    # ONNX external weights file: <model>.onnx.data
    data_path = onnx_path.with_suffix(onnx_path.suffix + ".data")
    if data_path.exists():
        extra.append(data_path)

    # Temperature artifact (your calibrate step writes this)
    if temperature_json_path and Path(temperature_json_path).exists():
        extra.append(Path(temperature_json_path))

    meta = {
        "dataset_version": dataset_version,
        "embedding_model": embedding_model,
        "fusion_type": fusion_type,
        "modalities": modalities,
        "format": "onnx",
        "metrics": metrics or {},
    }

    return register_model(
        role=role,
        model_name=model_name,
        version=version,
        artifact_path=onnx_path,
        metadata=meta,
        extra_artifacts=extra,
    )


def load_registered_model(
    *,
    role: str,
    model_name: str,
    version: str,
    filename: str = "model.onnx",
) -> ModelBundle:
    """
    One-line helper for inference routers:
      bundle = load_registered_model(role="student", model_name="feedback_classifier_multimodal", version="v1")

    Returns a ModelBundle with:
      - model/session
      - metadata
      - temperature (if your loader reads *.temperature.json)
      - extra_artifacts (e.g. model.onnx.data)
    """
    path = get_model_path(role, model_name, version, filename=filename)
    return load_model_by_extension(path)