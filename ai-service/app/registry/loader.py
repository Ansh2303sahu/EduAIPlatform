from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import json
import torch


# ----------------------------
# Bundle for registry usage
# ----------------------------
@dataclass
class ModelBundle:
    artifact_path: Path
    model: Any
    format: str
    metadata: Dict[str, Any]
    temperature: float
    extra_artifacts: Dict[str, Path]


# ----------------------------
# Helpers
# ----------------------------
def _read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _find_metadata(model_path: Path) -> Dict[str, Any]:
    """
    Your registry stores metadata as:
      - metadata.json (common)
    Some tools may write:
      - <stem>.metadata.json
    We support both.
    """
    candidates = [
        model_path.with_name(model_path.stem + ".metadata.json"),
        model_path.parent / "metadata.json",
    ]
    for p in candidates:
        data = _read_json_if_exists(p)
        if isinstance(data, dict):
            return data
    return {}


def _find_temperature(model_path: Path) -> float:
    """
    Supports:
      - temperature.json (preferred in registry)
      - <stem>.temperature.json
      - any *.temperature.json (fallback)
    """
    candidates = [
        model_path.parent / "temperature.json",
        model_path.with_name(model_path.stem + ".temperature.json"),
    ]
    # also accept any *.temperature.json in the folder (e.g. feedback_classifier_multimodal.temperature.json)
    candidates.extend(sorted(model_path.parent.glob("*.temperature.json")))

    for p in candidates:
        data = _read_json_if_exists(p)
        if isinstance(data, dict) and "temperature" in data:
            try:
                return max(float(data["temperature"]), 1e-6)
            except Exception:
                continue
    return 1.0


def _find_onnx_external_data(model_path: Path) -> Dict[str, Path]:
    """
    ONNX external data convention:
      model.onnx
      model.onnx.data
    """
    out: Dict[str, Path] = {}
    data_path = model_path.with_suffix(model_path.suffix + ".data")  # ".onnx.data"
    if data_path.exists():
        out["onnx_data"] = data_path
    return out


# ----------------------------
# Existing functions you already use
# ----------------------------
def load_torch_model(path: Path):
    model = torch.jit.load(str(path))
    model.eval()
    return model


def load_onnx_session(path: Path):
    import onnxruntime as ort
    try:
        return ort.InferenceSession(str(path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    except Exception:
        return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

# ----------------------------
# New unified loader (used by registry)
# ----------------------------
def load_model_by_extension(path: Path) -> ModelBundle:
    suffix = path.suffix.lower()
    metadata = _find_metadata(path)
    temperature = _find_temperature(path)
    extra: Dict[str, Path] = {}

    if suffix in [".pt", ".pth", ".bin"]:
        model = torch.load(path, map_location="cpu")
        return ModelBundle(
            artifact_path=path,
            model=model,
            format="torch",
            metadata=metadata,
            temperature=temperature,
            extra_artifacts=extra,
        )

    if suffix == ".torchscript":
        model = load_torch_model(path)
        return ModelBundle(
            artifact_path=path,
            model=model,
            format="torchscript",
            metadata=metadata,
            temperature=temperature,
            extra_artifacts=extra,
        )

    if suffix == ".onnx":
        if not path.exists():
            raise FileNotFoundError(f"ONNX model not found: {path}")

        extra.update(_find_onnx_external_data(path))
        sess = load_onnx_session(path)

        return ModelBundle(
            artifact_path=path,
            model=sess,
            format="onnx",
            metadata=metadata,
            temperature=temperature,
            extra_artifacts=extra,
        )

    raise ValueError(f"Unsupported model format: {suffix}")