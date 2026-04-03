from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


def _looks_like_models_root(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    if any((path / role).exists() for role in ("student", "professor", "similarity")):
        return True
    return any(path.glob("*/*/*/metadata.json"))


def resolve_models_root() -> Path:
    env_root = os.getenv("MODELS_DIR", "").strip()
    repo_default = Path(__file__).resolve().parents[1] / "models"
    candidates: list[Path] = []

    if env_root:
        candidates.append(Path(env_root))

    candidates.extend(
        [
            repo_default,
            Path("/app/app/models"),
            Path("/app/models"),
            Path("models"),
        ]
    )

    for candidate in candidates:
        if _looks_like_models_root(candidate):
            return candidate

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


MODELS_ROOT = resolve_models_root()

def ensure_model_dir(*, role: str, model_name: str, version: str) -> Path:
    """
    Ensures /app/models/<role>/<model_name>/<version>/ exists and returns that path.
    """
    p = MODELS_ROOT / role / model_name / version
    p.mkdir(parents=True, exist_ok=True)
    return p


def model_dir(role: str, model_name: str, version: str) -> Path:
    """
    Returns /app/models/<role>/<model_name>/<version> without creating it.
    """
    return MODELS_ROOT / role / model_name / version


def save_metadata(model_dir: Path, metadata: Dict[str, Any]) -> Path:
    """
    Writes metadata.json inside the model directory.
    """
    out = model_dir / "metadata.json"
    out.write_text(json.dumps(metadata, indent=2))
    return out



def load_metadata(model_dir: Path) -> Dict[str, Any]:
    """
    Reads metadata.json from the model directory.
    """
    p = model_dir / "metadata.json"
    return json.loads(p.read_text(encoding="utf-8"))


