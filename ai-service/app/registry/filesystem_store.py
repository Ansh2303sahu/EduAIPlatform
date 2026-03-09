from __future__ import annotations
import json
from pathlib import Path
from typing import Dict

BASE_MODELS_DIR = Path("models")
MODELS_ROOT = Path("/app/models")

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
    return json.loads(p.read_text())


