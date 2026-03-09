from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

SRC = Path("datasets/processed/student_feedback_multimodal_v1")
OUT = Path("datasets/processed/student_quality_band_multimodal_v1")

QUALITY_NAMES = ["low", "medium", "high"]

# Remap your 7-class feedback labels -> quality(3)
# You can adjust this mapping anytime.
# Default:
# - low: weaker feedback classes
# - medium: average
# - high: strong feedback classes
QUALITY_MAP_7_TO_3 = {
    0: 0,
    1: 0,
    2: 1,
    3: 1,
    4: 1,
    5: 2,
    6: 2,
}


def _pick(d: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return None


def _as_f32(x: Any, expected_dim: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size != expected_dim:
        raise ValueError(f"Expected dim {expected_dim}, got {arr.size}")
    return arr


def _extract_embeddings(sample: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    text = _pick(sample, ["text_embedding", "text_emb", "text"])
    ocr = _pick(sample, ["ocr_embedding", "ocr_emb", "ocr"])
    audio = _pick(sample, ["audio_embedding", "audio_emb", "audio"])
    table = _pick(sample, ["table_embedding", "table_emb", "table"])

    if text is None or ocr is None or audio is None or table is None:
        raise KeyError(
            f"Missing embeddings keys. Present keys: {sorted(sample.keys())[:50]}"
        )

    text = _as_f32(text, 384)
    ocr = _as_f32(ocr, 384)
    audio = _as_f32(audio, 384)
    table = _as_f32(table, 64)
    return text, ocr, audio, table


def _load_split(name: str) -> List[Dict[str, Any]]:
    p = SRC / f"{name}.pt"
    if not p.exists():
        raise FileNotFoundError(f"Missing split: {p}")
    return torch.load(p, map_location="cpu", mmap=True)


def _save_split(name: str, rows: List[Dict[str, Any]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(rows, OUT / f"{name}.pt")


def main() -> None:
    for split in ["train", "val", "test"]:
        data = _load_split(split)
        out_rows: List[Dict[str, Any]] = []

        for x in data:
            y = int(_pick(x, ["label", "y", "target"]))
            if y not in QUALITY_MAP_7_TO_3:
                continue
            q = QUALITY_MAP_7_TO_3[y]

            text, ocr, audio, table = _extract_embeddings(x)

            out_rows.append(
                {
                    "text_embedding": text,
                    "ocr_embedding": ocr,
                    "audio_embedding": audio,
                    "table_embedding": table,
                    "label": int(q),
                }
            )

        if len(out_rows) == 0:
            raise RuntimeError(f"Built 0 rows for split={split}. Check input dataset structure.")

        _save_split(split, out_rows)
        print(f"✅ {split}: {len(out_rows)} rows")

    meta = {
        "name": "student_quality_band_multimodal_v1",
        "source_dataset": str(SRC),
        "label_map": {"0": "low", "1": "medium", "2": "high"},
        "num_classes": 3,
        "mapping_from_feedback_7": QUALITY_MAP_7_TO_3,
    }
    (OUT / "build_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("✅ Built:", OUT)


if __name__ == "__main__":
    main()