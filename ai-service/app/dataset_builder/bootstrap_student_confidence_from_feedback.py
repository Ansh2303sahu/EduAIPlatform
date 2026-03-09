from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

SRC = Path("datasets/processed/student_feedback_multimodal_v1")
OUT = Path("datasets/processed/student_confidence_multimodal_v1")


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


def _is_present(vec: np.ndarray) -> bool:
    # embeddings were zeroed when modality missing -> norm near 0
    return float(np.linalg.norm(vec)) > 1e-6


def _confidence_0_4(text: np.ndarray, ocr: np.ndarray, audio: np.ndarray, table: np.ndarray) -> int:
    n = 0
    n += 1 if _is_present(text) else 0
    n += 1 if _is_present(ocr) else 0
    n += 1 if _is_present(audio) else 0
    n += 1 if _is_present(table) else 0
    # map count -> 0..4
    return int(max(0, min(4, n)))


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
            text, ocr, audio, table = _extract_embeddings(x)
            c = _confidence_0_4(text, ocr, audio, table)

            out_rows.append(
                {
                    "text_embedding": text,
                    "ocr_embedding": ocr,
                    "audio_embedding": audio,
                    "table_embedding": table,
                    "label": int(c),
                }
            )

        if len(out_rows) == 0:
            raise RuntimeError(f"Built 0 rows for split={split}. Check input dataset structure.")

        _save_split(split, out_rows)
        print(f"✅ {split}: {len(out_rows)} rows")

    meta = {
        "name": "student_confidence_multimodal_v1",
        "source_dataset": str(SRC),
        "label_map": {str(i): str(i) for i in range(5)},
        "num_classes": 5,
        "rule": "confidence = number of present modalities (0..4)",
    }
    (OUT / "build_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("✅ Built:", OUT)


if __name__ == "__main__":
    main()