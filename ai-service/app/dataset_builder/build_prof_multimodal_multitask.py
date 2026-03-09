from __future__ import annotations

import json
from pathlib import Path
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer

from app.dataset_builder.clean_text import clean_text

RAW = Path("datasets/raw/feedback_effectiveness")
OUT = Path("datasets/processed/prof_multimodal_v1")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BAND_MAP = {
    "Ineffective": 0,
    "Adequate": 1,
    "Effective": 2,
}


def compute_argument_depth(text: str) -> int:
    sents = text.count(".")
    length = len(text.split())

    if sents < 2 or length < 40:
        return 0
    if length < 80:
        return 1
    if length < 140:
        return 2
    return 3


def compute_moderation_consistency(text: str) -> int:
    words = text.split()
    unique_ratio = len(set(words)) / max(len(words), 1)

    if unique_ratio < 0.4:
        return 0
    if unique_ratio < 0.6:
        return 1
    return 2


def main():
    model = SentenceTransformer(EMBED_MODEL_NAME)

    df = pd.read_csv(RAW / "train.csv")

    rows = []
    for _, r in df.iterrows():
        txt_raw = r["discourse_text"]
        eff_raw = r["discourse_effectiveness"]

        txt = clean_text("" if pd.isna(txt_raw) else str(txt_raw))
        if not txt:
            continue

        eff = "" if pd.isna(eff_raw) else str(eff_raw).strip()
        if eff not in BAND_MAP:
            continue

        rubric_label = BAND_MAP[eff]
        arg_label = compute_argument_depth(txt)
        mod_label = compute_moderation_consistency(txt)

        text_emb = torch.tensor(model.encode(txt), dtype=torch.float32)
        zero_384 = torch.zeros(384)
        zero_64 = torch.zeros(64)

        rows.append(
            {
                "text_emb": text_emb,
                "ocr_emb": zero_384,
                "audio_emb": zero_384,
                "table_emb": zero_64,
                "mask": torch.tensor([1, 0, 0, 0], dtype=torch.bool),
                "labels": {
                    "rubric_band": torch.tensor(rubric_label),
                    "argument_depth": torch.tensor(arg_label),
                    "moderation_consistency": torch.tensor(mod_label),
                },
            }
        )

    OUT.mkdir(parents=True, exist_ok=True)

    torch.save(rows[: int(0.7 * len(rows))], OUT / "train.pt")
    torch.save(rows[int(0.7 * len(rows)) : int(0.85 * len(rows))], OUT / "val.pt")
    torch.save(rows[int(0.85 * len(rows)) :], OUT / "test.pt")

    meta = {
        "name": "prof_multimodal_v1",
        "tasks": {
            "rubric_band": 3,
            "argument_depth": 4,
            "moderation_consistency": 3,
        },
        "embedding_model": EMBED_MODEL_NAME,
    }

    (OUT / "build_meta.json").write_text(json.dumps(meta, indent=2))

    print("✅ Built:", OUT)


if __name__ == "__main__":
    main()