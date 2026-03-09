from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, ClassLabel, Features, Value

from app.dataset_builder.clean_text import clean_text

RAW = Path("datasets/raw/feedback_effectiveness")
OUT = Path("datasets/processed/prof_rubric_band_v1")

# Map effectiveness labels into rubric bands
BAND_NAMES = ["ineffective", "adequate", "effective"]
BAND_MAP = {
    "Ineffective": 0,
    "Adequate": 1,
    "Effective": 2,
}


def main() -> None:
    p = RAW / "train.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")

    df = pd.read_csv(p)

    # Expected columns for Feedback Prize Effectiveness
    required = ("discourse_text", "discourse_effectiveness")
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {p}. Columns: {list(df.columns)[:30]}")

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

        rows.append(
            {
                "text": txt,
                "label": BAND_MAP[eff],
                "source": "feedback_effectiveness",
            }
        )

    if len(rows) < 100:
        raise RuntimeError(f"Too few rows built ({len(rows)}). Check dataset contents/columns.")

    # Make label a ClassLabel so stratify works
    features = Features(
        {
            "text": Value("string"),
            "label": ClassLabel(names=BAND_NAMES),
            "source": Value("string"),
        }
    )

    ds = Dataset.from_list(rows, features=features).shuffle(seed=42)

    split = ds.train_test_split(test_size=0.15, seed=42, stratify_by_column="label")
    val_test = split["test"].train_test_split(test_size=0.5, seed=42, stratify_by_column="label")

    dsd = DatasetDict(
        train=split["train"],
        validation=val_test["train"],
        test=val_test["test"],
    )

    OUT.mkdir(parents=True, exist_ok=True)
    dsd.save_to_disk(str(OUT))

    meta = {
        "name": "prof_rubric_band_v1",
        "rows_total": len(ds),
        "splits": {k: len(dsd[k]) for k in dsd.keys()},
        "label_map": {"0": "ineffective", "1": "adequate", "2": "effective"},
        "source_file": str(p),
    }
    (OUT / "build_meta.json").write_text(json.dumps(meta, indent=2))

    print("✅ Built:", OUT)
    print(meta)


if __name__ == "__main__":
    main()