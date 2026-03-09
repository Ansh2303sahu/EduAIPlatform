from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from datasets import Dataset, DatasetDict

from app.dataset_builder.clean_text import clean_text

RAW = Path("datasets/raw")
OUT = Path("datasets/processed/student_feedback_v1")

# Feedback Effectiveness discourse types (labels)
DISCOURSE_TYPE_MAP = {
    "Lead": 0,
    "Position": 1,
    "Claim": 2,
    "Counterclaim": 3,
    "Rebuttal": 4,
    "Evidence": 5,
    "Concluding Statement": 6,
}
UNKNOWN_LABEL = 7


def map_discourse_type(x: str) -> int:
    x = (x or "").strip()
    return DISCOURSE_TYPE_MAP.get(x, UNKNOWN_LABEL)


def load_feedback_effectiveness() -> List[Dict]:
    path = RAW / "feedback_effectiveness" / "train.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

    df = pd.read_csv(path)

    required = {"discourse_text", "discourse_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"feedback_effectiveness missing columns: {missing}")

    rows: List[Dict] = []
    for _, r in df.iterrows():
        text = clean_text(str(r["discourse_text"]))
        if not text:
            continue
        label = map_discourse_type(str(r["discourse_type"]))
        rows.append(
            {
                "text": text,
                "label": label,
                "source": "feedback_effectiveness",
            }
        )
    return rows


def load_feedback_2021_optional() -> List[Dict]:
    # Feedback Prize 2021 also has discourse_type + discourse_text in train.csv in many versions.
    path = RAW / "feedback_2021" / "train.csv"
    if not path.exists():
        return []

    df = pd.read_csv(path)
    if "discourse_text" not in df.columns or "discourse_type" not in df.columns:
        return []

    rows: List[Dict] = []
    for _, r in df.iterrows():
        text = clean_text(str(r["discourse_text"]))
        if not text:
            continue
        label = map_discourse_type(str(r["discourse_type"]))
        rows.append(
            {
                "text": text,
                "label": label,
                "source": "feedback_2021",
            }
        )
    return rows


def main() -> None:
    all_rows: List[Dict] = []
    all_rows.extend(load_feedback_effectiveness())
    all_rows.extend(load_feedback_2021_optional())

    if len(all_rows) < 1000:
        raise RuntimeError(f"Too few rows built: {len(all_rows)} (check raw files).")

    ds = Dataset.from_list(all_rows).shuffle(seed=42)
    split = ds.train_test_split(test_size=0.15, seed=42)
    train = split["train"]
    val_test = split["test"].train_test_split(test_size=0.5, seed=42)

    dsd = DatasetDict(
        {
            "train": train,
            "validation": val_test["train"],
            "test": val_test["test"],
        }
    )

    OUT.mkdir(parents=True, exist_ok=True)
    dsd.save_to_disk(str(OUT))

    meta = {
        "name": "student_feedback_v1",
        "rows_total": len(ds),
        "splits": {k: len(dsd[k]) for k in dsd.keys()},
        "label_map": DISCOURSE_TYPE_MAP | {"UNKNOWN": UNKNOWN_LABEL},
        "sources": sorted(set([r["source"] for r in all_rows])),
    }
    (OUT / "build_meta.json").write_text(json.dumps(meta, indent=2))

    print("✅ Built dataset:", str(OUT))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
