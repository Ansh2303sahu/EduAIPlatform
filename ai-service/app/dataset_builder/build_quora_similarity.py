from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value

from app.dataset_builder.clean_text import clean_text

RAW = Path("datasets/raw/quora")
OUT = Path("datasets/processed/similarity_quora_v1")


def main() -> None:
    p = RAW / "train.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p} (did you unzip train.csv.zip?)")

    df = pd.read_csv(p)

    # Expected Quora columns
    required = ("question1", "question2", "is_duplicate")
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {p}. Columns: {list(df.columns)[:20]}")

    rows = []
    for _, r in df.iterrows():
        a = clean_text("" if pd.isna(r["question1"]) else str(r["question1"]))
        b = clean_text("" if pd.isna(r["question2"]) else str(r["question2"]))
        if not a or not b:
            continue

        y = int(r["is_duplicate"])
        y = 1 if y == 1 else 0

        rows.append({"text_a": a, "text_b": b, "label": y, "source": "quora"})

    if len(rows) < 1000:
        raise RuntimeError(f"Too few usable rows built: {len(rows)}")

    features = Features(
        {
            "text_a": Value("string"),
            "text_b": Value("string"),
            "label": ClassLabel(names=["not_duplicate", "duplicate"]),
            "source": Value("string"),
        }
    )

    ds = Dataset.from_list(rows, features=features).shuffle(seed=42)

    # Stratified split by label
    split = ds.train_test_split(test_size=0.15, seed=42, stratify_by_column="label")
    val_test = split["test"].train_test_split(test_size=0.5, seed=42, stratify_by_column="label")

    dsd = DatasetDict(
        {
            "train": split["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        }
    )

    OUT.mkdir(parents=True, exist_ok=True)
    dsd.save_to_disk(str(OUT))

    meta = {
        "name": "similarity_quora_v1",
        "rows_total": len(ds),
        "splits": {k: len(dsd[k]) for k in dsd.keys()},
        "label_map": {"0": "not_duplicate", "1": "duplicate"},
        "source_file": str(p),
    }
    (OUT / "build_meta.json").write_text(json.dumps(meta, indent=2))

    print("✅ Built:", OUT)
    print(meta)


if __name__ == "__main__":
    main()