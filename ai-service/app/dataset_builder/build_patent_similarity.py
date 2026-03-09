from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetDict
from app.dataset_builder.clean_text import clean_text

RAW = Path("datasets/raw/patents")
OUT = Path("datasets/processed/similarity_patent_v1")

def main():
    p = RAW / "train.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")

    df = pd.read_csv(p)

    # Kaggle USPPTM columns typically include: anchor, target, score, context
    needed = ("anchor", "target", "score")
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing '{c}'. Got columns: {list(df.columns)[:30]}")

    rows = []
    for _, r in df.iterrows():
        a = clean_text("" if pd.isna(r["anchor"]) else str(r["anchor"]))
        b = clean_text("" if pd.isna(r["target"]) else str(r["target"]))
        if not a or not b:
            continue
        y = float(r["score"])
        rows.append({"text_a": a, "text_b": b, "label": y, "source": "patent"})

    ds = Dataset.from_list(rows).shuffle(seed=42)

    split = ds.train_test_split(test_size=0.15, seed=42)
    val_test = split["test"].train_test_split(test_size=0.5, seed=42)

    dsd = DatasetDict(train=split["train"], validation=val_test["train"], test=val_test["test"])

    OUT.mkdir(parents=True, exist_ok=True)
    dsd.save_to_disk(str(OUT))

    meta = {
        "name": "similarity_patent_v1",
        "rows_total": len(ds),
        "splits": {k: len(dsd[k]) for k in dsd.keys()},
        "label_type": "regression_score_0_to_1",
        "source_file": str(p),
    }
    (OUT / "build_meta.json").write_text(json.dumps(meta, indent=2))
    print("✅ Built:", OUT)
    print(meta)

if __name__ == "__main__":
    main()