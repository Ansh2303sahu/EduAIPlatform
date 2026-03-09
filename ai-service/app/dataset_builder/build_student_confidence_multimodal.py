from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from app.dataset_builder.clean_text import clean_text
from app.dataset_builder.build_student_feedback_multimodal import build_multimodal_pt

RAW = Path("datasets/raw/student_confidence_multimodal")
OUT = Path("datasets/processed/student_confidence_multimodal_v1")


def _parse_table(v: Any) -> Any:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (dict, list)):
        return v
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return s
    return v


def _label_0_4(v: Any) -> Optional[int]:
    try:
        n = int(v)
        if 0 <= n <= 4:
            return n
    except Exception:
        pass
    return None


def main() -> None:
    p = RAW / "train.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")

    df = pd.read_csv(p)

    # required: either confidence_label OR confidence_score
    if "confidence_label" not in df.columns and "confidence_score" not in df.columns:
        raise ValueError(f"Need 'confidence_label' or 'confidence_score' in {p}. Columns: {list(df.columns)[:50]}")

    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        text = clean_text("" if pd.isna(r.get("text", "")) else str(r.get("text", "")))
        ocr = clean_text("" if pd.isna(r.get("ocr", "")) else str(r.get("ocr", "")))
        audio = clean_text("" if pd.isna(r.get("audio", "")) else str(r.get("audio", "")))
        table = _parse_table(r.get("table", None))

        label: Optional[int] = None

        if "confidence_label" in df.columns:
            label = _label_0_4(r.get("confidence_label"))

        if label is None and "confidence_score" in df.columns:
            try:
                s = float(r.get("confidence_score"))
                # allow 0-100
                if s > 1.0:
                    s = s / 100.0
                s = max(0.0, min(1.0, s))
                label = min(4, int(s * 5.0))
            except Exception:
                label = None

        if label is None:
            continue

        if not (text or ocr or audio or table is not None):
            continue

        rows.append({"text": text, "ocr": ocr, "audio": audio, "table": table, "label": int(label)})

    if len(rows) < 100:
        raise RuntimeError(f"Too few rows built ({len(rows)}). Add more data to {p}")

    OUT.mkdir(parents=True, exist_ok=True)
    build_multimodal_pt(
        rows=rows,
        out_dir=OUT,
        dataset_name="student_confidence_multimodal_v1",
        num_classes=5,
        label_map={str(i): str(i) for i in range(5)},
    )

    print("✅ Built:", OUT)


if __name__ == "__main__":
    main()