from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from app.registry.model_registry import register_model

MODEL_NAME = "distilbert-base-uncased"
ROLE = "similarity"
MODEL_KEY = "paraphrase_similarity"
VERSION = "v1"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="binary")),
    }


def train(dataset_path: str) -> None:
    dataset = load_from_disk(dataset_path)
    for split in ("train", "validation"):
        for col in ("text_a", "text_b", "label"):
            if col not in dataset[split].column_names:
                raise ValueError(f"Missing '{col}' in split '{split}'")

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    def tokenize(batch: Dict[str, Any]):
        return tokenizer(
            batch["text_a"],
            batch["text_b"],
            truncation=True,
            padding="max_length",
            max_length=128,  # ✅ pair input; keep smaller for RTX 3050 4GB
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    args = TrainingArguments(
        output_dir="tmp_train_quora",
        num_train_epochs=2,              # start with 2; can push to 3 later
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,   # effective 16
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=[],
        logging_steps=200,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    trainer.train()
    metrics = trainer.evaluate()

    tmp = Path("tmp_model_quora")
    tmp.mkdir(exist_ok=True)
    model.save_pretrained(tmp)
    tokenizer.save_pretrained(tmp)

    # Export to ONNX
    from optimum.exporters.onnx import main_export
    main_export(model_name_or_path=str(tmp), output=tmp, task="sequence-classification")

    artifact = tmp / "model.onnx"
    if not artifact.exists():
        raise RuntimeError("ONNX export failed: model.onnx not found")

    # Pull label_map from build_meta.json if present
    label_map = {"0": "not_duplicate", "1": "duplicate"}
    bm = Path(dataset_path) / "build_meta.json"
    if bm.exists():
        try:
            label_map = json.loads(bm.read_text()).get("label_map", label_map)
        except Exception:
            pass

    metadata: Dict[str, Any] = {
        "model_name": MODEL_KEY,
        "version": VERSION,
        "role": ROLE,
        "framework": "transformers",
        "base_model": MODEL_NAME,
        "metrics": metrics,
        "num_labels": 2,
        "label_map": label_map,
        "dataset_path": dataset_path,
        "artifact": "model.onnx",
    }

    register_model(
        role=ROLE,
        model_name=MODEL_KEY,
        version=VERSION,
        artifact_path=artifact,
        metadata=metadata,
    )

    print("✅ Quora paraphrase model trained + registered.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    args = p.parse_args()
    train(args.dataset_path)


if __name__ == "__main__":
    main()