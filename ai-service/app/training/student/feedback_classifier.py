from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict
from transformers import EarlyStoppingCallback
import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from app.registry.model_registry import register_model

MODEL_NAME = "distilbert-base-uncased"
ROLE = "student"
MODEL_KEY = "feedback_classifier"
VERSION = "v1"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="weighted")),
    }


def train(dataset_path: str) -> None:
    dataset = load_from_disk(dataset_path)

    # Basic validation
    for split in ("train", "validation"):
        if split not in dataset:
            raise ValueError(f"Dataset missing split: {split}")
        for col in ("text", "label"):
            if col not in dataset[split].column_names:
                raise ValueError(f"Dataset split '{split}' missing column: {col}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    def tokenize(batch: Dict[str, Any]):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Robust label count
    labels = dataset["train"]["label"]
    num_labels = int(len(set(int(x) for x in labels)))

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
    )

    # Transformers 5.2.0: use eval_strategy (NOT evaluation_strategy)
    training_args = TrainingArguments(
    output_dir="tmp_train",

    # Full training
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",

    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,   # effective batch size = 16

    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,

    fp16=True,  # RTX 3050 supports it
    logging_steps=100,

    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,

    report_to=[],
)

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)
    trainer.train()
    metrics = trainer.evaluate()

    tmp_model_dir = Path("tmp_model")
    tmp_model_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(tmp_model_dir)
    tokenizer.save_pretrained(tmp_model_dir)

    # Export to ONNX (optimum)
    from optimum.exporters.onnx import main_export

    main_export(
        model_name_or_path=str(tmp_model_dir),
        output=tmp_model_dir,
        task="sequence-classification",
    )

    artifact_path = tmp_model_dir / "model.onnx"
    if not artifact_path.exists():
        raise RuntimeError("ONNX export failed: model.onnx not found")

    metadata: Dict[str, Any] = {
        "model_name": MODEL_KEY,
        "version": VERSION,
        "role": ROLE,
        "framework": "transformers",
        "base_model": MODEL_NAME,
        "metrics": metrics,
        "num_labels": num_labels,
        "dataset_path": dataset_path,
        "max_steps": int(training_args.max_steps),
    }

    register_model(
        role=ROLE,
        model_name=MODEL_KEY,
        version=VERSION,
        artifact_path=artifact_path,
        metadata=metadata,
    )

    print("✅ Training complete and model registered.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    args = p.parse_args()
    train(args.dataset_path)


if __name__ == "__main__":
    main()

LABEL_MAP = {
    0: "grammar",
    1: "structure",
    2: "clarity",
    3: "evidence",
    4: "argument",
    5: "other",
    6: "style",
}