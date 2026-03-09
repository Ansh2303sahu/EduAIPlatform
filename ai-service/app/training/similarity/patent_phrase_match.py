from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from datasets import load_from_disk
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
MODEL_KEY = "patent_phrase_match"
VERSION = "v1"

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds.reshape(-1)
    labels = labels.reshape(-1)
    mse = mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)
    return {"mse": float(mse), "mae": float(mae)}

def train(dataset_path: str) -> None:
    dataset = load_from_disk(dataset_path)
    for split in ("train", "validation"):
        for col in ("text_a", "text_b", "label"):
            if col not in dataset[split].column_names:
                raise ValueError(f"Missing '{col}' in split '{split}'")

    tok = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    def tokenize(batch: Dict[str, Any]):
        return tok(
            batch["text_a"],
            batch["text_b"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Regression head
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,
        problem_type="regression",
    )

    args = TrainingArguments(
        output_dir="tmp_train_patent",
        num_train_epochs=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="mae",
        greater_is_better=False,
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

    tmp = Path("tmp_model_patent")
    tmp.mkdir(exist_ok=True)
    model.save_pretrained(tmp)
    tok.save_pretrained(tmp)

    from optimum.exporters.onnx import main_export
    main_export(model_name_or_path=str(tmp), output=tmp, task="sequence-classification")

    artifact = tmp / "model.onnx"
    if not artifact.exists():
        raise RuntimeError("ONNX export failed: model.onnx not found")

    bm = Path(dataset_path) / "build_meta.json"
    meta_build = json.loads(bm.read_text()) if bm.exists() else {}

    metadata = {
        "model_name": MODEL_KEY,
        "version": VERSION,
        "role": ROLE,
        "framework": "transformers",
        "base_model": MODEL_NAME,
        "metrics": metrics,
        "num_labels": 1,
        "problem_type": "regression",
        "dataset_path": dataset_path,
        "build_meta": meta_build,
        "artifact": "model.onnx",
    }

    register_model(role=ROLE, model_name=MODEL_KEY, version=VERSION, artifact_path=artifact, metadata=metadata)
    print("✅ Patent phrase match model trained + registered.")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    args = p.parse_args()
    train(args.dataset_path)

if __name__ == "__main__":
    main()