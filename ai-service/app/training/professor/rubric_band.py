from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

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
ROLE = "professor"
MODEL_KEY = "rubric_band_predictor"
VERSION = "v1"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="weighted")),
    }


def export_onnx(model: DistilBertForSequenceClassification, out_path: Path) -> None:
    """
    Export to ONNX without Optimum.
    Produces a model.onnx with input_ids + attention_mask.
    """
    model.eval()
    model.cpu()

    dummy_input_ids = torch.ones((1, 256), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, 256), dtype=torch.long)

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        str(out_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )


def train(dataset_path: str):
    dataset = load_from_disk(dataset_path)

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        MODEL_NAME,
        local_files_only=True,  # ✅ avoids huggingface timeouts
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    num_labels = len(set(dataset["train"]["label"]))

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        local_files_only=True,  # ✅ offline safe
    )

    # Transformers v5+ uses eval_strategy
    training_args = TrainingArguments(
        output_dir="tmp_train_prof_rubric",
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()

    tmp_model_dir = Path("tmp_model_prof_rubric")
    tmp_model_dir.mkdir(exist_ok=True)

    # Save HF format (optional but useful)
    model.save_pretrained(tmp_model_dir)
    tokenizer.save_pretrained(tmp_model_dir)

    # Export ONNX without Optimum
    onnx_path = tmp_model_dir / "model.onnx"
    export_onnx(model, onnx_path)

    # (optional) label map you can override later
    label_map = {str(i): f"band_{i}" for i in range(num_labels)}

    metadata: Dict = {
        "model_name": MODEL_KEY,
        "version": VERSION,
        "role": ROLE,
        "framework": "transformers",
        "base_model": MODEL_NAME,
        "metrics": metrics,
        "num_labels": num_labels,
        "dataset_path": dataset_path,
        "label_map": label_map,
    }

    register_model(
        role=ROLE,
        model_name=MODEL_KEY,
        version=VERSION,
        artifact_path=onnx_path,
        metadata=metadata,
    )

    print("✅ Professor rubric band training complete and model registered.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    args = p.parse_args()
    train(args.dataset_path)


if __name__ == "__main__":
    main()   