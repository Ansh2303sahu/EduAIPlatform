from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch

# ---- Import whichever models you actually use
from app.multimodal.model import MultimodalFusionNet
from app.multimodal.prof_multitask_model import ProfMultitaskMultimodalNet


def export_single_head(model_name: str, ckpt: dict, out_path: Path) -> None:
    model = MultimodalFusionNet(num_classes=int(ckpt["num_classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    B = 1
    text = torch.randn(B, 384)
    ocr = torch.randn(B, 384)
    audio = torch.randn(B, 384)
    table = torch.randn(B, 64)
    mask = torch.tensor([[True, True, False, True]], dtype=torch.bool)

    torch.onnx.export(
        model,
        (text, ocr, audio, table, mask),
        str(out_path),
        input_names=["text_emb", "ocr_emb", "audio_emb", "table_emb", "mask"],
        output_names=["logits"],
        dynamic_axes={
            "text_emb": {0: "batch"},
            "ocr_emb": {0: "batch"},
            "audio_emb": {0: "batch"},
            "table_emb": {0: "batch"},
            "mask": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )


def export_multitask(model_name: str, ckpt: dict, out_path: Path) -> None:
    head_dims: Dict[str, int] = ckpt["head_dims"]
    model = ProfMultitaskMultimodalNet(head_dims=head_dims)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    B = 1
    text = torch.randn(B, 384)
    ocr = torch.randn(B, 384)
    audio = torch.randn(B, 384)
    table = torch.randn(B, 64)
    mask = torch.tensor([[1, 1, 0, 1]], dtype=torch.int64)

    # Wrapper to force deterministic output order + output names
    class Wrap(torch.nn.Module):
        def __init__(self, m: torch.nn.Module, order):
            super().__init__()
            self.m = m
            self.order = order

        def forward(self, text_emb, ocr_emb, audio_emb, table_emb, mask):
            out = self.m(text_emb, ocr_emb, audio_emb, table_emb, mask)
            return tuple(out[name] for name in self.order)

    head_order = list(head_dims.keys())
    wrapped = Wrap(model, head_order)

    output_names = [f"logits_{h}" for h in head_order]
    dynamic_axes = {
        "text_emb": {0: "batch"},
        "ocr_emb": {0: "batch"},
        "audio_emb": {0: "batch"},
        "table_emb": {0: "batch"},
        "mask": {0: "batch"},
    }
    for name in output_names:
        dynamic_axes[name] = {0: "batch"}

    torch.onnx.export(
        wrapped,
        (text, ocr, audio, table, mask),
        str(out_path),
        input_names=["text_emb", "ocr_emb", "audio_emb", "table_emb", "mask"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint base name in app/models (without .pt)")
    ap.add_argument("--multitask", action="store_true", help="export multitask model (expects head_dims in ckpt)")
    ap.add_argument("--out", default="", help="output path (default: app/models/<model>.onnx)")
    args = ap.parse_args()

    ckpt_path = Path("app/models") / f"{args.model}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    out_path = Path(args.out) if args.out else (Path("app/models") / f"{args.model}.onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.multitask:
        if "head_dims" not in ckpt:
            raise ValueError("Checkpoint missing head_dims. Did you train the multitask model?")
        export_multitask(args.model, ckpt, out_path)
    else:
        if "num_classes" not in ckpt:
            raise ValueError("Checkpoint missing num_classes. Did you train the single-head model?")
        export_single_head(args.model, ckpt, out_path)

    print("✅ exported:", out_path)


if __name__ == "__main__":
    main()