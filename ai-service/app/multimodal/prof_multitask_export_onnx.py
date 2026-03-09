from __future__ import annotations
import argparse
from pathlib import Path
import torch

from app.multimodal.prof_multitask_model import ProfMultitaskMultimodalNet
from app.registry.model_registry import register_multimodal_onnx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--dataset_version", default="prof_multimodal_v1")
    args = ap.parse_args()

    pt_path = Path("app/models") / f"{args.model}.pt"
    ckpt = torch.load(pt_path, map_location="cpu")

    model = ProfMultitaskMultimodalNet(head_dims=ckpt["head_dims"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dummy = {
        "text_emb": torch.zeros(1, 384),
        "ocr_emb": torch.zeros(1, 384),
        "audio_emb": torch.zeros(1, 384),
        "table_emb": torch.zeros(1, 64),
        "mask": torch.ones(1, 4, dtype=torch.bool),
    }

    onnx_path = pt_path.with_suffix(".onnx")

    torch.onnx.export(
        model,
        (
            dummy["text_emb"],
            dummy["ocr_emb"],
            dummy["audio_emb"],
            dummy["table_emb"],
            dummy["mask"],
        ),
        onnx_path,
        input_names=["text_emb", "ocr_emb", "audio_emb", "table_emb", "mask"],
        output_names=list(ckpt["head_dims"].keys()),
        dynamic_axes={
            "text_emb": {0: "batch"},
            "ocr_emb": {0: "batch"},
            "audio_emb": {0: "batch"},
            "table_emb": {0: "batch"},
            "mask": {0: "batch"},
        },
        opset_version=17,
    )

    register_multimodal_onnx(
        role="professor",
        model_name=args.model,
        version=args.version,
        onnx_path=onnx_path,
        dataset_version=args.dataset_version,
    )

    print("✅ Exported & registered:", args.model)


if __name__ == "__main__":
    main()