from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.multimodal.model import MultimodalFusionNet


# ----------------------------
# Temperature scaler
# ----------------------------
class TemperatureScaler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / torch.clamp(self.temperature, min=1e-3)


# ----------------------------
# Robust sample normalization
# ----------------------------
def _to_tensor(x: Any, *, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype)
    return torch.tensor(x, dtype=dtype)


def _infer_mask(text_emb: torch.Tensor, ocr_emb: torch.Tensor, audio_emb: torch.Tensor, table_emb: torch.Tensor) -> torch.Tensor:
    # present if norm > 0
    def present(t: torch.Tensor) -> bool:
        return bool(torch.linalg.vector_norm(t).item() > 0.0)

    m = [present(text_emb), present(ocr_emb), present(audio_emb), present(table_emb)]
    return torch.tensor(m, dtype=torch.bool)


def normalize_sample(x: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Accepts older dataset formats and numpy arrays.
    Produces the exact schema expected by MultimodalFusionNet forward:
      text_emb [384], ocr_emb [384], audio_emb [384], table_emb [64], mask [4], label long
    """
    # support multiple key variants
    text = x.get("text_emb", x.get("text_embedding", x.get("text_embedding", x.get("text", None))))
    ocr = x.get("ocr_emb", x.get("ocr_embedding", x.get("ocr", None)))
    audio = x.get("audio_emb", x.get("audio_embedding", x.get("audio", None)))
    table = x.get("table_emb", x.get("table_embedding", x.get("table", None)))

    # defaults if missing
    if text is None:
        text = np.zeros((384,), dtype=np.float32)
    if ocr is None:
        ocr = np.zeros((384,), dtype=np.float32)
    if audio is None:
        audio = np.zeros((384,), dtype=np.float32)
    if table is None:
        table = np.zeros((64,), dtype=np.float32)

    text_t = _to_tensor(text, dtype=torch.float32).view(-1)
    ocr_t = _to_tensor(ocr, dtype=torch.float32).view(-1)
    audio_t = _to_tensor(audio, dtype=torch.float32).view(-1)
    table_t = _to_tensor(table, dtype=torch.float32).view(-1)

    # ensure expected dims
    if text_t.numel() != 384:
        text_t = torch.zeros((384,), dtype=torch.float32)
    if ocr_t.numel() != 384:
        ocr_t = torch.zeros((384,), dtype=torch.float32)
    if audio_t.numel() != 384:
        audio_t = torch.zeros((384,), dtype=torch.float32)
    if table_t.numel() != 64:
        table_t = torch.zeros((64,), dtype=torch.float32)

    mask = x.get("mask", None)
    if mask is None:
        mask_t = _infer_mask(text_t, ocr_t, audio_t, table_t)
    else:
        if isinstance(mask, torch.Tensor):
            mask_t = mask
        elif isinstance(mask, np.ndarray):
            mask_t = torch.from_numpy(mask)
        else:
            mask_t = torch.tensor(mask)
        mask_t = mask_t.to(torch.bool).view(-1)

    if mask_t.numel() != 4:
        mask_t = _infer_mask(text_t, ocr_t, audio_t, table_t)

    label = x.get("label", None)
    if label is None:
        raise KeyError("sample missing 'label'")
    label_t = torch.tensor(int(label), dtype=torch.long)

    return {
        "text_emb": text_t,
        "ocr_emb": ocr_t,
        "audio_emb": audio_t,
        "table_emb": table_t,
        "mask": mask_t,
        "label": label_t,
    }


def collate(batch: list[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Batch is a list[dict] where values may be torch.Tensor, numpy.ndarray, or older keys.
    Normalize each sample to the standard schema and stack.
    """
    batch_n = [normalize_sample(b) for b in batch]
    out: Dict[str, torch.Tensor] = {}
    for k in ("text_emb", "ocr_emb", "audio_emb", "table_emb", "mask"):
        out[k] = torch.stack([b[k] for b in batch_n], dim=0)
    out["label"] = torch.stack([b["label"] for b in batch_n], dim=0)
    return out


@torch.no_grad()
def collect_logits_labels(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["text_emb"], batch["ocr_emb"], batch["audio_emb"], batch["table_emb"], batch["mask"])
        all_logits.append(logits.detach().cpu())
        all_labels.append(batch["label"].detach().cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def _torch_load_dataset(path: Path):
    # PyTorch 2.6 changed default weights_only to True; datasets need False
    # mmap=True reduces memory spikes on large files
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)  # 0 avoids worker serialization edge cases
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path("app/models") / f"{args.model}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = MultimodalFusionNet(num_classes=int(ckpt["num_classes"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dataset_dir = Path("datasets/processed") / args.dataset
    val_path = dataset_dir / "val.pt"
    val_data = _torch_load_dataset(val_path)

    val_loader = DataLoader(
        val_data,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )

    logits, labels = collect_logits_labels(model, val_loader, device)

    scaler = TemperatureScaler()
    nll = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.1, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = nll(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    temp = float(scaler.temperature.detach().cpu().item())
    out = {"temperature": max(temp, 1e-6)}

    out_path = Path("app/models") / f"{args.model}.temperature.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("✅ saved temperature:", out_path, out)


if __name__ == "__main__":
    main()