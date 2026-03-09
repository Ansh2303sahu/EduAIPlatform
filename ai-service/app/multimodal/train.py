import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from app.multimodal.model import MultimodalFusionNet


def load_pt(dataset_dir: Path, name: str):
    p = dataset_dir / f"{name}.pt"
    # PyTorch 2.6: weights_only defaults to True -> breaks for dataset objects
    # mmap=True reduces RAM spikes for large .pt
    return torch.load(p, map_location="cpu", weights_only=False, mmap=True)


def collate(batch):
    """
    Robust collate:
    - accepts both {text_emb,...} and {text_embedding,...}
    - converts numpy -> torch tensors
    - builds mask if missing
    """
    import numpy as np
    import torch

    def to_tensor(x, dtype=torch.float32):
        if isinstance(x, torch.Tensor):
            return x.to(dtype=dtype) if dtype is not None else x
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(dtype=dtype) if dtype is not None else torch.from_numpy(x)
        # numbers / lists
        return torch.as_tensor(x, dtype=dtype) if dtype is not None else torch.as_tensor(x)

    def norm_sample(b: dict) -> dict:
        # key aliases (your builders sometimes output *_embedding)
        out = dict(b)

        if "text_emb" not in out and "text_embedding" in out:
            out["text_emb"] = out["text_embedding"]
        if "ocr_emb" not in out and "ocr_embedding" in out:
            out["ocr_emb"] = out["ocr_embedding"]
        if "audio_emb" not in out and "audio_embedding" in out:
            out["audio_emb"] = out["audio_embedding"]
        if "table_emb" not in out and "table_embedding" in out:
            out["table_emb"] = out["table_embedding"]

        # enforce required fields
        for k in ("text_emb", "ocr_emb", "audio_emb", "table_emb", "label"):
            if k not in out:
                raise KeyError(f"Dataset sample missing key '{k}'. Keys={list(out.keys())}")

        # convert embeddings
        out["text_emb"] = to_tensor(out["text_emb"], torch.float32).view(-1)
        out["ocr_emb"] = to_tensor(out["ocr_emb"], torch.float32).view(-1)
        out["audio_emb"] = to_tensor(out["audio_emb"], torch.float32).view(-1)
        out["table_emb"] = to_tensor(out["table_emb"], torch.float32).view(-1)

        # label always int64
        out["label"] = to_tensor(out["label"], torch.int64).view(())

        # mask: if missing, infer "modality present" by checking non-zero vectors
        if "mask" not in out:
            def present(v: torch.Tensor) -> bool:
                return bool(torch.any(v != 0))

            m = torch.tensor(
                [
                    present(out["text_emb"]),
                    present(out["ocr_emb"]),
                    present(out["audio_emb"]),
                    present(out["table_emb"]),
                ],
                dtype=torch.bool,
            )
            out["mask"] = m
        else:
            out["mask"] = to_tensor(out["mask"], None).to(torch.bool).view(-1)

        return out

    nbatch = [norm_sample(b) for b in batch]

    out = {}
    for k in ("text_emb", "ocr_emb", "audio_emb", "table_emb"):
        out[k] = torch.stack([b[k] for b in nbatch], dim=0)
    out["mask"] = torch.stack([b["mask"] for b in nbatch], dim=0)
    out["label"] = torch.stack([b["label"] for b in nbatch], dim=0)

    return out


def train(model, train_loader, val_loader, device, epochs=10, lr=2e-4, patience=3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_val = float("inf")
    bad = 0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        tr = 0.0

        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)

            with autocast(enabled=(device.type == "cuda")):
                logits = model(
                    batch["text_emb"],
                    batch["ocr_emb"],
                    batch["audio_emb"],
                    batch["table_emb"],
                    batch["mask"],
                )
                loss = loss_fn(logits, batch["label"])

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            tr += float(loss.item())

        tr /= max(1, len(train_loader))

        model.eval()
        va = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                logits = model(
                    batch["text_emb"],
                    batch["ocr_emb"],
                    batch["audio_emb"],
                    batch["table_emb"],
                    batch["mask"],
                )
                loss = loss_fn(logits, batch["label"])
                va += float(loss.item())

        va /= max(1, len(val_loader))

        print(f"epoch {ep} train={tr:.4f} val={va:.4f}")

        if va < best_val - 1e-4:
            best_val = va
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="processed dataset folder name")
    ap.add_argument("--model", required=True, help="output model name (without .pt)")
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--patience", type=int, default=3)
    args = ap.parse_args()

    dataset_dir = Path("datasets/processed") / args.dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

    train_data = load_pt(dataset_dir, "train")
    val_data = load_pt(dataset_dir, "val")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # pin_memory helps GPU transfers; safe to keep False on CPU
    pin_memory = (device.type == "cuda")

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch,
        shuffle=True,
        num_workers=2,
        pin_memory=pin_memory,
        collate_fn=collate,
        persistent_workers=True if 2 > 0 else False,
        prefetch_factor=2 if 2 > 0 else None,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
        collate_fn=collate,
        persistent_workers=True if 2 > 0 else False,
        prefetch_factor=2 if 2 > 0 else None,
    )

    model = MultimodalFusionNet(num_classes=args.num_classes).to(device)
    model, best_val = train(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
    )

    out_dir = Path("app/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}.pt"
    torch.save({"state_dict": model.state_dict(), "num_classes": args.num_classes}, out_path)
    print("✅ saved:", out_path, "best_val:", best_val)


if __name__ == "__main__":
    main()