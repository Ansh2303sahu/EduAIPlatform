import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from app.multimodal.prof_multitask_model import ProfMultitaskMultimodalNet

class PTDataset(Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def load_pt(dataset_dir: Path, name: str):
    return torch.load(dataset_dir / f"{name}.pt", map_location="cpu", mmap=True, weights_only=False)

def collate(batch):
    def stack(key): return torch.stack([b[key] for b in batch], dim=0)
    labels = {}
    for k in batch[0]["labels"].keys():
        labels[k] = torch.stack([b["labels"][k] for b in batch], dim=0).view(-1)
    return {
        "text_emb": stack("text_emb"),
        "ocr_emb": stack("ocr_emb"),
        "audio_emb": stack("audio_emb"),
        "table_emb": stack("table_emb"),
        "mask": stack("mask"),
        "labels": labels,
    }

@torch.no_grad()
def eval_loss(model, loader, device, losses, weights):
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        for k in ("text_emb","ocr_emb","audio_emb","table_emb","mask"):
            batch[k] = batch[k].to(device)
        for hk in batch["labels"]:
            batch["labels"][hk] = batch["labels"][hk].to(device)

        outs = model(batch["text_emb"], batch["ocr_emb"], batch["audio_emb"], batch["table_emb"], batch["mask"])
        loss = 0.0
        for name, logits in outs.items():
            loss = loss + weights[name] * losses[name](logits, batch["labels"][name])
        total += float(loss.item())
        n += 1
    return total / max(n, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--patience", type=int, default=3)

    # heads: pass like --head rubric_band=7 --head argument_depth=4 ...
    ap.add_argument("--head", action="append", required=True)

    # weights: optional like --weight rubric_band=1.0
    ap.add_argument("--weight", action="append", default=[])

    args = ap.parse_args()

    head_dims = {}
    for item in args.head:
        k, v = item.split("=")
        head_dims[k] = int(v)

    weights = {k: 1.0 for k in head_dims}
    for item in args.weight:
        k, v = item.split("=")
        weights[k] = float(v)

    dataset_dir = Path("datasets/processed") / args.dataset
    train_data = load_pt(dataset_dir, "train")
    val_data = load_pt(dataset_dir, "val")

    train_loader = DataLoader(PTDataset(train_data), batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate)
    val_loader = DataLoader(PTDataset(val_data), batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ProfMultitaskMultimodalNet(head_dims=head_dims).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    losses = {k: nn.CrossEntropyLoss() for k in head_dims}

    best = 1e9
    bad = 0
    best_state = None

    for ep in range(1, args.epochs+1):
        model.train()
        for batch in train_loader:
            for k in ("text_emb","ocr_emb","audio_emb","table_emb","mask"):
                batch[k] = batch[k].to(device)
            for hk in batch["labels"]:
                batch["labels"][hk] = batch["labels"][hk].to(device)

            opt.zero_grad(set_to_none=True)

            outs = model(batch["text_emb"], batch["ocr_emb"], batch["audio_emb"], batch["table_emb"], batch["mask"])
            loss = 0.0
            for name, logits in outs.items():
                loss = loss + weights[name] * losses[name](logits, batch["labels"][name])

            loss.backward()
            opt.step()

        v = eval_loss(model, val_loader, device, losses, weights)
        print(f"epoch {ep} val_loss={v:.4f}")

        if v < best - 1e-4:
            best = v
            bad = 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    out_path = Path("app/models") / f"{args.model}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "head_dims": head_dims}, out_path)
    print("saved:", out_path)

if __name__ == "__main__":
    main()