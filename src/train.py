"""Train a face classifier (scratch CNN or fine-tuned ResNet-18) on LFW.

Usage:
    python -m src.train --model resnet18 --epochs 10
    python -m src.train --model scratch --epochs 20
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import FIGURES_DIR, IMG_SIZE, MODELS_DIR, SEED
from .data import FaceDataset, load_lfw, make_splits
from .models import build_model


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
    return total_loss / seen, correct / seen


def plot_history(history, model_type):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for split in ("train", "val"):
        axes[0].plot(history[f"{split}_loss"], label=split)
        axes[1].plot(history[f"{split}_acc"], label=split)
    axes[0].set_title("Loss"); axes[1].set_title("Accuracy")
    for ax in axes:
        ax.set_xlabel("epoch"); ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle(f"Training history — {model_type}")
    fig.tight_layout()
    out = FIGURES_DIR / f"history_{model_type}.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["scratch", "resnet18"], required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = args.lr or (1e-3 if args.model == "scratch" else 3e-4)

    images, labels, class_names = load_lfw()
    train_idx, val_idx, _ = make_splits(labels)
    print(f"{len(images)} images, {len(class_names)} identities "
          f"(train={len(train_idx)}, val={len(val_idx)})")

    train_ds = FaceDataset(images[train_idx], labels[train_idx], train=True)
    val_ds = FaceDataset(images[val_idx], labels[val_idx], train=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    # Inverse-frequency class weights counter the heavy LFW class imbalance.
    counts = np.bincount(labels[train_idx], minlength=len(class_names))
    weights = torch.tensor(len(train_idx) / (len(class_names) * counts),
                           dtype=torch.float32, device=device)

    model = build_model(args.model, len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best_acc = 0.0
    ckpt_path = MODELS_DIR / f"{args.model}.pt"

    for epoch in tqdm(range(1, args.epochs + 1), desc=f"train {args.model}"):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_dl, criterion, device)
        scheduler.step()
        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss); history["val_acc"].append(va_acc)
        print(f"epoch {epoch:02d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f} | {time.time()-t0:.0f}s")
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({
                "model_type": args.model,
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "img_size": IMG_SIZE,
                "val_acc": va_acc,
            }, ckpt_path)

    (MODELS_DIR / f"history_{args.model}.json").write_text(json.dumps(history))
    plot_history(history, args.model)
    print(f"best val acc {best_acc:.3f} — checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
