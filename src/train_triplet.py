"""Train the embedding network with triplet margin loss for face verification.

Usage:
    python -m src.train_triplet --epochs 10
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import EMBEDDING_DIM, IMG_SIZE, MODELS_DIR, SEED
from .data import TripletDataset, load_lfw, make_splits
from .models import EmbeddingNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--margin", type=float, default=0.3)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images, labels, class_names = load_lfw()
    train_idx, val_idx, _ = make_splits(labels)
    train_ds = TripletDataset(images[train_idx], labels[train_idx], train=True)
    val_ds = TripletDataset(images[val_idx], labels[val_idx], train=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    model = EmbeddingNet().to(device)
    criterion = nn.TripletMarginLoss(margin=args.margin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    ckpt_path = MODELS_DIR / "triplet.pt"

    for epoch in tqdm(range(1, args.epochs + 1), desc="train triplet"):
        t0 = time.time()
        model.train()
        tr_loss, seen = 0.0, 0
        for a, p, n in train_dl:
            a, p, n = a.to(device), p.to(device), n.to(device)
            loss = criterion(model(a), model(p), model(n))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(a)
            seen += len(a)
        tr_loss /= seen

        model.eval()
        va_loss, seen = 0.0, 0
        with torch.no_grad():
            for a, p, n in val_dl:
                a, p, n = a.to(device), p.to(device), n.to(device)
                va_loss += criterion(model(a), model(p), model(n)).item() * len(a)
                seen += len(a)
        va_loss /= seen
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        print(f"epoch {epoch:02d} | train {tr_loss:.4f} | val {va_loss:.4f} "
              f"| {time.time()-t0:.0f}s")
        if va_loss < best_val:
            best_val = va_loss
            torch.save({
                "model_type": "triplet",
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "img_size": IMG_SIZE,
                "embedding_dim": EMBEDDING_DIM,
                "val_loss": va_loss,
            }, ckpt_path)

    (MODELS_DIR / "history_triplet.json").write_text(json.dumps(history))
    print(f"best val loss {best_val:.4f} — checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
