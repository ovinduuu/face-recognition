"""Evaluate trained models on the held-out test split and produce figures.

Usage:
    python -m src.evaluate --model all
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score,
                             roc_curve)
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader

from .config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, SEED
from .data import FaceDataset, load_lfw, make_splits
from .models import build_model


def load_checkpoint(model_type, num_classes, device):
    ckpt = torch.load(MODELS_DIR / f"{model_type}.pt", map_location=device,
                      weights_only=False)
    model = build_model(model_type, num_classes).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_all(model, loader, device):
    logits_all, ys = [], []
    for x, y in loader:
        logits_all.append(model(x.to(device)).cpu())
        ys.append(y)
    return torch.cat(logits_all), torch.cat(ys)


def eval_classifier(model_type, data, device, metrics):
    images, labels, class_names, splits = data
    _, _, test_idx = splits
    test_ds = FaceDataset(images[test_idx], labels[test_idx], train=False)
    loader = DataLoader(test_ds, batch_size=64)
    model, _ = load_checkpoint(model_type, len(class_names), device)

    logits, y_true = predict_all(model, loader, device)
    y_pred = logits.argmax(1).numpy()
    y_true = y_true.numpy()
    top5 = logits.topk(5, dim=1).indices.numpy()

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    top5_acc = float(np.mean([t in row for t, row in zip(y_true, top5)]))
    metrics[model_type] = {"test_acc": acc, "macro_f1": macro_f1,
                           "top5_acc": top5_acc}
    print(f"[{model_type}] acc={acc:.3f} macro-F1={macro_f1:.3f} "
          f"top5={top5_acc:.3f}")

    report = classification_report(y_true, y_pred, labels=np.unique(y_true),
                                   target_names=[class_names[i] for i in np.unique(y_true)],
                                   zero_division=0)
    (REPORTS_DIR / f"classification_report_{model_type}.txt").write_text(report)

    # Confusion matrix over the 12 most frequent identities keeps it readable.
    top_classes = np.argsort(np.bincount(y_true))[::-1][:12]
    mask = np.isin(y_true, top_classes)
    cm = confusion_matrix(y_true[mask], y_pred[mask], labels=top_classes)
    cm = cm / cm.sum(axis=1, keepdims=True)
    names = [class_names[i].split()[-1] for i in top_classes]
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=names, yticklabels=names, cbar=False)
    plt.title(f"Confusion matrix (top-12 identities) — {model_type}")
    plt.ylabel("true"); plt.xlabel("predicted")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"confusion_{model_type}.png", dpi=150)
    plt.close()

    if model_type == "resnet18":
        sample_grid(images[test_idx], y_true, y_pred, class_names)


def sample_grid(test_images, y_true, y_pred, class_names):
    rng = np.random.default_rng(SEED)
    picks = rng.choice(len(y_true), 12, replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(11, 9))
    for ax, i in zip(axes.flat, picks):
        ax.imshow(test_images[i])
        ok = y_true[i] == y_pred[i]
        ax.set_title(f"pred: {class_names[y_pred[i]]}\ntrue: {class_names[y_true[i]]}",
                     fontsize=9, color="green" if ok else "red")
        ax.axis("off")
    fig.suptitle("Sample test predictions — fine-tuned ResNet-18")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sample_predictions.png", dpi=150)
    plt.close(fig)


@torch.no_grad()
def embed(model, images, labels, device):
    ds = FaceDataset(images, labels, train=False)
    loader = DataLoader(ds, batch_size=64)
    out = [model(x.to(device)).cpu() for x, _ in loader]
    return torch.cat(out).numpy()


def eval_triplet(data, device, metrics):
    images, labels, class_names, splits = data
    train_idx, _, test_idx = splits
    model, _ = load_checkpoint("triplet", len(class_names), device)

    emb_train = embed(model, images[train_idx], labels[train_idx], device)
    emb_test = embed(model, images[test_idx], labels[test_idx], device)
    y_train, y_test = labels[train_idx], labels[test_idx]

    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    knn.fit(emb_train, y_train)
    knn_acc = knn.score(emb_test, y_test)

    # Face verification: cosine similarity over same/different identity pairs.
    rng = np.random.default_rng(SEED)
    sims, targets = [], []
    by_class = {c: np.where(y_test == c)[0] for c in np.unique(y_test)}
    multi = [c for c, idxs in by_class.items() if len(idxs) > 1]
    for _ in range(3000):
        c = rng.choice(multi)
        i, j = rng.choice(by_class[c], 2, replace=False)
        sims.append(float(emb_test[i] @ emb_test[j])); targets.append(1)
        c1, c2 = rng.choice(list(by_class), 2, replace=False)
        i = rng.choice(by_class[c1]); j = rng.choice(by_class[c2])
        sims.append(float(emb_test[i] @ emb_test[j])); targets.append(0)
    auc = roc_auc_score(targets, sims)
    fpr, tpr, _ = roc_curve(targets, sims)

    metrics["triplet"] = {"knn_acc": knn_acc, "verification_auc": auc}
    print(f"[triplet] kNN acc={knn_acc:.3f} verification AUC={auc:.3f}")

    plt.figure(figsize=(5.5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("false positive rate"); plt.ylabel("true positive rate")
    plt.title("Face verification ROC — triplet embeddings")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(FIGURES_DIR / "verification_roc.png", dpi=150)
    plt.close()

    # t-SNE of test embeddings for the 10 most frequent identities.
    top10 = np.argsort(np.bincount(y_test))[::-1][:10]
    mask = np.isin(y_test, top10)
    proj = TSNE(n_components=2, random_state=SEED,
                perplexity=20).fit_transform(emb_test[mask])
    plt.figure(figsize=(8, 6.5))
    for c in top10:
        m = y_test[mask] == c
        plt.scatter(proj[m, 0], proj[m, 1], s=18,
                    label=class_names[c].split()[-1])
    plt.legend(fontsize=8, ncol=2)
    plt.title("t-SNE of face embeddings (top-10 identities)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tsne_embeddings.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["scratch", "resnet18", "triplet", "all"],
                        default="all")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, labels, class_names = load_lfw()
    data = (images, labels, class_names, make_splits(labels))

    metrics_path = REPORTS_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    targets = ["scratch", "resnet18", "triplet"] if args.model == "all" else [args.model]
    for t in targets:
        if t == "triplet":
            eval_triplet(data, device, metrics)
        else:
            eval_classifier(t, data, device, metrics)

    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
