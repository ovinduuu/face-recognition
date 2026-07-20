"""Predict the identity of a single face image with a trained classifier.

Usage:
    python -m src.predict path/to/face.jpg --model resnet18
"""
import argparse

import torch
from PIL import Image

from .config import MODELS_DIR
from .data import get_transforms
from .models import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="path to a face image")
    parser.add_argument("--model", choices=["scratch", "resnet18"],
                        default="resnet18")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MODELS_DIR / f"{args.model}.pt", map_location=device,
                      weights_only=False)
    class_names = ckpt["class_names"]
    model = build_model(args.model, len(class_names)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    img = Image.open(args.image).convert("RGB")
    x = get_transforms(train=False)(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]

    top = probs.topk(args.topk)
    print(f"Top-{args.topk} predictions:")
    for p, i in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  {class_names[i]:<30s} {p:6.2%}")


if __name__ == "__main__":
    main()
