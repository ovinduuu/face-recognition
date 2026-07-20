"""Streamlit demo for the LFW face recognition models.

Run:
    streamlit run app.py

Requires trained checkpoints in models/ (see README "Reproduce").
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.config import MODELS_DIR, ROOT
from src.data import get_transforms
from src.models import build_model

SAMPLES_DIR = ROOT / "samples"
ACCENT = "#4477CC"

st.set_page_config(page_title="Face Recognition Demo", page_icon="🙂",
                   layout="centered")


@st.cache_resource
def load_model(name: str):
    path = MODELS_DIR / f"{name}.pt"
    if not path.exists():
        return None, None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(name, len(ckpt["class_names"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["class_names"]


def pick_image(label: str, key: str):
    source = st.radio(f"{label} source", ["Sample", "Upload"], key=f"src_{key}",
                      horizontal=True, label_visibility="collapsed")
    if source == "Upload":
        f = st.file_uploader(label, type=["jpg", "jpeg", "png"], key=f"up_{key}")
        return Image.open(f).convert("RGB") if f else None
    samples = sorted(SAMPLES_DIR.glob("*.jpg"))
    if not samples:
        st.info("No images in samples/ — upload a photo instead.")
        return None
    choice = st.selectbox(label, samples, key=f"sel_{key}",
                          format_func=lambda p: p.stem.replace("_", " "))
    return Image.open(choice).convert("RGB")


def prob_chart(names, probs):
    """Top-k probabilities as a single-hue, direct-labelled bar chart."""
    fig, ax = plt.subplots(figsize=(6, 2.6))
    y = np.arange(len(names))[::-1]
    ax.barh(y, probs, height=0.55, color=ACCENT)
    for yi, p in zip(y, probs):
        ax.text(min(p + 0.015, 1.0), yi, f"{p:.1%}", va="center",
                fontsize=9, color="#555555")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlim(0, 1.15)
    ax.xaxis.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


@torch.no_grad()
def identify(img: Image.Image, topk: int = 5):
    model, class_names = load_model("resnet18")
    if model is None:
        return None
    x = get_transforms(train=False)(img).unsqueeze(0)
    probs = torch.softmax(model(x), dim=1)[0]
    top = probs.topk(topk)
    return ([class_names[i] for i in top.indices.tolist()], top.values.tolist())


@torch.no_grad()
def embed(img: Image.Image):
    model, _ = load_model("triplet")
    if model is None:
        return None
    x = get_transforms(train=False)(img).unsqueeze(0)
    return model(x)[0].numpy()


st.title("Face Recognition Demo")
st.caption("LFW · 62 identities · fine-tuned ResNet-18 classifier + "
           "triplet-loss embeddings — see the "
           "[project README](https://github.com/ovinduuu/face-recognition)")

if load_model("resnet18")[0] is None:
    st.error("No trained checkpoints found in `models/`. "
             "Train first: `python -m src.train --model resnet18 --epochs 10` "
             "and `python -m src.train_triplet --epochs 8`.")
    st.stop()

tab_id, tab_verify = st.tabs(["🔎 Identify", "🆚 Verify (same person?)"])

with tab_id:
    st.write("Who is this? The classifier only knows the 62 LFW identities "
             "it was trained on.")
    img = pick_image("Face image", "identify")
    if img is not None:
        col_img, col_pred = st.columns([1, 2])
        col_img.image(img, width="stretch")
        names, probs = identify(img)
        col_pred.markdown(f"### {names[0]}")
        col_pred.caption(f"confidence {probs[0]:.1%}")
        col_pred.pyplot(prob_chart(names, probs), width="stretch")

with tab_verify:
    st.write("Are these two photos the same person? Uses cosine similarity "
             "between 128-d triplet embeddings — works even for faces the "
             "model never saw in training.")
    col_a, col_b = st.columns(2)
    with col_a:
        img_a = pick_image("Face A", "a")
        if img_a is not None:
            st.image(img_a, width="stretch")
    with col_b:
        img_b = pick_image("Face B", "b")
        if img_b is not None:
            st.image(img_b, width="stretch")

    threshold = st.slider("Decision threshold (cosine similarity)",
                          0.30, 0.90, 0.60, 0.01)
    if img_a is not None and img_b is not None:
        sim = float(embed(img_a) @ embed(img_b))
        st.metric("Cosine similarity", f"{sim:.3f}")
        if sim >= threshold:
            st.success(f"**Same person** (similarity {sim:.3f} ≥ "
                       f"threshold {threshold:.2f})")
        else:
            st.error(f"**Different people** (similarity {sim:.3f} < "
                     f"threshold {threshold:.2f})")
