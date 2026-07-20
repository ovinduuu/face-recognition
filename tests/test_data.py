import numpy as np
from PIL import Image

from src.data import TripletDataset, get_transforms, make_splits

RNG = np.random.default_rng(0)
# 10 identities x 30 images each mimics the LFW label structure
LABELS = np.repeat(np.arange(10), 30)
IMAGES = RNG.integers(0, 255, size=(300, 50, 40, 3), dtype=np.uint8)


def test_splits_are_disjoint_and_cover_everything():
    train, val, test = make_splits(LABELS)
    all_idx = np.concatenate([train, val, test])
    assert len(all_idx) == len(LABELS)
    assert len(np.unique(all_idx)) == len(LABELS)


def test_splits_are_stratified():
    train, val, test = make_splits(LABELS)
    for split in (train, val, test):
        counts = np.bincount(LABELS[split], minlength=10)
        # every identity appears in every split
        assert counts.min() > 0


def test_split_proportions():
    train, val, test = make_splits(LABELS, val_frac=0.15, test_frac=0.15)
    assert abs(len(train) / len(LABELS) - 0.70) < 0.02
    assert abs(len(val) / len(LABELS) - 0.15) < 0.02
    assert abs(len(test) / len(LABELS) - 0.15) < 0.02


def test_eval_transform_output_shape():
    img = Image.fromarray(IMAGES[0])
    out = get_transforms(train=False)(img)
    assert tuple(out.shape) == (3, 112, 112)


def test_triplet_dataset_label_semantics():
    ds = TripletDataset(IMAGES, LABELS, train=False)
    a, p, n = ds[5]
    for t in (a, p, n):
        assert tuple(t.shape) == (3, 112, 112)
    # the per-class index pools used for positive/negative sampling
    # must map every index back to its own label
    for c, idxs in ds.class_to_indices.items():
        assert np.all(LABELS[idxs] == c)
