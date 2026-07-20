"""LFW dataset loading, splitting and PyTorch dataset wrappers."""
import numpy as np
import torch
from PIL import Image
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from .config import DATA_DIR, IMG_SIZE, MIN_FACES_PER_PERSON, SEED

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_lfw(min_faces_per_person: int = MIN_FACES_PER_PERSON):
    """Download (if needed) and load the funneled LFW dataset in colour.

    Returns (images uint8 [N,H,W,3], labels int64 [N], class_names list[str]).
    """
    lfw = fetch_lfw_people(
        data_home=str(DATA_DIR),
        min_faces_per_person=min_faces_per_person,
        color=True,
        resize=1.0,
        funneled=True,
    )
    images = lfw.images
    if images.max() <= 1.0:  # sklearn scales pixels to [0, 1]
        images = images * 255.0
    images = np.clip(images, 0, 255).astype(np.uint8)
    labels = lfw.target.astype(np.int64)
    class_names = [str(n) for n in lfw.target_names]
    return images, labels, class_names


def make_splits(labels, val_frac=0.15, test_frac=0.15):
    """Stratified train/val/test index split."""
    idx = np.arange(len(labels))
    holdout = val_frac + test_frac
    train_idx, rest_idx = train_test_split(
        idx, test_size=holdout, stratify=labels, random_state=SEED
    )
    val_idx, test_idx = train_test_split(
        rest_idx,
        test_size=test_frac / holdout,
        stratify=labels[rest_idx],
        random_state=SEED,
    )
    return train_idx, val_idx, test_idx


def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class FaceDataset(Dataset):
    """Classification dataset over in-memory LFW images."""

    def __init__(self, images, labels, train: bool):
        self.images = images
        self.labels = labels
        self.transform = get_transforms(train)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.transform(Image.fromarray(self.images[i]))
        return img, int(self.labels[i])


class TripletDataset(Dataset):
    """Yields (anchor, positive, negative) triplets for metric learning."""

    def __init__(self, images, labels, train: bool = True):
        self.images = images
        self.labels = labels
        self.transform = get_transforms(train)
        self.class_to_indices = {
            c: np.where(labels == c)[0] for c in np.unique(labels)
        }
        self.classes = np.array(list(self.class_to_indices))
        self.rng = np.random.default_rng(SEED)

    def __len__(self):
        return len(self.labels)

    def _load(self, i):
        return self.transform(Image.fromarray(self.images[i]))

    def __getitem__(self, i):
        anchor_label = self.labels[i]
        pos_pool = self.class_to_indices[anchor_label]
        pos = i
        while pos == i and len(pos_pool) > 1:
            pos = int(self.rng.choice(pos_pool))
        neg_label = anchor_label
        while neg_label == anchor_label:
            neg_label = int(self.rng.choice(self.classes))
        neg = int(self.rng.choice(self.class_to_indices[neg_label]))
        return self._load(i), self._load(pos), self._load(neg)
