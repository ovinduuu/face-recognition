"""Model architectures: scratch CNN, transfer-learning ResNet-18, embedding net."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .config import EMBEDDING_DIM


class SimpleCNN(nn.Module):
    """Compact CNN trained from scratch as a baseline."""

    def __init__(self, num_classes: int):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128), block(128, 256)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def build_resnet18(num_classes: int, pretrained: bool = True):
    """ImageNet-pretrained ResNet-18 with a new classification head."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class EmbeddingNet(nn.Module):
    """ResNet-18 backbone mapping faces to L2-normalised embeddings."""

    def __init__(self, embedding_dim: int = EMBEDDING_DIM, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        backbone.fc = nn.Linear(backbone.fc.in_features, embedding_dim)
        self.backbone = backbone

    def forward(self, x):
        return F.normalize(self.backbone(x), p=2, dim=1)


def build_model(model_type: str, num_classes: int):
    if model_type == "scratch":
        return SimpleCNN(num_classes)
    if model_type == "resnet18":
        return build_resnet18(num_classes)
    if model_type == "triplet":
        return EmbeddingNet()
    raise ValueError(f"unknown model type: {model_type}")
