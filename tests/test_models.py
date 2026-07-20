import torch

from src.models import EmbeddingNet, SimpleCNN, build_resnet18

BATCH = 2
X = torch.randn(BATCH, 3, 112, 112)


def test_simple_cnn_output_shape():
    model = SimpleCNN(num_classes=62).eval()
    assert model(X).shape == (BATCH, 62)


def test_resnet18_output_shape():
    model = build_resnet18(num_classes=62, pretrained=False).eval()
    assert model(X).shape == (BATCH, 62)


def test_embedding_net_output_is_unit_norm():
    model = EmbeddingNet(pretrained=False).eval()
    emb = model(X)
    assert emb.shape == (BATCH, 128)
    norms = emb.norm(dim=1)
    assert torch.allclose(norms, torch.ones(BATCH), atol=1e-5)
