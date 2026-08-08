import torch
import pytest
from src.model import EmbeddingNet


def test_embedding_net_output_shape_and_l2_norm():
    """Verify EmbeddingNet output shape and exact L2 normalization = 1.0."""
    model = EmbeddingNet(embedding_dim=128, backbone_name="resnet18", pretrained=False)
    model.eval()

    dummy_input = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        embeddings = model(dummy_input)

    assert embeddings.shape == (4, 128)

    # Check L2 norms are all 1.0 within tolerance
    norms = torch.norm(embeddings, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
