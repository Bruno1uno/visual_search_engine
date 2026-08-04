import torch
import pytest
from src.model import EmbeddingNet
from src.loss import get_triplet_loss_and_miner


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


def test_triplet_loss_and_miner():
    """Verify triplet loss and miner initialization and basic execution."""
    loss_fn, miner = get_triplet_loss_and_miner(margin=0.2, type_of_triplets="semi-hard")

    embeddings = torch.randn(8, 128)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    mined_triplets = miner(embeddings, labels)
    loss = loss_fn(embeddings, labels, mined_triplets)

    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
