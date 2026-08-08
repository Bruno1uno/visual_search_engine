import torch
import pytest
from src.loss import get_triplet_loss_and_miner, get_proxy_anchor_loss


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


def test_proxy_anchor_loss():
    """Verify ProxyAnchorLoss initialization, forward pass, and parameters optimization."""
    loss_fn = get_proxy_anchor_loss(num_classes=10, embedding_dim=128)

    embeddings = torch.randn(8, 128)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    loss = loss_fn(embeddings, labels)

    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    # Check ProxyAnchorLoss contains learnable parameters
    assert len(list(loss_fn.parameters())) > 0
