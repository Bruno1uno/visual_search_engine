import torch
import pytest
from torch.utils.data import DataLoader
from src.model import EmbeddingNet
from src.loss import get_triplet_loss_and_miner, get_proxy_anchor_loss
from src.train import train_one_epoch


class DummyDataset:
    def __init__(self, num_samples=16, num_classes=4):
        self.images = torch.randn(num_samples, 3, 224, 224)
        self.labels = torch.tensor([i % num_classes for i in range(num_samples)])
        self.paths = [f"/path/img_{i}.jpg" for i in range(num_samples)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.paths[idx]


def test_train_one_epoch_triplet():
    """Smoke test: Single train_one_epoch step with Triplet loss produces finite loss."""
    model = EmbeddingNet(embedding_dim=128, backbone_name="resnet18", pretrained=False)
    loss_fn, miner = get_triplet_loss_and_miner(margin=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loader = DataLoader(DummyDataset(), batch_size=8)

    avg_loss = train_one_epoch(
        model=model,
        train_loader=loader,
        loss_fn=loss_fn,
        miner=miner,
        optimizer=optimizer,
        device="cpu",
    )

    assert isinstance(avg_loss, float)
    assert not torch.isnan(torch.tensor(avg_loss))


def test_train_one_epoch_proxy_anchor():
    """Smoke test: Single train_one_epoch step with Proxy Anchor loss produces finite loss."""
    model = EmbeddingNet(embedding_dim=128, backbone_name="resnet18", pretrained=False)
    loss_fn = get_proxy_anchor_loss(num_classes=4, embedding_dim=128)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()), lr=1e-4
    )
    loader = DataLoader(DummyDataset(), batch_size=8)

    avg_loss = train_one_epoch(
        model=model,
        train_loader=loader,
        loss_fn=loss_fn,
        miner=None,
        optimizer=optimizer,
        device="cpu",
    )

    assert isinstance(avg_loss, float)
    assert not torch.isnan(torch.tensor(avg_loss))
