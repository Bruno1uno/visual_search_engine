import torch
import pytest
from torch.utils.data import DataLoader
from src.model import EmbeddingNet
from src.evaluate import compute_recall_at_k, validate


class DummyDataset:
    def __init__(self, num_samples=20, num_classes=4):
        self.images = torch.randn(num_samples, 3, 224, 224)
        self.labels = torch.randint(0, num_classes, (num_samples,))
        self.paths = [f"/path/img_{i}.jpg" for i in range(num_samples)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.paths[idx]


def test_compute_recall_at_k_pure_math():
    """Verify pure math compute_recall_at_k with random embeddings and labels."""
    embeddings = torch.randn(20, 128)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    labels = torch.randint(0, 4, (20,))

    recalls = compute_recall_at_k(embeddings, labels, k_values=(1, 2, 4))

    assert 1 in recalls
    assert 2 in recalls
    assert 4 in recalls
    for k, val in recalls.items():
        assert 0.0 <= val <= 1.0


def test_validate_wrapper():
    """Verify validate function extracts embeddings and calls compute_recall_at_k."""
    model = EmbeddingNet(embedding_dim=128, backbone_name="resnet18", pretrained=False)

    dataset = DummyDataset(num_samples=20, num_classes=4)
    dataloader = DataLoader(dataset, batch_size=8)

    recalls, val_loss = validate(model, dataloader, device="cpu", k_values=(1, 2))

    assert 1 in recalls
    assert 2 in recalls
    assert val_loss == 0.0
    for k, val in recalls.items():
        assert 0.0 <= val <= 1.0
