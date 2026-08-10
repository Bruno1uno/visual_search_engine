import os
import torch
import pytest
from torch.utils.data import DataLoader
from src.model import EmbeddingNet
from src.evaluate import (
    compute_recall_at_k,
    compute_map,
    compute_nmi,
    validate,
    evaluate_seen_test_split,
    evaluate_unseen_test_split,
)


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


def test_compute_map():
    """Verify compute_map score computation and boundary behavior."""
    # 1. Identical class clusters -> mAP should be 1.0
    labels = torch.tensor([0, 0, 1, 1])
    # Create distinct orthogonal vectors per class
    embeds = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ])
    map_score = compute_map(embeds, labels)
    assert pytest.approx(map_score, abs=1e-5) == 1.0

    # 2. Random embeddings -> mAP within [0, 1]
    rand_embeds = torch.nn.functional.normalize(torch.randn(30, 128), dim=1)
    rand_labels = torch.randint(0, 5, (30,))
    m_val = compute_map(rand_embeds, rand_labels)
    assert 0.0 <= m_val <= 1.0


def test_compute_nmi():
    """Verify NMI score computation bounds."""
    embeds = torch.nn.functional.normalize(torch.randn(40, 128), dim=1)
    labels = torch.randint(0, 4, (40,))

    nmi_score = compute_nmi(embeds, labels, n_clusters=4)
    assert 0.0 <= nmi_score <= 1.0


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


def test_evaluate_splits(tmp_path):
    """Verify end-to-end split evaluation and plot generation."""
    embeds = torch.nn.functional.normalize(torch.randn(30, 128), dim=1)
    labels = torch.tensor([i % 3 for i in range(30)])

    cm_path = os.path.join(tmp_path, "confusion.png")
    tsne_seen_path = os.path.join(tmp_path, "tsne_seen.png")
    tsne_path = os.path.join(tmp_path, "tsne.png")

    seen_res = evaluate_seen_test_split(embeds, labels, save_plot_path=cm_path, save_tsne_path=tsne_seen_path)
    assert "knn_accuracy_top1" in seen_res
    assert 0.0 <= seen_res["knn_accuracy_top1"] <= 1.0
    assert os.path.exists(cm_path)
    assert os.path.exists(tsne_seen_path)

    unseen_res = evaluate_unseen_test_split(embeds, labels, save_plot_path=tsne_path)
    assert "recall_at_1" in unseen_res
    assert "mAP" in unseen_res
    assert "NMI" in unseen_res
    assert os.path.exists(tsne_path)
