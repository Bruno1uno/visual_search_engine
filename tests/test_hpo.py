import os
import pytest
import torch
from unittest.mock import patch
from torch.utils.data import DataLoader
import optuna

from src.model import EmbeddingNet
from src.loss import get_proxy_anchor_loss
from src.hpo import run_hpo, objective


class DummyDataset:
    def __init__(self, num_samples=16, num_classes=4):
        self.images = torch.randn(num_samples, 3, 224, 224)
        self.labels = torch.tensor([i % num_classes for i in range(num_samples)])
        self.paths = [f"/path/img_{i}.jpg" for i in range(num_samples)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.paths[idx]


def mock_create_training_setup(**kwargs):
    device_obj = torch.device("cpu")
    loader = DataLoader(DummyDataset(), batch_size=8)

    model = EmbeddingNet(
        embedding_dim=kwargs.get("embedding_dim", 128),
        backbone_name="resnet18",
        pretrained=False,
    ).to(device_obj)

    loss_fn = get_proxy_anchor_loss(
        num_classes=4,
        embedding_dim=kwargs.get("embedding_dim", 128),
    ).to(device_obj)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()), lr=1e-4
    )

    return {
        "model": model,
        "train_loader": loader,
        "val_loader": loader,
        "seen_test_loader": loader,
        "unseen_test_loader": loader,
        "loss_fn": loss_fn,
        "miner": None,
        "optimizer": optimizer,
        "scheduler": None,
        "device": device_obj,
    }


def test_optuna_hpo_single_trial():
    """Smoke test: Verify Optuna objective function executes 1 trial using mocked setup."""
    study = optuna.create_study(direction="maximize")

    with patch("src.hpo.create_training_setup", side_effect=mock_create_training_setup):
        study.optimize(
            lambda trial: objective(
                trial,
                loss_type="proxy_anchor",
                num_epochs=1,
                patience=1,
                device="cpu",
            ),
            n_trials=1,
        )

    assert len(study.trials) == 1
    assert study.best_value >= 0.0
    assert "lr" in study.best_params
