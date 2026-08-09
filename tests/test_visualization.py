import os
import numpy as np
import pytest
from src.visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_tsne,
)


def test_plot_training_history(tmp_path):
    """Verify plot_training_history creates a PNG file."""
    history = {
        "train_loss": [0.8, 0.5, 0.3],
        "val_loss": [0.9, 0.6, 0.4],
        "val_recall1": [0.4, 0.6, 0.75],
        "best_epoch": 3,
    }
    save_path = str(tmp_path / "test_history.png")
    out_path = plot_training_history(history, save_path=save_path)

    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


def test_plot_confusion_matrix(tmp_path):
    """Verify plot_confusion_matrix creates a PNG file."""
    cm = np.array([[10, 2], [1, 12]])
    class_names = ["Bird_A", "Bird_B"]
    save_path = str(tmp_path / "test_cm.png")
    out_path = plot_confusion_matrix(cm, class_names=class_names, save_path=save_path)

    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


def test_plot_tsne(tmp_path):
    """Verify plot_tsne creates a PNG file."""
    embeddings = np.random.randn(30, 128)
    labels = np.array([0] * 10 + [1] * 10 + [2] * 10)
    class_names = ["Class0", "Class1", "Class2"]
    save_path = str(tmp_path / "test_tsne.png")
    out_path = plot_tsne(embeddings, labels, class_names=class_names, save_path=save_path)

    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0
