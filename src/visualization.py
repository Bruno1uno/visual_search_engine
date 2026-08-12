import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless / script execution
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE


def plot_training_history(
    history: dict,
    save_path: str = "plots/training_history.png",
    title: str = "Training Progress",
) -> str:
    """Plots training and validation loss curves alongside validation Recall@1.

    Args:
        history: Dict containing 'train_loss', 'val_loss', 'val_recall1'.
        save_path: Output PNG filepath (default: 'plots/training_history.png').
        title: Overall figure title.

    Returns:
        Absolute filepath to the saved figure PNG.
    """
    path_obj = Path(save_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history.get("train_loss", [])) + 1)
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    val_recall1 = history.get("val_recall1", [])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Subplot 1: Loss curves
    if train_loss:
        ax1.plot(epochs, train_loss, label="Train Loss", color="#1f77b4", linewidth=2, marker="o")
    if val_loss:
        ax1.plot(epochs, val_loss, label="Val Loss", color="#ff7f0e", linewidth=2, linestyle="--", marker="s")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves (Train vs Validation)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Subplot 2: Validation Recall@1
    if val_recall1:
        ax2.plot(epochs, [r * 100 for r in val_recall1], label="Val Recall@1 (%)", color="#2ca02c", linewidth=2, marker="^")
        best_epoch = history.get("best_epoch", np.argmax(val_recall1) + 1 if val_recall1 else 1)
        best_val = max(val_recall1) * 100 if val_recall1 else 0
        ax2.axvline(x=best_epoch, color="#d62728", linestyle=":", label=f"Best Epoch ({best_epoch}: {best_val:.2f}%)")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Recall@1 (%)")
    ax2.set_title("Validation Recall@1 Score")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(path_obj, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved training history plot to {path_obj}")
    return str(path_obj.resolve())


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] | None = None,
    save_path: str = "plots/confusion_seen.png",
    title: str = "Seen Classes Confusion Matrix",
) -> str:
    """Plots and saves confusion matrix heatmap for seen test classes.

    Args:
        cm: 2D numpy confusion matrix array [C, C].
        class_names: Optional list of class names.
        save_path: Output PNG filepath.
        title: Plot title.

    Returns:
        Absolute filepath to the saved figure PNG.
    """
    path_obj = Path(save_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=cm.shape[0] <= 20,  # Show numbers only if matrix is not too dense
        fmt="d",
        cmap="Blues",
        xticklabels=class_names if class_names and len(class_names) <= 20 else False,
        yticklabels=class_names if class_names and len(class_names) <= 20 else False,
        ax=ax,
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")

    plt.tight_layout()
    plt.savefig(path_obj, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved confusion matrix plot to {path_obj}")
    return str(path_obj.resolve())


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: list[str] | None = None,
    save_path: str = "plots/tsne_unseen.png",
    title: str = "t-SNE Embedding Space Projection (Unseen Classes)",
    max_classes: int = 15,
) -> str:
    """Computes t-SNE 2D projection of embeddings and saves scatter plot.

    Args:
        embeddings: 2D numpy array of embeddings [N, D].
        labels: 1D numpy array of class labels [N].
        class_names: Optional mapping from label ID to class name.
        save_path: Output PNG filepath.
        title: Plot title.
        max_classes: Max number of classes to visualize for clarity.

    Returns:
        Absolute filepath to the saved figure PNG.
    """
    path_obj = Path(save_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Filter to top max_classes for clean visual scatter
    unique_labels = np.unique(labels)
    if len(unique_labels) > max_classes:
        selected_labels = unique_labels[:max_classes]
        mask = np.isin(labels, selected_labels)
        embeddings = embeddings[mask]
        labels = labels[mask]

    # Compute 2D t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    embeddings_2d = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_selected = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_selected)))

    for lbl, color in zip(unique_selected, colors):
        pts = embeddings_2d[labels == lbl]
        label_name = class_names[lbl] if class_names and lbl < len(class_names) else f"Class {lbl}"
        ax.scatter(pts[:, 0], pts[:, 1], color=color, label=label_name, alpha=0.7, edgecolors="none", s=30)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(path_obj, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved t-SNE plot to {path_obj}")
    return str(path_obj.resolve())
