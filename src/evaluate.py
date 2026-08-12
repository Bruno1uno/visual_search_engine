import os
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, confusion_matrix

from src.dataset import get_cub200_dataloaders
from src.model import load_resnet_model
from src.visualization import plot_confusion_matrix, plot_tsne


def compute_recall_at_k(
    embeddings_matrix: torch.Tensor,
    labels_tensor: torch.Tensor,
    k_values: tuple[int, ...] = (1, 2, 4, 8),
) -> dict[int, float]:
    """Pure mathematical function to compute Recall@K from embeddings and labels.

    Args:
        embeddings_matrix: L2-normalized 2D embeddings tensor of shape [N, D].
        labels_tensor: 1D class labels tensor of shape [N].
        k_values: Tuple of K values to evaluate (e.g. (1, 2, 4, 8)).

    Returns:
        Dict mapping K -> Recall@K score (float between 0.0 and 1.0).
    """
    num_samples = embeddings_matrix.shape[0]

    if num_samples <= 1:
        return {k: 0.0 for k in k_values}

    # Pairwise Cosine Similarity matrix [N, N]
    sim_matrix = torch.mm(embeddings_matrix, embeddings_matrix.t())

    # Set self-similarity to -inf so query image does not match itself
    sim_matrix.fill_diagonal_(-float("inf"))

    max_k = min(max(k_values), num_samples - 1)
    top_k_indices = torch.topk(sim_matrix, k=max_k, dim=1).indices  # [N, max_k]

    recalls = {}
    for k in k_values:
        k_eff = min(k, max_k)
        retrieved_labels = labels_tensor[top_k_indices[:, :k_eff]]  # [N, k_eff]
        query_labels = labels_tensor.unsqueeze(1)                   # [N, 1]

        # Check if true label appears in top-K retrieved labels for each query
        matches = (retrieved_labels == query_labels).any(dim=1)      # [N]
        recall = matches.float().mean().item()
        recalls[k] = recall

    return recalls


def compute_map(
    embeddings_matrix: torch.Tensor,
    labels_tensor: torch.Tensor,
) -> float:
    """Computes mean Average Precision (mAP) for image retrieval across queries.

    Args:
        embeddings_matrix: L2-normalized 2D embeddings tensor of shape [N, D].
        labels_tensor: 1D class labels tensor of shape [N].

    Returns:
        Mean Average Precision float score (between 0.0 and 1.0).
    """
    num_samples = embeddings_matrix.shape[0]
    if num_samples <= 1:
        return 0.0

    # Pairwise similarity matrix [N, N]
    sim_matrix = torch.mm(embeddings_matrix, embeddings_matrix.t())

    average_precisions = []
    for i in range(num_samples):
        query_label = labels_tensor[i]

        # Exclude self query item from candidates
        mask = torch.arange(num_samples) != i
        candidate_sims = sim_matrix[i, mask]
        candidate_labels = labels_tensor[mask]

        num_candidates = candidate_sims.shape[0]
        sorted_indices = torch.argsort(candidate_sims, descending=True)
        retrieved_labels = candidate_labels[sorted_indices]

        # Binary indicator mask of relevant items (same class label as query)
        rel_mask = (retrieved_labels == query_label).float()
        num_rel = rel_mask.sum().item()

        if num_rel == 0:
            continue

        # Precision@rank r = (relevant items found up to rank r) / r
        cum_rel = torch.cumsum(rel_mask, dim=0)
        ranks = torch.arange(1, num_candidates + 1, dtype=torch.float32)
        precision_at_r = cum_rel / ranks

        # Average Precision for query i
        ap = (precision_at_r * rel_mask).sum().item() / num_rel
        average_precisions.append(ap)

    return float(np.mean(average_precisions)) if average_precisions else 0.0


def compute_nmi(
    embeddings_matrix: torch.Tensor,
    labels_tensor: torch.Tensor,
    n_clusters: int | None = None,
) -> float:
    """Computes Normalized Mutual Information (NMI) score via K-Means clustering.

    Args:
        embeddings_matrix: L2-normalized embeddings tensor [N, D].
        labels_tensor: Ground truth class labels tensor [N].
        n_clusters: Optional number of clusters (defaults to number of unique class labels).

    Returns:
        NMI score float (between 0.0 and 1.0).
    """
    embeds_np = embeddings_matrix.numpy() if isinstance(embeddings_matrix, torch.Tensor) else embeddings_matrix
    labels_np = labels_tensor.numpy() if isinstance(labels_tensor, torch.Tensor) else labels_tensor

    unique_classes = np.unique(labels_np)
    k = n_clusters or len(unique_classes)

    if len(embeds_np) < k or k <= 1:
        return 0.0

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeds_np)

    return float(normalized_mutual_info_score(labels_np, cluster_labels))


@torch.inference_mode()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | str,
    k_values: tuple[int, ...] = (1,),
    loss_fn: nn.Module | None = None,
    miner: nn.Module | None = None,
) -> tuple[dict[int, float], float]:
    """Prepares evaluation data and calculates Recall@K and optional validation loss during training.

    Args:
        model: EmbeddingNet model instance.
        dataloader: DataLoader for validation data.
        device: Computation device ("cuda" or "cpu").
        k_values: Tuple of K values to evaluate (default: (1,)).
        loss_fn: Optional loss function instance to compute validation loss.
        miner: Optional miner instance for Triplet loss.

    Returns:
        Tuple of (recalls_dict, avg_val_loss).
    """
    model.eval()

    all_embeddings = []
    all_labels = []
    total_val_loss = 0.0
    num_batches = 0

    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        embeddings = model(images)

        if loss_fn is not None:
            if miner is not None:
                mined_triplets = miner(embeddings, labels)
                loss = loss_fn(embeddings, labels, mined_triplets)
            else:
                loss = loss_fn(embeddings, labels)
            total_val_loss += loss.item()
            num_batches += 1

        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels.cpu())

    embeddings_matrix = torch.cat(all_embeddings, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    recalls = compute_recall_at_k(embeddings_matrix, labels_tensor, k_values=k_values)
    avg_val_loss = (total_val_loss / max(num_batches, 1)) if loss_fn is not None else 0.0

    return recalls, avg_val_loss


@torch.inference_mode()
def extract_embeddings_and_labels(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extracts all L2-normalized embeddings and class labels from a DataLoader.

    Args:
        model: EmbeddingNet model instance.
        dataloader: Target PyTorch DataLoader.
        device: Computation device.

    Returns:
        Tuple of (embeddings_matrix [N, D], labels_tensor [N]).
    """
    model.eval()
    comp_device = torch.device(device)

    all_embeddings = []
    all_labels = []

    for images, labels, _ in dataloader:
        images = images.to(comp_device)
        embeddings = model(images)

        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels.cpu())

    embeddings_matrix = torch.cat(all_embeddings, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    return embeddings_matrix, labels_tensor


def evaluate_seen_test_split(
    embeddings_matrix: torch.Tensor,
    labels_tensor: torch.Tensor,
    save_plot_path: str = "plots/confusion_seen.png",
    save_tsne_path: str | None = "plots/tsne_seen.png",
    loss_name: str = "proxy_anchor",
) -> dict[str, float]:
    """Sanity-check evaluation on seen test split (classes 1-100).

    Computes 1-NN classification accuracy, plots confusion matrix, and optionally plots t-SNE scatter.

    CRITICAL RULE (AGENTS.md):
        k-NN accuracy and Confusion Matrix are calculated ONLY on seen test split (1-100),
        never on unseen classes, because fixed classification labels exist only for seen classes.

    Args:
        embeddings_matrix: L2-normalized embeddings tensor [N, D].
        labels_tensor: 1D class labels tensor [N].
        save_plot_path: Filepath to save confusion matrix heatmap.
        save_tsne_path: Optional filepath to save t-SNE scatter plot for seen classes.
        loss_name: Identifier of loss function used (e.g. 'proxy_anchor' or 'triplet').

    Returns:
        Dict containing 'knn_accuracy_top1'.
    """
    num_samples = embeddings_matrix.shape[0]
    if num_samples <= 1:
        return {"knn_accuracy_top1": 0.0}

    # Pairwise similarity matrix [N, N]
    sim_matrix = torch.mm(embeddings_matrix, embeddings_matrix.t())
    sim_matrix.fill_diagonal_(-float("inf"))

    # Top-1 nearest neighbor index
    nn_indices = torch.argmax(sim_matrix, dim=1)
    predicted_labels = labels_tensor[nn_indices]

    correct = (predicted_labels == labels_tensor).float().sum().item()
    accuracy = correct / num_samples

    # Compute confusion matrix and plot via src.visualization
    labels_np = labels_tensor.numpy()
    preds_np = predicted_labels.numpy()
    cm = confusion_matrix(labels_np, preds_np)

    plot_confusion_matrix(
        cm,
        save_path=save_plot_path,
        title=f"Seen Classes (1-100) 1-NN Confusion Matrix [{loss_name}]",
    )

    if save_tsne_path:
        embeds_np = embeddings_matrix.numpy()
        plot_tsne(
            embeddings=embeds_np,
            labels=labels_np,
            save_path=save_tsne_path,
            title=f"t-SNE Projection on Seen Test Classes (1-100) [{loss_name}]",
            max_classes=15,
        )

    return {"knn_accuracy_top1": float(accuracy)}


def evaluate_unseen_test_split(
    embeddings_matrix: torch.Tensor,
    labels_tensor: torch.Tensor,
    save_plot_path: str = "plots/tsne_unseen.png",
    k_values: tuple[int, ...] = (1, 2, 4, 8),
    loss_name: str = "proxy_anchor",
) -> dict[str, float]:
    """Zero-Shot evaluation on unseen test split (classes 101-200).

    Evaluates generalization metrics: Recall@K, mAP, NMI, and plots t-SNE scatter.

    Args:
        embeddings_matrix: L2-normalized embeddings tensor [N, D].
        labels_tensor: 1D class labels tensor [N].
        save_plot_path: Filepath to save t-SNE plot.
        k_values: Tuple of K values for Recall@K evaluation.
        loss_name: Identifier of loss function used (e.g. 'proxy_anchor' or 'triplet').

    Returns:
        Dict containing Recall@1, Recall@2, Recall@4, Recall@8, mAP, and NMI.
    """
    recalls = compute_recall_at_k(embeddings_matrix, labels_tensor, k_values=k_values)
    map_score = compute_map(embeddings_matrix, labels_tensor)
    nmi_score = compute_nmi(embeddings_matrix, labels_tensor)

    results = {
        "recall_at_1": recalls.get(1, 0.0),
        "recall_at_2": recalls.get(2, 0.0),
        "recall_at_4": recalls.get(4, 0.0),
        "recall_at_8": recalls.get(8, 0.0),
        "mAP": map_score,
        "NMI": nmi_score,
    }

    # Generate t-SNE scatter plot via src.visualization
    embeds_np = embeddings_matrix.numpy()
    labels_np = labels_tensor.numpy()

    plot_tsne(
        embeddings=embeds_np,
        labels=labels_np,
        save_path=save_plot_path,
        title=f"t-SNE Projection on Unseen Test Classes (101-200) [{loss_name}]",
        max_classes=15,
    )

    return results


def run_evaluation(
    checkpoint_path: str = "checkpoints/best_resnet34_proxy_anchor.pt",
    data_dir: str = "data",
    output_metrics_path: str | None = None,
    loss_name: str | None = None,
    batch_size: int = 64,
    device: str | None = None,
) -> dict:
    """Master offline evaluation pipeline.

    1. Loads trained ResNet encoder checkpoint.
    2. Runs sanity check on seen test split (classes 1-100) -> k-NN accuracy & confusion matrix.
    3. Runs zero-shot evaluation on unseen test split (classes 101-200) -> Recall@K, mAP, NMI & t-SNE.
    4. Saves visualizations to plots/ and results to metrics/ as JSON for Streamlit and FastAPI.

    Args:
        checkpoint_path: Path to trained PyTorch model checkpoint (.pt).
        data_dir: Local dataset directory path.
        output_metrics_path: Output JSON file for evaluation metrics.
        loss_name: Name identifier for loss function ('proxy_anchor' or 'triplet').
        batch_size: DataLoader batch size.
        device: Target device (CPU or CUDA).

    Returns:
        Dict containing full evaluation metrics.
    """
    comp_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Infer loss_name if not provided
    if loss_name is None:
        if "proxy_anchor" in checkpoint_path.lower():
            loss_name = "proxy_anchor"
        elif "triplet" in checkpoint_path.lower():
            loss_name = "triplet"
        else:
            loss_name = "resnet"

    # Infer plot and metrics output filepaths
    confusion_path = f"plots/confusion_seen_{loss_name}.png"
    tsne_seen_path = f"plots/tsne_seen_{loss_name}.png"
    tsne_path = f"plots/tsne_unseen_{loss_name}.png"
    metrics_path = output_metrics_path or f"metrics/eval_results_{loss_name}.json"

    print(f"Starting evaluation [{loss_name}] on device: {comp_device}")

    # Load model weights
    model = load_resnet_model(checkpoint_path, comp_device)

    # Get DataLoaders
    _, _, seen_test_loader, unseen_test_loader = get_cub200_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        use_m_per_class_sampler=False,
    )

    # Evaluate Seen Test Split (1-100)
    print(f"\n--- Evaluating Seen Test Split (Classes 1-100) [{loss_name}] ---")
    seen_embeds, seen_labels = extract_embeddings_and_labels(model, seen_test_loader, comp_device)
    seen_results = evaluate_seen_test_split(
        seen_embeds,
        seen_labels,
        save_plot_path=confusion_path,
        save_tsne_path=tsne_seen_path,
        loss_name=loss_name,
    )
    print(f"  Seen Test 1-NN Accuracy: {seen_results['knn_accuracy_top1']:.2%}")

    # Evaluate Unseen Test Split (101-200)
    print(f"\n--- Evaluating Unseen Test Split (Classes 101-200) [{loss_name}] ---")
    unseen_embeds, unseen_labels = extract_embeddings_and_labels(model, unseen_test_loader, comp_device)
    unseen_results = evaluate_unseen_test_split(
        unseen_embeds, unseen_labels, save_plot_path=tsne_path, loss_name=loss_name
    )

    print(f"  Recall@1: {unseen_results['recall_at_1']:.2%}")
    print(f"  Recall@2: {unseen_results['recall_at_2']:.2%}")
    print(f"  Recall@4: {unseen_results['recall_at_4']:.2%}")
    print(f"  Recall@8: {unseen_results['recall_at_8']:.2%}")
    print(f"  mAP:      {unseen_results['mAP']:.4f}")
    print(f"  NMI:      {unseen_results['NMI']:.4f}")

    # Save combined evaluation report JSON
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)

    eval_report = {
        "loss_name": loss_name,
        "checkpoint_path": checkpoint_path,
        "embedding_dim": getattr(model, "embedding_dim", 128),
        "seen_test_metrics": seen_results,
        "unseen_test_metrics": unseen_results,
        "confusion_plot_path": confusion_path,
        "tsne_seen_plot_path": tsne_seen_path,
        "tsne_plot_path": tsne_path,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)

    print(f"\nSaved evaluation metrics to {metrics_path}")
    return eval_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline evaluation for ResNet metric learning encoder.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/best_resnet34_proxy_anchor.pt",
        help="Path to model checkpoint.",
    )
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing CUB-200 dataset.")
    parser.add_argument(
        "--output_metrics_path",
        type=str,
        default=None,
        help="Output metrics JSON path (defaults to metrics/eval_results_<loss_name>.json).",
    )
    parser.add_argument(
        "--loss_name",
        type=str,
        default=None,
        help="Identifier of loss function ('proxy_anchor' or 'triplet'). Auto-inferred if None.",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation.")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu).")

    args = parser.parse_args()
    run_evaluation(
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        output_metrics_path=args.output_metrics_path,
        loss_name=args.loss_name,
        batch_size=args.batch_size,
        device=args.device,
    )
