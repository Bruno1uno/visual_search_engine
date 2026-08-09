import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_recall_at_k(
    embeddings_matrix: torch.Tensor,
    labels_tensor: torch.Tensor,
    k_values: tuple[int, ...] = (1,),
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


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | str,
    k_values: tuple[int, ...] = (1,),
    loss_fn: nn.Module | None = None,
    miner: nn.Module | None = None,
) -> tuple[dict[int, float], float]:
    """Prepares evaluation data and calculates Recall@K and optional validation loss.

    Sets model to eval mode, extracts all embeddings, calculates average validation loss
    (if loss_fn is provided), and delegates metric math to compute_recall_at_k.

    Args:
        model: EmbeddingNet model instance.
        dataloader: DataLoader for validation/test data.
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
