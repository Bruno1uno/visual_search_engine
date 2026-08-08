from pytorch_metric_learning import losses, miners

VALID_TRIPLET_TYPES = {"all", "hard", "semi-hard", "easy"}


def get_triplet_loss_and_miner(
    margin: float = 0.2,
    type_of_triplets: str = "semi-hard",
) -> tuple[losses.TripletMarginLoss, miners.TripletMarginMiner]:
    """Creates a (loss_fn, miner) pair for Triplet Margin training.

    Args:
        margin: Triplet loss margin (distance threshold). Default: 0.2.
        type_of_triplets: Mining strategy ('semi-hard', 'hard', 'all', 'easy').

    Returns:
        Tuple of (TripletMarginLoss, TripletMarginMiner).
    """
    if type_of_triplets not in VALID_TRIPLET_TYPES:
        raise ValueError(
            f"Invalid type_of_triplets '{type_of_triplets}'. "
            f"Choose from: {sorted(VALID_TRIPLET_TYPES)}"
        )

    loss_fn = losses.TripletMarginLoss(margin=margin)
    miner = miners.TripletMarginMiner(margin=margin, type_of_triplets=type_of_triplets)

    return loss_fn, miner


def get_proxy_anchor_loss(
    num_classes: int = 100,
    embedding_dim: int = 128,
    margin: float = 0.1,
    alpha: float = 32.0,
) -> losses.ProxyAnchorLoss:
    """Creates a ProxyAnchorLoss instance (Kim et al., CVPR 2020).

    Unlike Triplet Loss, Proxy Anchor Loss does NOT require a miner.
    Instead, it maintains a learnable proxy vector for each class.
    Therefore it must be also passed into the optimizer.

    Args:
        num_classes: Number of training classes (default: 100 seen classes).
        embedding_dim: Dimension of embeddings (default: 128).
        margin: Soft margin parameter delta (default: 0.1).
        alpha: Scaling factor for distance exponent (default: 32.0).

    Returns:
        losses.ProxyAnchorLoss instance.
    """
    return losses.ProxyAnchorLoss(
        num_classes=num_classes,
        embedding_size=embedding_dim,
        margin=margin,
        alpha=alpha,
    )