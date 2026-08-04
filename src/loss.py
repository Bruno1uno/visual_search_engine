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

    Should be called only once in the code.
    """
    if type_of_triplets not in VALID_TRIPLET_TYPES:
        raise ValueError(
            f"Invalid type_of_triplets '{type_of_triplets}'. "
            f"Choose from: {sorted(VALID_TRIPLET_TYPES)}"
        )

    loss_fn = losses.TripletMarginLoss(margin=margin)
    miner = miners.TripletMarginMiner(margin=margin, type_of_triplets=type_of_triplets)

    return loss_fn, miner
