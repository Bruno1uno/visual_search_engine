import os
import argparse
import json
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import EmbeddingNet
from src.loss import get_triplet_loss_and_miner, get_proxy_anchor_loss
from src.dataset import get_cub200_dataloaders
from src.evaluate import validate


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    miner: nn.Module | None,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
) -> float:
    """Trains the model for one epoch and returns average train loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(train_loader, desc="  Train", leave=False)
    for images, labels, _ in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        embeddings = model(images)

        if miner is not None:
            mined_triplets = miner(embeddings, labels)
            loss = loss_fn(embeddings, labels, mined_triplets)
        else:
            loss = loss_fn(embeddings, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(num_batches, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    miner: nn.Module | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device | str,
    num_epochs: int = 20,
    patience: int = 5,
    checkpoint_path: str = "checkpoints/best_model.pt",
    loss_type: str = "proxy_anchor",
) -> dict[str, list]:
    """Clean training loop. Receives pre-instantiated dependencies.

    Args:
        model: EmbeddingNet model instance.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        loss_fn: TripletMarginLoss or ProxyAnchorLoss instance.
        miner: Optional TripletMarginMiner instance (None for ProxyAnchor).
        optimizer: PyTorch Optimizer (AdamW).
        scheduler: Optional PyTorch LR scheduler.
        device: Computation device.
        num_epochs: Maximum epochs to train.
        patience: Early stopping patience based on validation Recall@1.
        checkpoint_path: Filepath to save best model weights.
        loss_type: Loss type string identifier ('proxy_anchor' or 'triplet').

    Returns:
        Dict history containing train_loss, val_recall1, best_epoch.
    """
    ckpt_path = Path(checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    Path("metrics").mkdir(parents=True, exist_ok=True)
    Path("plots").mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "val_recall1": [], "best_epoch": 0}
    best_val_recall1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch:02d}/{num_epochs}")

        # 1. Train single epoch
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            miner=miner,
            optimizer=optimizer,
            device=device,
        )

        # 2. Validate Recall@1 and Validation Loss
        val_metrics, val_loss = validate(
            model=model,
            dataloader=val_loader,
            device=device,
            k_values=(1,),
            loss_fn=loss_fn,
            miner=miner,
        )
        val_recall1 = val_metrics[1]

        if scheduler is not None:
            scheduler.step(val_recall1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_recall1"].append(val_recall1)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Recall@1: {val_recall1:.2%} | "
            f"LR: {lr:.2e}"
        )

        # 3. Checkpointing based on Validation Recall@1
        if val_recall1 > best_val_recall1:
            best_val_recall1 = val_recall1
            history["best_epoch"] = epoch
            epochs_without_improvement = 0

            checkpoint_payload = {
                "epoch": epoch,
                "embedding_dim": getattr(model, "embedding_dim", 128),
                "loss_type": loss_type,
                "best_val_recall1": best_val_recall1,
                "model_state_dict": model.state_dict(),
            }
            if hasattr(loss_fn, "state_dict") and loss_type == "proxy_anchor":
                checkpoint_payload["loss_state_dict"] = loss_fn.state_dict()

            torch.save(checkpoint_payload, ckpt_path)
            print(f"  Saved checkpoint to {ckpt_path} (Best Val Recall@1: {best_val_recall1:.2%})")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement in Val Recall@1 ({epochs_without_improvement}/{patience})")

        # 4. Early Stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    # Save training history JSON
    history_file = Path("metrics") / f"history_{loss_type}.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

    # Plot training history curves
    try:
        from src.visualization import plot_training_history
        plot_path = f"plots/training_history_{loss_type}.png"
        plot_training_history(history, save_path=plot_path, title=f"Training Progress ({loss_type})")
    except Exception as e:
        print(f"Warning: Could not plot training history: {e}")

    # Restore best checkpoint
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"\nTraining complete. Best checkpoint restored from epoch {history['best_epoch']} with Val Recall@1: {best_val_recall1:.2%}")

    return history


def create_training_setup(
    loss_type: str = "proxy_anchor",
    backbone_name: str = "resnet34",
    embedding_dim: int = 128,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    margin: float = 0.1,
    alpha: float = 32.0,
    data_dir: str = "data",
    device: str | None = None,
):
    """Factory helper to build and return all instantiated training components."""
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_loader, val_loader, seen_test_loader, unseen_test_loader = get_cub200_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        samples_per_class=4,
    )

    model = EmbeddingNet(
        embedding_dim=embedding_dim,
        backbone_name=backbone_name,
        pretrained=True,
    ).to(device_obj)

    miner = None
    if loss_type == "triplet":
        loss_fn, miner = get_triplet_loss_and_miner(margin=margin, type_of_triplets="semi-hard")
        trainable_params = list(model.parameters())
    elif loss_type == "proxy_anchor":
        loss_fn = get_proxy_anchor_loss(
            num_classes=100,
            embedding_dim=embedding_dim,
            margin=margin,
            alpha=alpha,
        ).to(device_obj)
        trainable_params = list(model.parameters()) + list(loss_fn.parameters())
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    return {
        "model": model,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "seen_test_loader": seen_test_loader,
        "unseen_test_loader": unseen_test_loader,
        "loss_fn": loss_fn,
        "miner": miner,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "device": device_obj,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Train Metric Learning Encoder on CUB-200-2011.\n\n"
            "Usage modes:\n"
            "  1. Manual parameters: Pass hyperparameters via CLI flags (e.g. --lr 1e-4 --margin 0.1)\n"
            "  2. Optuna HPO config: Pass --config_path configs/best_config_proxy_anchor.yaml to auto-load best hyperparameters."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Optional path to YAML config file exported by Optuna HPO (e.g. configs/best_config_proxy_anchor.yaml). Overrides CLI hyperparameter flags if specified.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="proxy_anchor",
        choices=["proxy_anchor", "triplet"],
        help="Loss function type ('proxy_anchor' or 'triplet'). Default: proxy_anchor.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet34",
        choices=["resnet18", "resnet34", "resnet50"],
        help="ResNet backbone variant. Default: resnet34.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="Output embedding dimension. Default: 128.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximum training epochs. Default: 20.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="DataLoader batch size. Default: 64.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for AdamW optimizer. Default: 1e-4.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.1,
        help="Loss margin parameter. Default: 0.1.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=32.0,
        help="ProxyAnchor scaling exponent parameter. Default: 32.0.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (epochs without Val Recall@1 improvement). Default: 5.",
    )
    args = parser.parse_args()

    # If YAML config is provided (e.g. from HPO study), load parameters from it
    if args.config_path and Path(args.config_path).exists():
        print(f"Loading hyperparameter configuration from {args.config_path}...")
        with open(args.config_path, "r") as f:
            yaml_cfg = yaml.safe_load(f)
        
        loss_type = yaml_cfg.get("loss_type", args.loss_type)
        params = yaml_cfg.get("hyperparameters", {})

        backbone_name = params.get("backbone_name", args.backbone)
        embedding_dim = params.get("embedding_dim", args.embedding_dim)
        batch_size = params.get("batch_size", args.batch_size)
        learning_rate = params.get("lr", args.lr)
        margin = params.get("margin", args.margin)
        alpha = params.get("alpha", args.alpha)
    else:
        loss_type = args.loss_type
        backbone_name = args.backbone
        embedding_dim = args.embedding_dim
        batch_size = args.batch_size
        learning_rate = args.lr
        margin = args.margin
        alpha = args.alpha

    setup = create_training_setup(
        loss_type=loss_type,
        backbone_name=backbone_name,
        embedding_dim=embedding_dim,
        batch_size=batch_size,
        learning_rate=learning_rate,
        margin=margin,
        alpha=alpha,
    )

    checkpoint_path = f"checkpoints/best_{backbone_name}_{loss_type}.pt"

    train_model(
        model=setup["model"],
        train_loader=setup["train_loader"],
        val_loader=setup["val_loader"],
        loss_fn=setup["loss_fn"],
        miner=setup["miner"],
        optimizer=setup["optimizer"],
        scheduler=setup["scheduler"],
        device=setup["device"],
        num_epochs=args.epochs,
        patience=args.patience,
        checkpoint_path=checkpoint_path,
        loss_type=loss_type,
    )
