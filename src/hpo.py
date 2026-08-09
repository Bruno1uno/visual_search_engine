import os
import argparse
import json
import yaml
import optuna
import torch

from src.train import create_training_setup, train_one_epoch
from src.evaluate import validate


def objective(
    trial: optuna.Trial,
    loss_type: str = "proxy_anchor",
    data_dir: str = "data",
    num_epochs: int = 5,
    patience: int = 3,
    device: str | None = None,
) -> float:
    """Optuna objective function for tuning metric learning encoder hyperparameters.

    Args:
        trial: Optuna Trial instance.
        loss_type: 'proxy_anchor' or 'triplet'.
        data_dir: CUB-200 dataset directory.
        num_epochs: Max epochs per trial.
        patience: Early stopping patience per trial.
        device: Computation device ("cuda" or "cpu").

    Returns:
        Best Validation Recall@1 score for the trial.
    """
    # Hyperparameter search space
    lr = trial.suggest_float("lr", 5e-5, 5e-4, log=True)
    backbone_name = trial.suggest_categorical("backbone_name", ["resnet18", "resnet34"])
    embedding_dim = trial.suggest_categorical("embedding_dim", [128, 256])
    batch_size = trial.suggest_categorical("batch_size", [32, 64])

    if loss_type == "triplet":
        margin = trial.suggest_float("margin", 0.1, 0.5)
        alpha = 32.0
    elif loss_type == "proxy_anchor":
        margin = trial.suggest_float("margin", 0.05, 0.3)
        alpha = trial.suggest_float("alpha", 16.0, 48.0)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    trial_num = trial.number
    params_str = f"backbone={backbone_name}, dim={embedding_dim}, batch={batch_size}, lr={lr:.2e}, margin={margin:.4f}"
    if loss_type == "proxy_anchor":
        params_str += f", alpha={alpha:.2f}"

    print(f"\n" + "=" * 60)
    print(f"[Trial {trial_num}] Hyperparameters: {params_str}")
    print("=" * 60)

    # Instantiate training setup
    setup = create_training_setup(
        loss_type=loss_type,
        backbone_name=backbone_name,
        embedding_dim=embedding_dim,
        batch_size=batch_size,
        learning_rate=lr,
        margin=margin,
        alpha=alpha,
        data_dir=data_dir,
        device=device,
    )

    model = setup["model"]
    train_loader = setup["train_loader"]
    val_loader = setup["val_loader"]
    loss_fn = setup["loss_fn"]
    miner = setup["miner"]
    optimizer = setup["optimizer"]
    device_obj = setup["device"]

    best_val_recall1 = -1.0
    epochs_without_improvement = 0

    # Epoch execution with Optuna Pruning
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            miner=miner,
            optimizer=optimizer,
            device=device_obj,
        )

        val_recalls, val_loss = validate(
            model=model,
            dataloader=val_loader,
            device=device_obj,
            k_values=(1,),
            loss_fn=loss_fn,
            miner=miner,
        )
        val_recall1 = val_recalls[1]

        if val_recall1 > best_val_recall1:
            best_val_recall1 = val_recall1
            epochs_without_improvement = 0
            status_tag = " (New Best!)"
        else:
            epochs_without_improvement += 1
            status_tag = ""

        print(
            f"Trial {trial_num:02d} | Epoch {epoch:02d}/{num_epochs:02d} -> "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Recall@1: {val_recall1:.2%}{status_tag}"
        )

        # Report to Optuna and check pruning condition
        trial.report(val_recall1, epoch)
        if trial.should_prune():
            print(f"--> [Trial {trial_num:02d}] PRUNED at epoch {epoch} (Val Recall@1: {val_recall1:.2%})\n")
            raise optuna.TrialPruned()

        if epochs_without_improvement >= patience:
            print(f"--> [Trial {trial_num:02d}] Early stopped after {patience} epochs without improvement.\n")
            break

    print(f"--> [Trial {trial_num:02d}] Completed. Best Val Recall@1: {best_val_recall1:.2%}\n")
    return best_val_recall1


def run_hpo(
    loss_type: str = "proxy_anchor",
    n_trials: int = 10,
    num_epochs: int = 5,
    patience: int = 3,
    data_dir: str = "data",
    config_dir: str = "configs",
    db_storage: str = "sqlite:///hpo_study.db",
    device: str | None = None,
) -> dict:
    """Executes Optuna study, saves best config to YAML and summary to JSON.

    Returns:
        Dict of best hyperparameters.
    """
    study_name = f"cub200_metric_hpo_{loss_type}"

    # Use MedianPruner for aggressive early trial pruning
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2)

    study = optuna.create_study(
        direction="maximize",
        storage=db_storage,
        study_name=study_name,
        pruner=pruner,
        load_if_exists=True,
    )

    print(f"Starting HPO study '{study_name}' for {n_trials} trials...")

    study.optimize(
        lambda trial: objective(
            trial,
            loss_type=loss_type,
            data_dir=data_dir,
            num_epochs=num_epochs,
            patience=patience,
            device=device,
        ),
        n_trials=n_trials,
    )

    best_trial = study.best_trial
    best_params = best_trial.params
    print(f"\nHPO Complete. Best Val Recall@1: {best_trial.value:.2%}")
    print(f"Best Hyperparameters: {best_params}")

    # Save best config to YAML
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, f"best_config_{loss_type}.yaml")
    with open(config_path, "w") as f:
        yaml.dump({"loss_type": loss_type, "hyperparameters": best_params}, f, indent=2)
    print(f"Saved best YAML config to {config_path}")

    # Save study summary to JSON
    os.makedirs("metrics", exist_ok=True)
    summary_path = os.path.join("metrics", f"hpo_summary_{loss_type}.json")
    summary_data = {
        "study_name": study_name,
        "best_val_recall1": best_trial.value,
        "best_params": best_params,
        "n_trials": len(study.trials),
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved HPO summary JSON to {summary_path}")

    return best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Optimization")
    parser.add_argument("--loss_type", type=str, default="proxy_anchor", choices=["proxy_anchor", "triplet"])
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    run_hpo(
        loss_type=args.loss_type,
        n_trials=args.n_trials,
        num_epochs=args.epochs,
        patience=args.patience,
    )
