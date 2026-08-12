import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EmbeddingNet(nn.Module):
    """ResNet backbone with a custom FC head and L2 normalization.

    Outputs unit vectors.

    Args:
        embedding_dim: Output vector dimension (default: 128).
        backbone_name: ResNet variant to use ('resnet18', 'resnet34', 'resnet50').
        pretrained: Whether to use ImageNet pretrained weights.
        freeze_backbone: Freeze backbone parameters initially.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        backbone_name: str = "resnet34",
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.backbone_name = backbone_name

        # Load selected torchvision ResNet model
        if hasattr(models, backbone_name):
            weights = "DEFAULT" if pretrained else None
            backbone_fn = getattr(models, backbone_name)
            backbone = backbone_fn(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Get features dimension before FC layer
        in_features = backbone.fc.in_features

        # Replace classification FC with Identity
        backbone.fc = nn.Identity()
        self.backbone = backbone

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Custom FC head mapping to embedding_dim
        self.embedding_head = nn.Linear(in_features, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: [B, 3, H, W] -> L2-normalized [B, embedding_dim]."""
        features = self.backbone(x)
        embeddings = self.embedding_head(features)
        return F.normalize(embeddings, p=2, dim=1)

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters for end-to-end fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: torch.device | str = "cpu",
        default_backbone: str = "resnet34",
        default_dim: int = 128,
    ) -> "EmbeddingNet":
        """Factory method to load EmbeddingNet from checkpoint file."""
        return load_resnet_model(checkpoint_path, device, default_backbone, default_dim)


def load_resnet_model(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
    default_backbone: str = "resnet34",
    default_dim: int = 128,
) -> EmbeddingNet:
    """Instantiates EmbeddingNet model and loads fine-tuned weights from a checkpoint file.

    Infers embedding_dim and backbone parameters from saved state_dict.

    Args:
        checkpoint_path: Filepath to PyTorch checkpoint (.pt).
        device: Computation device (CPU or CUDA).
        default_backbone: Fallback ResNet backbone if not specified in checkpoint.
        default_dim: Fallback embedding dimension if not specified in checkpoint.

    Returns:
        Evaluated EmbeddingNet model loaded on target device.
    """
    comp_device = torch.device(device)
    path_obj = Path(checkpoint_path)
    if path_obj.exists():
        checkpoint = torch.load(path_obj, map_location=comp_device, weights_only=False)
        embedding_dim = default_dim
        backbone_name = default_backbone
        state_dict = None

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            embedding_dim = checkpoint.get("embedding_dim", state_dict["embedding_head.weight"].shape[0])
            backbone_name = checkpoint.get("backbone_name", default_backbone)
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
            if "embedding_head.weight" in state_dict:
                embedding_dim = state_dict["embedding_head.weight"].shape[0]

        model = EmbeddingNet(embedding_dim=embedding_dim, backbone_name=backbone_name)
        if state_dict is not None:
            model.load_state_dict(state_dict)
        print(f"Loaded ResNet checkpoint from '{checkpoint_path}' (embedding_dim={embedding_dim})")
    else:
        model = EmbeddingNet(embedding_dim=default_dim, backbone_name=default_backbone)
        print(f"Warning: Checkpoint '{checkpoint_path}' not found. Using default pretrained backbone weights.")

    model.to(comp_device)
    model.eval()
    return model

