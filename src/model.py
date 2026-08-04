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
