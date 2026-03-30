"""
Model definitions

Two architectures with ImageNet-pretrained backbones:
  1. DenseNet-121 (CNN)  — the backbone used in the original CheXNet paper
  2. ViT-B/16 (Vision Transformer) — from torchvision

Both output NUM_CLASSES logits (one per pathology) for multi-label classification.
"""

import torch
import torch.nn as nn
from torchvision import models

import config


# =============================================================================
# 1. DenseNet-121 (CNN baseline)
# =============================================================================
class DenseNet121Classifier(nn.Module):
    """
    DenseNet-121 with a custom classification head.

    Architecture:
      - DenseNet-121 backbone (pretrained on ImageNet)
      - Global average pooling (built into DenseNet)
      - Dropout → Linear(1024, NUM_CLASSES)
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = models.densenet121(weights=weights)

        # Replace the original classifier head
        in_features = self.backbone.classifier.in_features  # 1024
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# =============================================================================
# 2. Vision Transformer (ViT-B/16)
# =============================================================================
class ViTClassifier(nn.Module):
    """
    ViT-B/16 with a custom classification head.

    Architecture:
      - ViT-B/16 backbone (pretrained on ImageNet)
      - [CLS] token → Dropout → Linear(768, NUM_CLASSES)
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        self.backbone = models.vit_b_16(weights=weights)

        # Replace the head
        in_features = self.backbone.heads.head.in_features  # 768
        self.backbone.heads.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# =============================================================================
# Factory
# =============================================================================
def build_model(model_name: str, pretrained: bool = True) -> nn.Module:
    """
    Instantiate a model by name.

    Args:
        model_name: One of config.SUPPORTED_MODELS ("densenet121", "vit_b_16").
        pretrained: Whether to load ImageNet-pretrained weights.

    Returns:
        nn.Module ready for training.
    """
    if model_name == "densenet121":
        return DenseNet121Classifier(pretrained=pretrained)
    elif model_name == "vit_b_16":
        return ViTClassifier(pretrained=pretrained)
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. Supported: {config.SUPPORTED_MODELS}"
        )
