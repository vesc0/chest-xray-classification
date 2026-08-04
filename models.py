"""
Model definitions

Architectures with ImageNet-pretrained backbones:
  1. DenseNet-121 (CNN) — the backbone used in the original CheXNet paper
  2. EfficientNet-B4 (CNN) — compound-scaled CNN, strong on medical imaging
  3. ViT-B/16 (Vision Transformer) — from torchvision
  4. SwinV2-B (Shifted-window Transformer)

All output NUM_CLASSES logits (one per pathology) for multi-label classification.
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

        # Load pretrained DenseNet-121 backbone
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = models.densenet121(weights=weights)

        # Replace the original classifier head
        in_features = self.backbone.classifier.in_features  # 1024
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through DenseNet backbone
        return self.backbone(x)


# =============================================================================
# 2. EfficientNet-B4 (compound-scaled CNN)
# =============================================================================
class EfficientNetB4Classifier(nn.Module):
    """
    EfficientNet-B4 with a custom classification head.

    Architecture:
      - EfficientNet-B4 backbone (pretrained on ImageNet)
      - Adaptive average pooling (built into EfficientNet)
      - Dropout → Linear(1792, NUM_CLASSES)

    EfficientNet-B4 is the best trade-off variant for medical imaging at
    224×224: larger capacity than B0-B2 (which underfit on multi-label
    pathology detection) without the diminishing returns and memory cost
    of B5-B7.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES, pretrained: bool = True):
        super().__init__()

        # Load pretrained EfficientNet-B4 backbone
        weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b4(weights=weights)

        # Replace the original classifier head
        in_features = self.backbone.classifier[1].in_features  # 1792
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through EfficientNet backbone
        return self.backbone(x)


# =============================================================================
# 3. ViT-B/16 (ViT)
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

        # Load pretrained ViT-B/16 backbone
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        self.backbone = models.vit_b_16(weights=weights)

        # Replace classification head
        in_features = self.backbone.heads.head.in_features  # 768
        self.backbone.heads.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through ViT backbone
        return self.backbone(x)


# =============================================================================
# 4. SwinV2-B (modern ViT)
# =============================================================================
class SwinV2Classifier(nn.Module):
    """
    Swin Transformer V2-B with custom classification head
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES, pretrained: bool = True):
        super().__init__()

        # Load pretrained SwinV2-B backbone
        weights = models.Swin_V2_B_Weights.DEFAULT if pretrained else None
        self.backbone = models.swin_v2_b(weights=weights)

        # Replace classification head
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through Swin ViT backbone
        return self.backbone(x)


# =============================================================================
# Model factory
# =============================================================================
def build_model(model_name: str, pretrained: bool = True) -> nn.Module:
    """
    Instantiate a model by name.

    Args:
        model_name: One of config.SUPPORTED_MODELS.
        pretrained: Whether to load ImageNet-pretrained weights.

    Returns:
        nn.Module ready for training.
    """
    if model_name == "densenet121":
        return DenseNet121Classifier(pretrained=pretrained)
    elif model_name == "vit_b_16":
        return ViTClassifier(pretrained=pretrained)
    elif model_name == "swin_v2_b":
        return SwinV2Classifier(pretrained=pretrained)
    elif model_name == "efficientnet_b4":
        return EfficientNetB4Classifier(pretrained=pretrained)
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. Supported: {config.SUPPORTED_MODELS}"
        )
