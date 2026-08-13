"""
Model definitions

One backbone per architecture family, so a result can be attributed to the
architecture rather than to capacity, resolution, or pretraining data
(parameter counts are as built here, with the 14-class head):

  1. DenseNet-121   (7.0M)  CNN baseline — the backbone used in CheXNet
  2. ViT-S/16      (21.7M)  pure Vision Transformer
  3. ConvNeXtV2-T  (27.9M)  modern CNN
  4. SwinV2-T      (27.6M)  modern (hierarchical, windowed) Vision Transformer
  5. MaxViT-T      (30.4M)  hybrid — MBConv plus block/grid attention per stage

The four non-baseline models sit in a 22-31M band, so "modern CNN vs modern
ViT" is a comparison at matched capacity. DenseNet-121 is deliberately *not*
scale-matched: its value is being the exact backbone the CheXNet results were
produced with, which changing the depth would throw away.

Two invariants hold across all five, and both are load-bearing for the
comparison:

  - **ImageNet-1k pretraining only.** No IN21k/IN22k checkpoints, even where
    they exist (ConvNeXtV2 and ViT-S both ship them). DenseNet, SwinV2 and
    MaxViT have no IN21k option here, so 21k anywhere would confound
    architecture with pretraining data — and would specifically flatter the
    pure ViT, the architecture most dependent on pretraining scale.
  - **Identical preprocessing.** Every backbone below expects 224x224 input
    with ImageNet mean/std, so dataset.py can apply one shared transform.
    This is why ViT-S uses DeiT weights (see ViTSmallClassifier).

Three come from torchvision; ViT-S and ConvNeXtV2 come from timm, which is the
only source for them. The timm tags are pinned below rather than resolved by
`.DEFAULT`, because a tag is what decides which of several checkpoints a name
means.

All output NUM_CLASSES logits (one per pathology) for multi-label classification.
"""

import timm
import torch
import torch.nn as nn
from torchvision import models

import config


# Pinned timm weight tags. Unlike torchvision's `.DEFAULT`, a timm model name
# maps to several checkpoints that differ in pretraining data, so the tag is
# part of the experiment definition and belongs in version control.
VIT_S_WEIGHT_TAG = "deit_small_patch16_224.fb_in1k"
CONVNEXTV2_T_WEIGHT_TAG = "convnextv2_tiny.fcmae_ft_in1k"

# Dropout before every classification head, so the five heads differ only in
# their input width.
HEAD_DROPOUT = 0.3


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

    Kept at 121 layers on purpose: this is the reference baseline because it is
    the CheXNet backbone, and a deeper variant chosen to close the parameter
    gap with the other four would no longer be that baseline. Note its ImageNet
    weights come from torchvision's original recipe (74.4% top-1) rather than
    the modern recipes behind SwinV2-T and MaxViT-T, so the comparison is
    between architectures *as they are normally obtained*, not between
    architectures trained identically on ImageNet.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES, pretrained: bool = True):
        super().__init__()

        # Load pretrained DenseNet-121 backbone
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = models.densenet121(weights=weights)

        # Replace the original classifier head
        in_features = self.backbone.classifier.in_features  # 1024
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=HEAD_DROPOUT),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through DenseNet backbone
        return self.backbone(x)


# =============================================================================
# 2. ViT-S/16 (pure Vision Transformer)
# =============================================================================
class ViTSmallClassifier(nn.Module):
    """
    ViT-S/16 with a custom classification head.

    Architecture:
      - ViT-S/16 backbone, 12 blocks of global self-attention, width 384
      - [CLS] token → Dropout → Linear(384, NUM_CLASSES)

    **Why DeiT weights on a ViT.** `deit_small_patch16_224` *is* ViT-S/16 —
    same blocks, same patch size, same width — differing only in the recipe
    that produced the ImageNet weights. Two reasons it is the right checkpoint
    here over `vit_small_patch16_224.augreg_in1k`:

      - DeiT is the canonical answer to "train ViT-S on ImageNet-1k alone",
        which is the pretraining budget every other model in this roster gets.
      - The augreg weights are ports of the original JAX checkpoints and expect
        mean/std = 0.5, not ImageNet statistics. Using them would force
        per-model normalization and a separate DataLoader per architecture;
        with DeiT the whole roster shares one transform.

    Expect this model to be the weakest of the five under ImageNet-1k-only
    pretraining. That is a result, not a misconfiguration: pure ViTs are the
    architecture most dependent on pretraining scale.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES, pretrained: bool = True):
        super().__init__()

        # Load pretrained ViT-S/16 backbone
        self.backbone = timm.create_model(
            VIT_S_WEIGHT_TAG, pretrained=pretrained, num_classes=num_classes
        )

        # Replace classification head
        in_features = self.backbone.head.in_features  # 384
        self.backbone.head = nn.Sequential(
            nn.Dropout(p=HEAD_DROPOUT),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through ViT backbone
        return self.backbone(x)


# =============================================================================
# 3. ConvNeXtV2-T (modern CNN)
# =============================================================================
class ConvNeXtV2TinyClassifier(nn.Module):
    """
    ConvNeXtV2-Tiny with a custom classification head.

    Architecture:
      - ConvNeXtV2-T backbone (4 stages, global response normalization)
      - Global average pooling → LayerNorm (both inside timm's head)
      - Dropout → Linear(768, NUM_CLASSES)

    The `fcmae_ft_in1k` tag is the FCMAE-pretrained model fine-tuned on
    ImageNet-1k. Deliberately not `fcmae_ft_in22k_in1k`: the 22k variant would
    give this model training data no other backbone in the roster had.

    Only the final `head.fc` is replaced — the pooling and LayerNorm above it
    are pretrained and worth keeping. They stay inside `backbone.head`, so
    train.py's head/backbone split still counts them as head parameters.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES, pretrained: bool = True):
        super().__init__()

        # Load pretrained ConvNeXtV2-T backbone
        self.backbone = timm.create_model(
            CONVNEXTV2_T_WEIGHT_TAG, pretrained=pretrained, num_classes=num_classes
        )

        # Replace the final projection, keeping timm's pooling and norm
        in_features = self.backbone.head.fc.in_features  # 768
        self.backbone.head.fc = nn.Sequential(
            nn.Dropout(p=HEAD_DROPOUT),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through ConvNeXtV2 backbone
        return self.backbone(x)


# =============================================================================
# 4. SwinV2-T (modern Vision Transformer)
# =============================================================================
class SwinV2TinyClassifier(nn.Module):
    """
    Swin Transformer V2-Tiny with a custom classification head.

    Architecture:
      - SwinV2-T backbone (hierarchical, shifted-window attention)
      - Dropout → Linear(768, NUM_CLASSES)

    **Resolution caveat.** torchvision's SwinV2 weights — every size, not just
    Tiny — were trained at 256x256, and this pipeline runs at 224 because
    MaxViT-T cannot run at anything else (see MaxViTTinyClassifier). At 224 the
    final stage is a 7x7 map against a window size of 8, so its windowed
    attention covers the whole map rather than a shifted window. SwinV2's
    log-spaced continuous position bias is designed for exactly this kind of
    cross-resolution transfer — it is the headline change from SwinV1 — so the
    effect is a soft degradation rather than a break, but it is a real
    limitation of this comparison and is reported as one.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES, pretrained: bool = True):
        super().__init__()

        # Load pretrained SwinV2-T backbone
        weights = models.Swin_V2_T_Weights.DEFAULT if pretrained else None
        self.backbone = models.swin_v2_t(weights=weights)

        # Replace classification head
        in_features = self.backbone.head.in_features  # 768
        self.backbone.head = nn.Sequential(
            nn.Dropout(p=HEAD_DROPOUT),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through Swin ViT backbone
        return self.backbone(x)


# =============================================================================
# 5. MaxViT-T (hybrid CNN / Transformer)
# =============================================================================
class MaxViTTinyClassifier(nn.Module):
    """
    MaxViT-Tiny with a custom classification head.

    Architecture:
      - MaxViT-T backbone: every stage runs MBConv, then block attention
        (local), then grid attention (sparse global)
      - Adaptive average pooling → LayerNorm → Linear → Tanh (all pretrained)
      - Dropout → Linear(512, NUM_CLASSES)

    The hybrid slot in the roster: unlike a conv-stem-then-transformer design,
    convolution and attention are interleaved at every stage, so there is no
    point in the network where it stops being a CNN and starts being a
    transformer.

    **This model fixes the pipeline at 224x224.** torchvision builds MaxViT's
    partition sizes from a declared input size and reshapes against them, so a
    different resolution raises inside the attention partitioning rather than
    adapting. config.IMAGE_SIZE is 224 for this reason.

    Only the final projection is replaced; the pretrained pooling/norm/Tanh
    stack above it is kept. torchvision's original final layer has no bias
    (it follows a Tanh); the replacement uses the default bias, which is
    standard for a freshly initialized head.

    **Consequence for head_only tuning.** MaxViT is the only model here whose
    architectural head is an MLP rather than a single projection, so keeping it
    means head_only trains 271k parameters against 5-14k for the other four.
    That is faithful to the architecture, and evaluate.py records
    `trainable_params` per run so the difference is visible in the results
    rather than hidden — but it does mean the head_only arm is not comparing
    equal amounts of trainable capacity. Replacing the whole `classifier`
    instead of its last element would even that up, at the cost of discarding
    pretrained weights that are part of MaxViT's design.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES, pretrained: bool = True):
        super().__init__()

        # Load pretrained MaxViT-T backbone
        weights = models.MaxVit_T_Weights.DEFAULT if pretrained else None
        self.backbone = models.maxvit_t(weights=weights)

        # Replace the final projection, keeping torchvision's pooling/norm/Tanh
        in_features = self.backbone.classifier[-1].in_features  # 512
        self.backbone.classifier[-1] = nn.Sequential(
            nn.Dropout(p=HEAD_DROPOUT),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through MaxViT backbone
        return self.backbone(x)


# =============================================================================
# Model factory
# =============================================================================
# Keyed by the `--model` value. Kept as a table rather than a branch chain so
# that it and config.SUPPORTED_MODELS can be checked against each other; a
# model listed in one but not the other is a test failure, not a crash halfway
# into a sweep.
MODEL_BUILDERS: dict[str, type[nn.Module]] = {
    "densenet121": DenseNet121Classifier,
    "vit_s_16": ViTSmallClassifier,
    "convnextv2_t": ConvNeXtV2TinyClassifier,
    "swin_v2_t": SwinV2TinyClassifier,
    "maxvit_t": MaxViTTinyClassifier,
}


def build_model(model_name: str, pretrained: bool = True) -> nn.Module:
    """
    Instantiate a model by name.

    Args:
        model_name: One of config.SUPPORTED_MODELS.
        pretrained: Whether to load ImageNet-pretrained weights. False keeps
            construction offline, which is what the test suite relies on.

    Returns:
        nn.Module ready for training.
    """
    if model_name not in MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model '{model_name}'. Supported: {config.SUPPORTED_MODELS}"
        )
    return MODEL_BUILDERS[model_name](pretrained=pretrained)
