"""
One backbone per architecture family, each with a 14-class multi-label head.

Two invariants hold across the roster and are load-bearing for the comparison:
ImageNet-1k pretraining only, and identical 224x224 ImageNet-normalized input
so dataset.py can share one transform. `--weight-tag` is the sanctioned way
past the first; `build_model` rejects any tag that would break the second.
`densenet121_xrv` sits outside both on purpose. See docs/design-notes.md.
"""

import sys

import timm
import torch
import torch.nn as nn
from torchvision import models

import config


def _import_torchxrayvision():
    """
    Import torchxrayvision without letting it shadow this project's modules.

    Its vendored models each `sys.path.insert(0, <own folder>)` at import time,
    and one of those folders holds a package named `config` that would then win
    over ours. Harmless in this process, fatal in spawned DataLoader workers,
    which get the path but no preloaded modules. Restoring sys.path is enough.
    """
    saved_path = list(sys.path)
    try:
        import torchxrayvision
    except ImportError as error:
        raise ImportError(
            "densenet121_xrv needs torchxrayvision. Install it with "
            "`pip install torchxrayvision`, or pick another model."
        ) from error
    finally:
        sys.path[:] = saved_path

    return torchxrayvision


# Pinned rather than resolved by `.DEFAULT`: one timm name maps to several
# checkpoints differing in pretraining data, so the tag defines the experiment.
VIT_S_WEIGHT_TAG = "deit3_small_patch16_224.fb_in1k"
CONVNEXTV2_T_WEIGHT_TAG = "convnextv2_tiny.fcmae_ft_in1k"

# Only these accept `--weight-tag`; the torchvision models have no equivalent,
# so overriding them is an error rather than a silent no-op.
TIMM_BACKED_MODELS = ("vit_s_16", "convnextv2_t")

# CheXpert is the largest of the leakage-free options and the closest match to
# this pipeline's frontal-radiograph setting.
XRV_DEFAULT_WEIGHT_TAG = "densenet121-res224-chex"

# XRV checkpoints whose training data overlaps ChestX-ray14, keyed to why.
# Using one silently turns the held-out test split into training data.
XRV_LEAKY_WEIGHTS = {
    "densenet121-res224-all": (
        "trained on nih-pc-chex-mimic_ch-google-openi-rsna, which includes "
        "ChestX-ray14 itself"
    ),
    "densenet121-res224-nih": "trained on ChestX-ray14, this project's own dataset",
    "densenet121-res224-rsna": (
        "trained on the RSNA Pneumonia Detection Challenge, whose images are "
        "drawn from ChestX-ray8/14"
    ),
}

# torchvision builds MaxViT's attention partitions from a declared input size,
# so this one model cannot follow config.IMAGE_SIZE.
MAXVIT_FIXED_IMAGE_SIZE = 224

# Every head is dropout + linear, so they differ only in input width.
HEAD_DROPOUT = 0.3


class DenseNet121Classifier(nn.Module):
    """
    DenseNet-121 (7.0M), the CNN baseline.

    Kept at 121 layers because it is the CheXNet backbone; deepening it to close
    the parameter gap with the rest of the roster would throw that away.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES, pretrained: bool = True):
        super().__init__()

        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = models.densenet121(weights=weights)

        in_features = self.backbone.classifier.in_features  # 1024
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=HEAD_DROPOUT),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ViTSmallClassifier(nn.Module):
    """
    ViT-S/16 (21.7M), the pure Vision Transformer.

    DeiT III weights because `deit3_small_patch16_224` *is* ViT-S/16, differing
    only in the recipe: it is the strongest ImageNet-1k-only checkpoint, it has
    a matched IN22k sibling that makes the pretraining ablation a one-tag swap,
    and unlike the augreg ports it expects ImageNet statistics. Expect this to
    be the weakest roster model — pure ViTs depend most on pretraining scale.
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        pretrained: bool = True,
        weight_tag: str | None = None,
    ):
        super().__init__()

        # Resolved here so the results file names the actual checkpoint.
        self.weight_tag = weight_tag or VIT_S_WEIGHT_TAG

        # img_size lets timm interpolate the position embeddings off 224.
        self.backbone = timm.create_model(
            self.weight_tag,
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=config.IMAGE_SIZE,
        )

        in_features = self.backbone.head.in_features  # 384
        self.backbone.head = nn.Sequential(
            nn.Dropout(p=HEAD_DROPOUT),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ConvNeXtV2TinyClassifier(nn.Module):
    """
    ConvNeXtV2-T (27.9M), the modern CNN.

    `fcmae_ft_in1k`, not `fcmae_ft_in22k_in1k`, which would give this model
    training data no other backbone had. Only the final `head.fc` is replaced;
    timm's pretrained pooling and LayerNorm above it are kept.
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        pretrained: bool = True,
        weight_tag: str | None = None,
    ):
        super().__init__()

        self.weight_tag = weight_tag or CONVNEXTV2_T_WEIGHT_TAG
        self.backbone = timm.create_model(
            self.weight_tag,
            pretrained=pretrained,
            num_classes=num_classes,
        )

        in_features = self.backbone.head.fc.in_features  # 768
        self.backbone.head.fc = nn.Sequential(
            nn.Dropout(p=HEAD_DROPOUT),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class SwinTinyClassifier(nn.Module):
    """
    Swin-T (27.5M), the hierarchical windowed Vision Transformer.

    Holds this slot because it is pretrained at 224, the roster's resolution.
    SwinV2-T was the original choice and failed — see SwinV2TinyClassifier.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES, pretrained: bool = True):
        super().__init__()

        weights = models.Swin_T_Weights.DEFAULT if pretrained else None
        self.backbone = models.swin_t(weights=weights)

        in_features = self.backbone.head.in_features  # 768
        self.backbone.head = nn.Sequential(
            nn.Dropout(p=HEAD_DROPOUT),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class SwinV2TinyClassifier(nn.Module):
    """
    SwinV2-T (27.6M), off-roster, kept so its failure stays reproducible.

    torchvision's weights are 256-native, and at 224 the final stage is a 7x7
    map against a window size of 8 — a transfer SwinV2's continuous position
    bias is meant to absorb and measurably does not (train AUROC 0.656 against
    0.79-0.81 elsewhere). Run at `--image-size 256` for the resolution study.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES, pretrained: bool = True):
        super().__init__()

        weights = models.Swin_V2_T_Weights.DEFAULT if pretrained else None
        self.backbone = models.swin_v2_t(weights=weights)

        in_features = self.backbone.head.in_features  # 768
        self.backbone.head = nn.Sequential(
            nn.Dropout(p=HEAD_DROPOUT),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class MaxViTTinyClassifier(nn.Module):
    """
    MaxViT-T (30.4M), the hybrid: MBConv, block attention and grid attention
    interleaved at every stage rather than a conv stem under a transformer.

    Fixes the pipeline at 224x224 — torchvision builds the attention partitions
    from a declared input size and raises rather than adapting.

    Only the final projection is replaced, keeping the pretrained
    pooling/norm/Tanh stack. That makes this the one model whose architectural
    head is an MLP, so `head_only` trains 271k parameters here against 5-14k
    elsewhere; evaluate.py records `trainable_params` so it stays visible.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES, pretrained: bool = True):
        super().__init__()

        # Checked here: torchvision otherwise fails deep inside the attention
        # partitioning with an uninterpretable shape error.
        if config.IMAGE_SIZE != MAXVIT_FIXED_IMAGE_SIZE:
            raise ValueError(
                f"MaxViT-T only runs at {MAXVIT_FIXED_IMAGE_SIZE}x{MAXVIT_FIXED_IMAGE_SIZE}, "
                f"but config.IMAGE_SIZE is {config.IMAGE_SIZE}. Pick another model for a "
                f"resolution study; densenet121, convnextv2_t and swin_t all resize."
            )

        weights = models.MaxVit_T_Weights.DEFAULT if pretrained else None
        self.backbone = models.maxvit_t(weights=weights)

        in_features = self.backbone.classifier[-1].in_features  # 512
        self.backbone.classifier[-1] = nn.Sequential(
            nn.Dropout(p=HEAD_DROPOUT),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class DenseNet121XRVClassifier(nn.Module):
    """
    DenseNet-121 initialized from TorchXRayVision chest-radiograph weights.

    Off-roster: it breaks the ImageNet-1k-only invariant deliberately, to ask
    how much of the roster's spread is pretraining domain rather than
    architecture. Its control is the densenet121 run, which shares the module
    layout exactly — so the Grad-CAM target, randomization stages and
    head/backbone split all reuse the densenet121 entries and only the
    pretraining corpus moves.

    The checkpoint takes 1-channel input on XRV's own scale; `_to_xrv_scale`
    converts inside `forward` rather than forking the shared transform.
    `XRV_LEAKY_WEIGHTS` blocks any corpus containing ChestX-ray14.
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        pretrained: bool = True,
        weight_tag: str | None = None,
    ):
        super().__init__()

        # Imported here so the rest of the roster builds without XRV installed.
        xrv = _import_torchxrayvision()

        self.weight_tag = weight_tag or XRV_DEFAULT_WEIGHT_TAG
        _check_xrv_weight_tag(self.weight_tag)

        # `weights=` fixes the output width at the checkpoint's own pathology
        # count, so num_classes cannot be passed; the head is replaced below.
        self.backbone = xrv.models.DenseNet(
            weights=self.weight_tag if pretrained else None
        )

        # Set, XRV's forward applies a sigmoid and a per-class rescale indexed
        # by *its* pathology list. This pipeline needs raw logits.
        self.backbone.op_threshs = None

        # XRV otherwise resizes non-native input back to 224 inside `forward`,
        # quietly overriding `--image-size`. DenseNet is fully convolutional.
        if hasattr(self.backbone, "input_resolution"):
            del self.backbone.input_resolution

        in_features = self.backbone.classifier.in_features  # 1024
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=HEAD_DROPOUT),
            nn.Linear(in_features, num_classes),
        )

        # Buffers so `.to(device)` carries them; a constant would force a
        # host/device sync mid-batch.
        self.register_buffer(
            "_imagenet_mean", torch.tensor(config.IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_imagenet_std", torch.tensor(config.IMAGENET_STD).view(1, 3, 1, 1)
        )

    def _to_xrv_scale(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map an ImageNet-normalized 3-channel batch onto XRV's input scale.

        Exact, not approximate: undoing dataset.py's normalization and applying
        `xrv.utils.normalize`'s is a composition of two affine maps, so nothing
        is lost and gradients pass through. Channels are averaged rather than
        indexed, which is identical on grayscale input and degrades gracefully.
        """
        gray = (x * self._imagenet_std + self._imagenet_mean).mean(dim=1, keepdim=True)
        return (2.0 * gray - 1.0) * 1024.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self._to_xrv_scale(x))


# --- Model factory ------------------------------------------------------------
# A table rather than a branch chain, so it can be checked against
# config.SUPPORTED_MODELS by a test instead of crashing mid-sweep.
MODEL_BUILDERS: dict[str, type[nn.Module]] = {
    "densenet121": DenseNet121Classifier,
    "vit_s_16": ViTSmallClassifier,
    "convnextv2_t": ConvNeXtV2TinyClassifier,
    "swin_t": SwinTinyClassifier,
    "swin_v2_t": SwinV2TinyClassifier,
    "maxvit_t": MaxViTTinyClassifier,
    "densenet121_xrv": DenseNet121XRVClassifier,
}


def _check_xrv_weight_tag(weight_tag: str) -> None:
    """
    Reject an XRV checkpoint that has already seen the NIH test split.

    Such a run completes normally and reports a test AUROC several points high,
    which reads as "medical pretraining wins" and is memorization. Nothing
    downstream can detect it, so it is blocked here.
    """
    xrv = _import_torchxrayvision()

    if weight_tag in XRV_LEAKY_WEIGHTS:
        clean = sorted(set(_xrv_densenet_tags()) - set(XRV_LEAKY_WEIGHTS))
        raise ValueError(
            f"Weight tag '{weight_tag}' leaks the evaluation set: "
            f"{XRV_LEAKY_WEIGHTS[weight_tag]}. Test scores from it would measure "
            f"memorization of images this pipeline reports as held out. "
            f"Leakage-free alternatives: {clean}."
        )

    if weight_tag not in xrv.models.model_urls:
        raise ValueError(
            f"Unknown TorchXRayVision weight tag '{weight_tag}'. "
            f"Available DenseNet-121 tags: {sorted(_xrv_densenet_tags())}."
        )


def _xrv_densenet_tags() -> list[str]:
    """The DenseNet-121 entries of XRV's weight registry, long-form names only."""
    xrv = _import_torchxrayvision()

    return [name for name in xrv.models.model_urls if name.startswith("densenet121-")]


def _check_weight_tag_normalization(weight_tag: str) -> None:
    """
    Reject a weight tag whose checkpoint expects different input statistics.

    One transform is shared across every loader, so such a run would train,
    converge, and report a plausible number that is simply wrong.
    `vit_small_patch16_224.augreg_*` is the live example: it wants mean/std 0.5.
    """
    # Returns None rather than raising on an unknown tag, which would otherwise
    # surface as an AttributeError several frames away.
    cfg = timm.get_pretrained_cfg(weight_tag, allow_unregistered=True)
    if cfg is None:
        raise ValueError(
            f"Unknown timm weight tag '{weight_tag}'. List the candidates with "
            f"timm.list_models('<name>*', pretrained=True)."
        )

    expected = (tuple(config.IMAGENET_MEAN), tuple(config.IMAGENET_STD))
    actual = (tuple(cfg.mean), tuple(cfg.std))
    if actual != expected:
        raise ValueError(
            f"Weight tag '{weight_tag}' expects mean/std {actual}, but the shared "
            f"transform normalizes with {expected}. Using it would train under "
            f"different statistics than the checkpoint was pretrained with. "
            f"Pick a tag with ImageNet statistics, or add per-model "
            f"normalization to dataset.py first."
        )


def build_model(
    model_name: str, pretrained: bool = True, weight_tag: str | None = None
) -> nn.Module:
    """
    Instantiate one of config.SUPPORTED_MODELS.

    `pretrained=False` keeps construction offline, which the test suite relies
    on. `weight_tag` overrides the pinned checkpoint for the pretraining-data
    ablation: a timm tag for TIMM_BACKED_MODELS, an XRV weight name for
    densenet121_xrv, and an error for the torchvision-backed models.
    """
    if model_name not in MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model '{model_name}'. Supported: {config.SUPPORTED_MODELS}"
        )

    if weight_tag is None:
        return MODEL_BUILDERS[model_name](pretrained=pretrained)

    # Validated inside the class: the check needs torchxrayvision, which would
    # otherwise become a hard dependency of every model.
    if model_name == "densenet121_xrv":
        return MODEL_BUILDERS[model_name](pretrained=pretrained, weight_tag=weight_tag)

    if model_name not in TIMM_BACKED_MODELS:
        raise ValueError(
            f"--weight-tag is only supported for "
            f"{[*TIMM_BACKED_MODELS, 'densenet121_xrv']}; "
            f"'{model_name}' resolves its weights through torchvision and has no "
            f"tag to override. Drop the flag, or select a model that takes one."
        )

    _check_weight_tag_normalization(weight_tag)
    return MODEL_BUILDERS[model_name](pretrained=pretrained, weight_tag=weight_tag)
