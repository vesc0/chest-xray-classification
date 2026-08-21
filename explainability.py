"""
Grad-CAM for every architecture, plus Attention Rollout for ViT-S/16.

Grad-CAM runs everywhere because a "which model localizes better" claim is
meaningless if CNNs and transformers are explained by different methods. Each
architecture pairs a target layer with a reshape transform in GRADCAM_TARGETS,
following one rule: explain each model at the tensor its classifier pools.

Rollout is ViT-only by necessity — Swin's shifted windows and MaxViT's MBConv
blocks leave no single token set to multiply across layers — and is
class-agnostic, showing where the model looks rather than what supported a
pathology. Two asymmetries to read the localization numbers with (ViT's 14x14
grid against the others' 7x7, and LayerNorm-pooled features) are in
docs/design-notes.md.
"""

from functools import partial

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

import config


# --- 1. Grad-CAM (all architectures) ------------------------------------------
def _identity_transform(tensor: torch.Tensor) -> torch.Tensor:
    """CNN feature maps are already (B, C, H, W)."""
    return tensor


def _nhwc_to_nchw(tensor: torch.Tensor) -> torch.Tensor:
    """Swin emits (B, H, W, C); move the channel axis where the CAM expects it."""
    return tensor.permute(0, 3, 1, 2).contiguous()


def _tokens_to_grid(tensor: torch.Tensor, num_prefix_tokens: int = 1) -> torch.Tensor:
    """
    Reshape ViT tokens (B, P + N, C) into a (B, C, sqrt(N), sqrt(N)) map.

    num_prefix_tokens is bound from the backbone rather than assumed: [CLS] for
    ViT-S/16, but distillation or register tokens push it higher on other
    checkpoints, and dropping the wrong number shifts every patch by one.
    """
    tokens = tensor[:, num_prefix_tokens:, :]  # prefix tokens carry no location
    batch, num_tokens, channels = tokens.shape

    grid_size = int(round(num_tokens ** 0.5))
    if grid_size * grid_size != num_tokens:
        raise ValueError(
            f"Expected a square patch grid, got {num_tokens} tokens."
        )

    return tokens.reshape(batch, grid_size, grid_size, channels).permute(0, 3, 1, 2).contiguous()


def _normalize_per_sample(maps: torch.Tensor) -> torch.Tensor:
    """
    Min-max each (H, W) map in a batch to [0, 1], independently.

    A display and thresholding normalization only. Min-subtraction removes the
    uniform component that localization's random baseline is defined against,
    so the energy fraction reads `last_raw_maps` instead.
    """
    flat = maps.flatten(1)
    minimum = flat.min(dim=1).values.view(-1, 1, 1)
    maximum = flat.max(dim=1).values.view(-1, 1, 1)
    return (maps - minimum) / (maximum - minimum + 1e-8)


def _fixed(transform):
    """Wrap a model-independent reshape as the (model -> transform) factory
    that GRADCAM_TARGETS entries have to supply for ViT's sake."""
    return lambda model: transform


# Target layer + layout transform per architecture. Getting this wrong is quiet
# rather than loud: a transformer layout fed into the CNN maths produces a
# plausible-looking but meaningless heatmap.
GRADCAM_TARGETS: dict = {
    # The one deliberate exception to the pooling rule: torchvision's DenseNet
    # runs `F.relu(features, inplace=True)` before the pool, so hooking
    # `features` captures a tensor the ReLU then overwrites, pairing post-ReLU
    # activations with pre-ReLU gradients. denseblock4 is the last safe tensor.
    "densenet121": (
        lambda model: model.backbone.features.denseblock4,
        _fixed(_identity_transform),
    ),
    # Same graph as densenet121, so the same trap and the same answer.
    "densenet121_xrv": (
        lambda model: model.backbone.features.denseblock4,
        _fixed(_identity_transform),
    ),
    "convnextv2_t": (
        lambda model: model.backbone.stages[-1],
        _fixed(_identity_transform),
    ),
    # The final LayerNorm, whose output is what torchvision pools. The pre-norm
    # alternative gives a substantially different map; see the design notes.
    "swin_t": (
        lambda model: model.backbone.norm,
        _fixed(_nhwc_to_nchw),
    ),
    "swin_v2_t": (
        lambda model: model.backbone.norm,
        _fixed(_nhwc_to_nchw),
    ),
    # Ends in grid attention but still emits NCHW, so no reshape is needed.
    "maxvit_t": (
        lambda model: model.backbone.blocks[-1],
        _fixed(_identity_transform),
    ),
    # NOT backbone.norm: the head reads only [CLS], so gradients w.r.t. every
    # patch token there are exactly zero and the CAM comes out blank.
    "vit_s_16": (
        lambda model: model.backbone.blocks[-1].norm1,
        lambda model: partial(
            _tokens_to_grid, num_prefix_tokens=model.backbone.num_prefix_tokens
        ),
    ),
}


class GradCAM:
    """
    Class-discriminative heatmaps from the layer registered in GRADCAM_TARGETS,
    reshaped to NCHW by that entry's transform.
    """

    class_agnostic = False

    # Both set by the most recent generate_batch. last_raw_maps is BEFORE
    # per-sample normalization, which is what localization's energy needs.
    last_logits: torch.Tensor | None = None
    last_raw_maps: np.ndarray | None = None

    def __init__(self, model: nn.Module, model_name: str = "densenet121"):
        self.model = model
        self.model.eval()

        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        # The hook is live for the object's lifetime but only captures inside
        # generate_batch. Ungated, an instance built alongside another explainer
        # would fire on its forward passes and call requires_grad_() on the
        # captured tensor, which raises under torch.inference_mode().
        self._capturing = False

        if model_name not in GRADCAM_TARGETS:
            raise ValueError(
                f"No Grad-CAM target layer registered for '{model_name}'. "
                f"Known: {sorted(GRADCAM_TARGETS)}"
            )

        resolve_layer, build_transform = GRADCAM_TARGETS[model_name]
        target_layer = resolve_layer(model)
        self.reshape_transform = build_transform(model)

        # Untransformed target-layer output, held between forward and grad.
        self._captured: torch.Tensor | None = None

        # Kept so the hook can be detached; one left attached makes every
        # forward pass store activations.
        self.handles = [target_layer.register_forward_hook(self._save_activation)]

    def remove_hooks(self) -> None:
        """Detach the forward hook. Safe to call more than once."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.activations = None
        self.gradients = None
        self._captured = None
        self._capturing = False

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *exc_info) -> None:
        self.remove_hooks()

    def _save_activation(self, module, input, output):
        """
        Capture the target layer's output and make it the root of the graph.

        Rooting here rather than at the network input matters twice: with a
        frozen backbone nothing upstream requires grad, so the layer would never
        enter the graph at all; and retaining the whole network's activations
        falls off an MPS memory cliff (0.89s/batch at 32, 277s at 64).
        """
        if not self._capturing:
            return
        if not output.requires_grad:
            output.requires_grad_(True)
        self._captured = output

    def generate_batch(
        self,
        images: torch.Tensor,
        class_indices: torch.Tensor | list[int],
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmaps for a batch, one target class per image.

        Samples in a batch are independent, so backpropagating the *sum* of the
        selected logits gives the same per-sample gradients as looping, cheaply.

        Returns (B, H, W), each map normalized to [0, 1].
        """
        if not self.handles:
            raise RuntimeError("GradCAM hooks have been removed; build a new instance.")

        self._captured = None

        # Gradients are needed even under an outer torch.no_grad().
        self._capturing = True
        try:
            with torch.enable_grad():
                logits = self.model(images.detach())

                if self._captured is None:
                    raise RuntimeError(
                        "Target layer produced no activation — the registered layer "
                        "was not executed during the forward pass."
                    )

                if not isinstance(class_indices, torch.Tensor):
                    class_indices = torch.tensor(class_indices, device=logits.device)
                class_indices = class_indices.to(logits.device).long()

                rows = torch.arange(logits.size(0), device=logits.device)
                selected = logits[rows, class_indices].sum()

                # autograd.grad walks back only this far and leaves parameter
                # .grad untouched, so this is safe mid-training.
                (raw_gradients,) = torch.autograd.grad(selected, self._captured)
        finally:
            self._capturing = False

        self.activations = self.reshape_transform(self._captured.detach())
        self.gradients = self.reshape_transform(raw_gradients.detach())
        # Part of the explainer interface: lets callers reuse this forward pass
        # rather than running the model again for the probabilities.
        self.last_logits = logits.detach()
        self._captured = None

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=images.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)

        # Grad-CAM's ReLU already floors these at 0, so min-max is a pure
        # rescale here — but the energy metric reads the same attribute for
        # every method, and for rollout the difference is real.
        self.last_raw_maps = cam.detach().cpu().numpy()

        cam = _normalize_per_sample(cam)

        return cam.detach().cpu().numpy()

    def generate(self, image: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """One (H, W) heatmap in [0, 1] for a single (1, 3, H, W) image."""
        return self.generate_batch(image, [int(class_idx or 0)])[0]


# --- 2. Attention Rollout for ViT ---------------------------------------------
class AttentionRollout:
    """
    Attention Rollout for ViT-S/16: attention multiplied across all layers,
    with residuals, into one [CLS]-to-patch heatmap.

    Capture has two halves because timm dispatches to SDPA whenever
    `fused_attn` is set — and SDPA never materializes the attention matrix, so
    a naive port returns blank maps *without raising*. Every block is switched
    to the unfused path for the forward pass, and each block's `attn_drop`
    (the identity in eval mode) is hooked. Both changes are reverted after.

    **Expect border artifacts, and do not read them as a bug.** ViTs repurpose
    low-information background patches as scratch space (Darcet et al., 2024),
    and rolling attention across 12 layers compounds exactly those tokens, so
    the peak often lands on an edge. This drives the pointing game to ~0
    against a ~0.07 random baseline: the method meeting the architecture.
    """

    class_agnostic = True

    # As on GradCAM. last_raw_maps matters more here: rolled-up attention
    # carries a large uniform floor that min-subtraction would remove, and that
    # floor is exactly what localization's random baseline is defined against.
    last_logits: torch.Tensor | None = None
    last_raw_maps: np.ndarray | None = None

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

        self.attention_maps: list[torch.Tensor] = []
        self._handles: list = []
        self._saved_fused_attn: list[bool] = []

    @property
    def _blocks(self):
        """The transformer blocks whose attention gets rolled up."""
        return self.model.backbone.blocks

    def _capture_attention(self, module, inputs, output):
        """Record one layer's head-averaged attention matrix."""
        # Averaging at capture time keeps 6x less attention resident. The
        # assert checks the head axis is there before collapsing it.
        torch._assert(
            output.dim() == 4,
            f"Expected (B, heads, S, S) attention, got {tuple(output.shape)}",
        )
        self.attention_maps.append(output.detach().mean(dim=1))

    def _start_capture(self):
        """Force the unfused attention path and hook each block's attn_drop."""
        self._saved_fused_attn = []
        for block in self._blocks:
            # Remembered rather than assumed True, so this restores the
            # model's own configuration whatever timm's default becomes.
            self._saved_fused_attn.append(block.attn.fused_attn)
            block.attn.fused_attn = False
            self._handles.append(
                block.attn.attn_drop.register_forward_hook(self._capture_attention)
            )

    def _stop_capture(self):
        """Detach the hooks and put fused attention back the way it was."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

        for block, fused in zip(self._blocks, self._saved_fused_attn):
            block.attn.fused_attn = fused
        self._saved_fused_attn = []

    @torch.no_grad()
    def generate_batch(
        self,
        images: torch.Tensor,
        class_indices: torch.Tensor | list[int] | None = None,
    ) -> np.ndarray:
        """
        Generate attention rollout heatmaps for a batch.

        class_indices is accepted for a uniform explainer interface but
        ignored; rollout is class-agnostic. Returns (B, H, W) maps in [0, 1].
        """
        self.attention_maps.clear()

        self._start_capture()
        try:
            # Same interface contract as GradCAM: expose this forward's logits.
            self.last_logits = self.model(images).detach()
        finally:
            self._stop_capture()

        batch, _, height, width = images.shape

        # Raise rather than fall back to a constant map, which would sail
        # through localization scoring as a real result.
        if not self.attention_maps:
            raise RuntimeError(
                "Attention Rollout captured no attention matrices. The forward "
                "pass did not take the unfused attention path, so there was "
                "nothing to hook."
            )

        result = None
        for attn_avg in self.attention_maps:
            # A lower-rank tensor would silently broadcast against the identity
            # below and yield a bogus map rather than failing.
            if attn_avg.dim() != 3 or attn_avg.size(-1) != attn_avg.size(-2):
                raise RuntimeError(
                    "Expected batched square (B, seq_len, seq_len) attention, got "
                    f"{tuple(attn_avg.shape)}."
                )

            # Residual connection, rows re-normalized, then rolled up.
            identity = torch.eye(attn_avg.size(-1), device=attn_avg.device).unsqueeze(0)
            attn_avg = 0.5 * attn_avg + 0.5 * identity
            attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
            result = attn_avg if result is None else attn_avg @ result

        # [CLS] attention to all patches, dropping the same prefix tokens
        # _tokens_to_grid drops.
        num_prefix_tokens = self.model.backbone.num_prefix_tokens
        cls_attention = result[:, 0, num_prefix_tokens:]  # (B, num_patches)

        num_patches = cls_attention.size(1)
        grid_size = int(round(num_patches ** 0.5))
        if grid_size * grid_size != num_patches:
            raise ValueError(f"Expected a square patch grid, got {num_patches} tokens.")

        grid = cls_attention.reshape(batch, 1, grid_size, grid_size)
        heatmaps = F.interpolate(
            grid, size=(height, width), mode="bilinear", align_corners=False
        ).squeeze(1)

        self.last_raw_maps = heatmaps.cpu().numpy()

        return _normalize_per_sample(heatmaps).cpu().numpy()

    def generate(self, image: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """One (H, W) heatmap in [0, 1]; class_idx is ignored."""
        return self.generate_batch(image)[0]


# --- Visualization helpers (shared with localization.py) ----------------------
def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalization into an (H, W, 3) uint8 array.

    Public so localization.py's figures use the identical inverse; keep it in
    step with get_eval_transforms()'s Normalize.
    """
    mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def overlay_heatmap(
    image_tensor: torch.Tensor,
    heatmap: np.ndarray,
    title: str,
    save_path: str,
) -> None:
    """Save a three-panel figure: original, heatmap, and the two overlaid."""
    original = denormalize(image_tensor)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original X-ray")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Heatmap")
    axes[1].axis("off")

    axes[2].imshow(original)
    axes[2].imshow(heatmap, cmap="jet", alpha=0.4)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- Explainer selection ------------------------------------------------------
def available_explainers(model_name: str) -> list[tuple[str, str]]:
    """
    Methods defined for an architecture, as (key, label) — nothing instantiated.

    Separate from build_explainer so a caller can plan a run without holding a
    live explainer, which lets localization.py keep exactly one attached.
    """
    methods: list[tuple[str, str]] = []
    if model_name in GRADCAM_TARGETS:
        methods.append(("gradcam", "Grad-CAM"))
    if model_name == "vit_s_16":
        methods.append(("rollout", "Attention Rollout"))
    return methods


def build_explainer(model: nn.Module, model_name: str, method_key: str) -> object:
    """
    Instantiate one explainer by key. Grad-CAM attaches a hook — release it with
    release_explainers, or use it as a context manager.

    Checked against what the architecture supports, not just what is
    implemented: a rollout over a CNN would build and fail several layers later.
    """
    available = [key for key, _ in available_explainers(model_name)]
    if method_key not in available:
        raise ValueError(
            f"Unknown explainer '{method_key}' for '{model_name}'. "
            f"Available: {available}"
        )
    if method_key == "gradcam":
        return GradCAM(model, model_name=model_name)
    return AttentionRollout(model)


def build_explainers(model: nn.Module, model_name: str) -> list[tuple[str, str, object]]:
    """
    Every explainer for an architecture at once, as (key, label, instance).

    Only for callers that need all methods live together, like
    generate_explanations, which would otherwise re-read the loader per method.
    Grad-CAM instances attach hooks — call release_explainers() when finished.
    """
    return [
        (key, label, build_explainer(model, model_name, key))
        for key, label in available_explainers(model_name)
    ]


def release_explainers(explainers: list[tuple[str, str, object]]) -> None:
    """Detach any hooks the explainers installed."""
    for _, _, explainer in explainers:
        if isinstance(explainer, GradCAM):
            explainer.remove_hooks()


# --- End-to-end XAI generation ------------------------------------------------
def generate_explanations(
    model: nn.Module,
    test_loader: DataLoader,
    model_name: str,
    thresholds: np.ndarray | None = None,
    num_samples: int | None = None,
) -> None:
    """
    Render every available explainer over the first `num_samples` test images,
    into xai/<model>/<method>/.

    num_samples defaults to config.XAI_NUM_SAMPLES, read here rather than bound
    as a default argument: --xai-samples rewrites it at runtime, long after
    import, so a default bound at import time would always be the 0 the CLI
    was meant to override.
    """
    if num_samples is None:
        num_samples = config.XAI_NUM_SAMPLES

    device = next(model.parameters()).device
    model.eval()

    if thresholds is None:
        thresholds = np.full(config.NUM_CLASSES, config.DEFAULT_THRESHOLD, dtype=np.float32)
    else:
        thresholds = np.asarray(thresholds, dtype=np.float32)

    xai_dir = config.XAI_DIR / model_name

    explainers = build_explainers(model, model_name)

    if not explainers:
        print(f"[xai] No XAI method defined for '{model_name}' — skipping.")
        return

    for method_key, _, _ in explainers:
        (xai_dir / method_key).mkdir(parents=True, exist_ok=True)

    method_list = ", ".join(label for _, label, _ in explainers)
    print(f"[xai] Generating {method_list} visualizations for {model_name} …")

    # A batch at a time: the target class has to be picked from the
    # probabilities first, so the cost per group is one plain forward plus one
    # per method, whatever the group size.
    count = 0
    try:
        for batch in test_loader:
            take = min(batch["image"].size(0), num_samples - count)
            if take <= 0:
                break

            images = batch["image"][:take]
            labels = batch["label"][:take]
            filenames = batch["filename"][:take]
            device_images = images.to(device)

            with torch.no_grad():
                probs = torch.sigmoid(model(device_images)).cpu().numpy()

            # One target class per image, from the calibrated thresholds.
            target_classes: list[int] = []
            selection_notes: list[str] = []
            for row in probs:
                positive_indices = np.where(row >= thresholds)[0]
                if positive_indices.size > 0:
                    target_classes.append(
                        int(positive_indices[row[positive_indices].argmax()])
                    )
                    selection_notes.append("Calibrated positive")
                else:
                    target_classes.append(int(row.argmax()))
                    selection_notes.append("No calibrated positive; showing top score")

            # Same images and target classes through every explainer.
            for method_key, method_label, explainer in explainers:
                heatmaps = explainer.generate_batch(device_images, target_classes)

                for i in range(take):
                    top_class_idx = target_classes[i]
                    gt_classes = [
                        config.CLASS_NAMES[j]
                        for j, value in enumerate(labels[i].numpy())
                        if value > 0.5
                    ]

                    note = selection_notes[i]
                    if method_key == "rollout":
                        note = f"{note} (rollout is class-agnostic)"

                    title = (
                        f"{method_label} — {model_name}\n"
                        f"{note}: {config.CLASS_NAMES[top_class_idx]} "
                        f"({probs[i, top_class_idx]:.2f}, "
                        f"thr={thresholds[top_class_idx]:.2f})"
                        f"  |  GT: {', '.join(gt_classes) if gt_classes else 'No Finding'}"
                    )
                    stem = filenames[i].replace(".png", "")
                    overlay_heatmap(
                        images[i],
                        heatmaps[i],
                        title,
                        str(xai_dir / method_key / f"sample_{count + i:03d}_{stem}.png"),
                    )

            count += take
    finally:
        # Grad-CAM leaves hooks on the model; always take them off again.
        release_explainers(explainers)

    print(
        f"[xai] Saved {count} samples x {len(explainers)} method(s) → {xai_dir}"
    )
