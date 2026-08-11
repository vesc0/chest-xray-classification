"""
Explainability module

  1. Grad-CAM — for all four architectures
     Weights the last spatial feature map by the gradient of the target class.
     Applied to every model so that explanations are comparable across
     architectures: a "which model localizes better" claim is meaningless if
     the CNNs and the transformers are explained by different methods.

     Transformers do not emit NCHW feature maps, so each architecture pairs a
     target layer with a reshape transform that normalizes its output:

       DenseNet-121     features.denseblock4          NCHW already
       EfficientNet-B4  features[-1]                  NCHW already
       SwinV2-B         norm                          NHWC -> NCHW
       ViT-B/16         encoder.layers[-1].ln_1       tokens -> 14x14 grid

  2. Attention Rollout — ViT-B/16 only, as an architecture-native second view
     Recursively multiplies attention matrices across all transformer layers
     to produce a single spatial attention map from the [CLS] token. Unlike
     Grad-CAM this is class-agnostic: it shows where the model looks, not what
     evidence supported a particular pathology.

Both methods overlay a heatmap on the original X-ray and save the result.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib

# Non-interactive backend for saving figures
matplotlib.use("Agg")

import config


# =============================================================================
# 1. Grad-CAM (all architectures)
# =============================================================================
def _identity_transform(tensor: torch.Tensor) -> torch.Tensor:
    """CNN feature maps are already (B, C, H, W)."""
    return tensor


def _nhwc_to_nchw(tensor: torch.Tensor) -> torch.Tensor:
    """Swin emits (B, H, W, C); move the channel axis where the CAM expects it."""
    return tensor.permute(0, 3, 1, 2).contiguous()


def _tokens_to_grid(tensor: torch.Tensor) -> torch.Tensor:
    """Reshape ViT tokens (B, 1 + N, C) into a (B, C, sqrt(N), sqrt(N)) map."""
    tokens = tensor[:, 1:, :]  # drop the [CLS] token, which has no location
    batch, num_tokens, channels = tokens.shape

    grid_size = int(round(num_tokens ** 0.5))
    if grid_size * grid_size != num_tokens:
        raise ValueError(
            f"Expected a square patch grid, got {num_tokens} tokens."
        )

    return tokens.reshape(batch, grid_size, grid_size, channels).permute(0, 3, 1, 2).contiguous()


# Target layer + layout transform per architecture. Getting the layer wrong is
# quiet rather than loud: a transformer layout fed straight into the CNN maths
# produces a plausible-looking but meaningless heatmap.
GRADCAM_TARGETS: dict = {
    "densenet121": (
        lambda model: model.backbone.features.denseblock4,
        _identity_transform,
    ),
    "efficientnet_b4": (
        lambda model: model.backbone.features[-1],
        _identity_transform,
    ),
    "swin_v2_b": (
        lambda model: model.backbone.norm,
        _nhwc_to_nchw,
    ),
    # NOT encoder.ln: the head reads only token 0, so gradients w.r.t. every
    # patch token at that final norm are exactly zero and the CAM comes out
    # blank. ln_1 of the last block still routes through attention to all tokens.
    "vit_b_16": (
        lambda model: model.backbone.encoder.layers[-1].ln_1,
        _tokens_to_grid,
    ),
}


class GradCAM:
    """
    Grad-CAM for any of the supported architectures.

    Hooks the architecture's last spatial layer to capture activations and
    gradients, normalizes their layout to NCHW via the registered reshape
    transform, then produces a class-discriminative heatmap.
    """

    # Heatmap depends on the target class
    class_agnostic = False

    # Logits from the most recent generate_batch forward pass
    last_logits: torch.Tensor | None = None

    def __init__(self, model: nn.Module, model_name: str = "densenet121"):
        self.model = model
        self.model.eval()

        # Store intermediate activations and gradients
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        if model_name not in GRADCAM_TARGETS:
            raise ValueError(
                f"No Grad-CAM target layer registered for '{model_name}'. "
                f"Known: {sorted(GRADCAM_TARGETS)}"
            )

        resolve_layer, self.reshape_transform = GRADCAM_TARGETS[model_name]
        target_layer = resolve_layer(model)

        # Raw (untransformed) target-layer output, kept between the forward pass
        # and the gradient call
        self._captured: torch.Tensor | None = None

        # Keep the handle so the hook can be detached again — a GradCAM left
        # attached keeps every forward pass storing activations.
        self.handles = [target_layer.register_forward_hook(self._save_activation)]

    def remove_hooks(self) -> None:
        """Detach the forward hook. Safe to call more than once."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.activations = None
        self.gradients = None
        self._captured = None

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *exc_info) -> None:
        self.remove_hooks()

    def _save_activation(self, module, input, output):
        """
        Capture the target layer's output and make it the root of the graph.

        Grad-CAM only needs the gradient of the class score with respect to
        *this* tensor, so the graph is started here rather than at the network
        input. Two reasons that matters:

          - With a frozen backbone (head_only, or the warmup phase of partial)
            nothing upstream requires grad, so without this the layer never
            enters the graph at all and no gradient is available.
          - Rooting the graph at the input instead would retain activations for
            the entire network. On MPS that falls off a memory cliff: measured
            0.89s per batch at 32 but 277s at 64, which reads as a hung run.

        requires_grad_ is only valid on a leaf; when the backbone *is* trainable
        the output already carries a grad_fn and needs no help.
        """
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
        selected logits yields the same per-sample gradients as looping — at a
        fraction of the cost.

        Args:
            images: (B, 3, H, W) tensor on the model's device.
            class_indices: length-B target class per image.

        Returns:
            heatmaps: (B, H, W) numpy array, each normalized to [0, 1].
        """
        if not self.handles:
            raise RuntimeError("GradCAM hooks have been removed; build a new instance.")

        self._captured = None

        # Gradients are needed even under an outer torch.no_grad()
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

            # Only the target layer's gradient is required. autograd.grad walks
            # back just that far and leaves parameter .grad untouched, so this
            # is safe to call on a model mid-training.
            (raw_gradients,) = torch.autograd.grad(selected, self._captured)

        self.activations = self.reshape_transform(self._captured.detach())
        self.gradients = self.reshape_transform(raw_gradients.detach())
        # Part of the explainer interface: lets callers reuse this forward pass
        # instead of running the model a second time for the probabilities.
        self.last_logits = logits.detach()
        self._captured = None

        # Channel importance via global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

        # Weighted sum of feature maps, positive contributions only
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))

        # Upsample to input resolution
        cam = F.interpolate(cam, size=images.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)

        # Normalize each sample independently
        flat = cam.flatten(1)
        minimum = flat.min(dim=1).values.view(-1, 1, 1)
        maximum = flat.max(dim=1).values.view(-1, 1, 1)
        cam = (cam - minimum) / (maximum - minimum + 1e-8)

        return cam.detach().cpu().numpy()

    def generate(self, image: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for a single image.

        Args:
            image: (1, 3, H, W) tensor on the model's device.
            class_idx: Index of the target class.

        Returns:
            heatmap: (H, W) numpy array in [0, 1].
        """
        return self.generate_batch(image, [int(class_idx or 0)])[0]


# =============================================================================
# 2. Attention Rollout for ViT
# =============================================================================
class AttentionRollout:
    """
    Attention Rollout for Vision Transformer (ViT-B/16).

    Torchvision's ViT calls self_attention with need_weights=False by default,
    so each encoder block's forward method is temporarily patched to force
    need_weights=True and capture the attention weight matrices.

    The captured attention maps are then multiplied across layers (with
    residual connections) to produce a single spatial heatmap from the
    [CLS] token to every image patch.
    """

    # One map per image, whatever the class: metrics computed per class still
    # work, but they measure where the model attends overall rather than what
    # supported a specific pathology.
    class_agnostic = True

    # Logits from the most recent generate_batch forward pass
    last_logits: torch.Tensor | None = None

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

        # Store attention maps across layers
        self.attention_maps: list[torch.Tensor] = []
        self._patched = False

    def _patch_encoder_blocks(self):
        """Replace each encoder block's forward to capture attention weights."""
        import types

        self._original_forwards: list = []

        for block in self.model.backbone.encoder.layers:
            self._original_forwards.append(block.forward)

            # Closure to capture `block` correctly
            def make_patched_forward(blk):
                def patched_forward(input: torch.Tensor):
                    # Ensure transformer input shape is correct
                    import torch as _torch
                    _torch._assert(input.dim() == 3, f"Expected 3D got {input.shape}")
                    x = blk.ln_1(input)

                    # Force need_weights=True so attn_weights are computed.
                    # average_attn_weights=False keeps the per-head matrices;
                    # the default pre-averages them and drops the head axis,
                    # which makes the rollout below average the wrong dimension.
                    x, attn_weights = blk.self_attention(
                        x, x, x, need_weights=True, average_attn_weights=False
                    )

                    # Verify the head axis survived before collapsing it. Doing
                    # the head-mean here rather than in generate_batch keeps
                    # 12x less attention in memory: a batch of 32 would
                    # otherwise hold 12 layers x (32, 12, 197, 197).
                    _torch._assert(
                        attn_weights.dim() == 4,
                        f"Expected (B, heads, S, S) attention, got {attn_weights.shape}",
                    )
                    self.attention_maps.append(attn_weights.detach().mean(dim=1))
                    x = blk.dropout(x)
                    x = x + input
                    y = blk.ln_2(x)
                    y = blk.mlp(y)

                    return x + y
                
                return patched_forward

            # Replace original forward
            block.forward = types.MethodType(
                lambda self_block, input, _fn=make_patched_forward(block): _fn(input),
                block,
            )

        self._patched = True

    # Restore original transformer behavior
    def _unpatch_encoder_blocks(self):
        for block, orig in zip(self.model.backbone.encoder.layers, self._original_forwards):
            block.forward = orig
        self._patched = False

    @torch.no_grad()
    def generate_batch(
        self,
        images: torch.Tensor,
        class_indices: torch.Tensor | list[int] | None = None,
    ) -> np.ndarray:
        """
        Generate attention rollout heatmaps for a batch.

        Args:
            images: (B, 3, H, W) tensor on the model's device.
            class_indices: Accepted for a uniform explainer interface but
                ignored — rollout is class-agnostic, so every class of a given
                image receives the same map.

        Returns:
            heatmaps: (B, H, W) numpy array, each normalized to [0, 1].
        """
        self.attention_maps.clear()

        # Temporarily patch to capture attention weights
        self._patch_encoder_blocks()
        try:
            # Same interface contract as GradCAM: expose this forward's logits
            self.last_logits = self.model(images).detach()
        finally:
            self._unpatch_encoder_blocks()

        batch, _, height, width = images.shape

        # Fallback if no attention captured
        if not self.attention_maps:
            return np.full((batch, height, width), 0.5, dtype=np.float32)

        # Average attention across heads, then rollout across layers
        result = None
        for attn_avg in self.attention_maps:
            # Already head-averaged at capture time: (B, seq_len, seq_len).
            # A lower-rank tensor here would silently broadcast against the
            # identity below and yield a bogus map rather than failing.
            if attn_avg.dim() != 3 or attn_avg.size(-1) != attn_avg.size(-2):
                raise RuntimeError(
                    "Expected batched square (B, seq_len, seq_len) attention, got "
                    f"{tuple(attn_avg.shape)}."
                )

            # Add residual connection (identity matrix)
            identity = torch.eye(attn_avg.size(-1), device=attn_avg.device).unsqueeze(0)
            attn_avg = 0.5 * attn_avg + 0.5 * identity

            # Re-normalize rows
            attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)

            # Multiply across layers (rollout)
            result = attn_avg if result is None else attn_avg @ result

        # [CLS] token attention to all patches (skip [CLS] itself)
        cls_attention = result[:, 0, 1:]  # (B, num_patches)

        num_patches = cls_attention.size(1)
        grid_size = int(round(num_patches ** 0.5))
        if grid_size * grid_size != num_patches:
            raise ValueError(f"Expected a square patch grid, got {num_patches} tokens.")

        grid = cls_attention.reshape(batch, 1, grid_size, grid_size)
        heatmaps = F.interpolate(
            grid, size=(height, width), mode="bilinear", align_corners=False
        ).squeeze(1)

        # Normalize each sample independently
        flat = heatmaps.flatten(1)
        minimum = flat.min(dim=1).values.view(-1, 1, 1)
        maximum = flat.max(dim=1).values.view(-1, 1, 1)
        heatmaps = (heatmaps - minimum) / (maximum - minimum + 1e-8)

        return heatmaps.cpu().numpy()

    def generate(self, image: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """
        Generate an attention rollout heatmap for a single image.

        Args:
            image: (1, 3, H, W) tensor on the model's device.
            class_idx: Ignored; rollout is class-agnostic.

        Returns:
            heatmap: (H, W) numpy array in [0, 1].
        """
        return self.generate_batch(image)[0]


# =============================================================================
# Visualization helpers (shared with localization.py)
# =============================================================================
def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalization and convert to (H, W, 3) uint8 array.

    Public because localization.py renders its own box-annotated figures and
    needs the identical inverse of the transform in dataset.py — if the two
    drifted apart, one module's overlays would be tinted relative to the
    other's, which is exactly the kind of difference nobody notices in a
    heatmap. Keep this in step with get_eval_transforms()'s Normalize.
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
    """
    Overlay a heatmap on the original image and save to disk.

    Args:
        image_tensor: (3, H, W) normalized tensor.
        heatmap: (H, W) array in [0, 1].
        title: Plot title.
        save_path: Output file path.
    """
    original = denormalize(image_tensor)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(original)
    axes[0].set_title("Original X-ray")
    axes[0].axis("off")

    # Heatmap alone
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Heatmap")
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(original)
    axes[2].imshow(heatmap, cmap="jet", alpha=0.4)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Explainer selection
# =============================================================================
def build_explainers(model: nn.Module, model_name: str) -> list[tuple[str, str, object]]:
    """
    Explainers available for an architecture, as (key, label, instance).

    Grad-CAM for every model so results are comparable across architectures;
    Attention Rollout additionally for ViT as a native second view. Shared by
    the qualitative sampler and the localization scorer so the two can never
    disagree about which methods exist.

    Grad-CAM instances attach hooks — call remove_hooks() when finished.
    """
    explainers: list[tuple[str, str, object]] = []
    if model_name in GRADCAM_TARGETS:
        explainers.append(("gradcam", "Grad-CAM", GradCAM(model, model_name=model_name)))
    if model_name == "vit_b_16":
        explainers.append(("rollout", "Attention Rollout", AttentionRollout(model)))
    return explainers


def release_explainers(explainers: list[tuple[str, str, object]]) -> None:
    """Detach any hooks the explainers installed."""
    for _, _, explainer in explainers:
        if isinstance(explainer, GradCAM):
            explainer.remove_hooks()


# =============================================================================
# End-to-end XAI generation
# =============================================================================
def generate_explanations(
    model: nn.Module,
    test_loader: DataLoader,
    model_name: str,
    thresholds: np.ndarray | None = None,
    num_samples: int | None = None,
) -> None:
    """
    Generate and save XAI visualizations for a trained model.

    Grad-CAM runs for every architecture, so heatmaps are comparable across
    models. ViT-B/16 additionally gets Attention Rollout as a native second
    view. Output goes to xai/<model>/<method>/, which makes it easy to line up
    one method across all four models.

    Args:
        model: Trained model (already on the right device, in eval mode).
        test_loader: Test DataLoader.
        model_name: One of config.SUPPORTED_MODELS.
        thresholds: Per-class calibrated decision thresholds.
        num_samples: Number of sample images to visualize. Defaults to
            config.XAI_NUM_SAMPLES, read here rather than as a default argument:
            --xai-samples rewrites that value at runtime, long after this module
            was imported, so a default bound at import time would always be the
            0 the CLI was meant to override.
    """
    if num_samples is None:
        num_samples = config.XAI_NUM_SAMPLES

    device = next(model.parameters()).device
    model.eval()

    # Use default thresholds if not provided
    if thresholds is None:
        thresholds = np.full(config.NUM_CLASSES, config.DEFAULT_THRESHOLD, dtype=np.float32)
    else:
        thresholds = np.asarray(thresholds, dtype=np.float32)

    # Output directory
    xai_dir = config.XAI_DIR / model_name

    explainers = build_explainers(model, model_name)

    if not explainers:
        print(f"[xai] No XAI method defined for '{model_name}' — skipping.")
        return

    for method_key, _, _ in explainers:
        (xai_dir / method_key).mkdir(parents=True, exist_ok=True)

    method_list = ", ".join(label for _, label, _ in explainers)
    print(f"[xai] Generating {method_list} visualizations for {model_name} …")

    # Iterate over test dataset
    count = 0
    try:
        for batch in test_loader:
            images = batch["image"]
            labels = batch["label"]
            filenames = batch["filename"]

            for i in range(images.size(0)):
                if count >= num_samples:
                    break

                img = images[i].unsqueeze(0).to(device)
                label_vec = labels[i].numpy()

                # Find the predicted (or ground-truth) top class for visualization - Forward pass
                with torch.no_grad():
                    logits = model(img)
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()

                # Select class using calibrated thresholds
                positive_indices = np.where(probs >= thresholds)[0]
                if positive_indices.size > 0:
                    top_class_idx = int(positive_indices[probs[positive_indices].argmax()])
                    selection_note = "Calibrated positive"
                else:
                    top_class_idx = int(probs.argmax())
                    selection_note = "No calibrated positive; showing top score"

                top_class_name = config.CLASS_NAMES[top_class_idx]
                top_prob = probs[top_class_idx]
                top_threshold = thresholds[top_class_idx]

                # Ground truth labels
                gt_classes = [config.CLASS_NAMES[j] for j, v in enumerate(label_vec) if v > 0.5]
                gt_str = ", ".join(gt_classes) if gt_classes else "No Finding"

                # Same image and same target class through every explainer
                stem = filenames[i].replace(".png", "")
                for method_key, method_label, explainer in explainers:
                    heatmap = explainer.generate(img, top_class_idx)

                    note = selection_note
                    if method_key == "rollout":
                        note = f"{selection_note} (rollout is class-agnostic)"

                    title = (
                        f"{method_label} — {model_name}\n"
                        f"{note}: {top_class_name} "
                        f"({top_prob:.2f}, thr={top_threshold:.2f})"
                        f"  |  GT: {gt_str}"
                    )
                    save_path = str(
                        xai_dir / method_key / f"sample_{count:03d}_{stem}.png"
                    )
                    overlay_heatmap(images[i], heatmap, title, save_path)

                count += 1

            if count >= num_samples:
                break
    finally:
        # Grad-CAM leaves hooks on the model; always take them off again
        release_explainers(explainers)

    print(
        f"[xai] Saved {count} samples x {len(explainers)} method(s) → {xai_dir}"
    )
