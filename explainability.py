"""
Explainability module

Implements two XAI techniques appropriate for each architecture:

  1. Grad-CAM — for DenseNet-121 and EfficientNet-B4 (CNN)
     Visualizes which spatial regions influenced a prediction by weighting
     the last convolutional feature maps by the gradient of the target class.

  2. Attention Rollout — for ViT-B/16 (ViT)
     Recursively multiplies attention matrices across all transformer layers
     to produce a single spatial attention map from the [CLS] token.

Both methods overlay a heatmap on the original X-ray and save the result.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib

# Non-interactive backend for saving figures
matplotlib.use("Agg")

import config


# =============================================================================
# 1. Grad-CAM for CNN (DenseNet-121 / EfficientNet-B4)
# =============================================================================
class GradCAM:
    """
    Grad-CAM implementation for CNN models (DenseNet-121 and EfficientNet-B4).

    Hooks into the last convolutional layer to capture activations and
    gradients, then produces a class-discriminative heatmap.

    Target layers:
      - DenseNet-121: features.denseblock4
      - EfficientNet-B4: features[-1] (last MBConv block)
    """

    def __init__(self, model: nn.Module, model_name: str = "densenet121"):
        self.model = model
        self.model.eval()

        # Store intermediate activations and gradients
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        # Select target layer based on architecture
        if model_name == "efficientnet_b4":
            target_layer = model.backbone.features[-1]
        else:
            # DenseNet-121 default
            target_layer = model.backbone.features.denseblock4
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    # Save forward activations
    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    # Save backward gradients
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for a given class.

        Args:
            image: (1, 3, H, W) tensor on the model's device.
            class_idx: Index of the target class.

        Returns:
            heatmap: (H, W) numpy array in [0, 1].
        """
        self.model.zero_grad()

        # Forward pass
        logits = self.model(image)

        # Select target class score
        target = logits[0, class_idx]
        target.backward()

        # Channel importance via global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of feature maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)

        # Keep only positive contributions
        cam = F.relu(cam)

        # Upsample to input resolution and normalize
        cam = F.interpolate(cam, size=image.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


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

                    # Store the attention weights
                    self.attention_maps.append(attn_weights.detach())
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
    def generate(self, image: torch.Tensor) -> np.ndarray:
        """
        Generate an attention rollout heatmap.

        Args:
            image: (1, 3, H, W) tensor on the model's device.

        Returns:
            heatmap: (H, W) numpy array in [0, 1].
        """
        self.attention_maps.clear()

        # Temporarily patch to capture attention weights
        self._patch_encoder_blocks()
        try:
            _ = self.model(image)
        finally:
            self._unpatch_encoder_blocks()

        # Fallback if no attention captured
        if not self.attention_maps:
            h = image.shape[2]
            return np.ones((h, h), dtype=np.float32) * 0.5

        # Average attention across heads, then rollout across layers
        result = None
        for attn in self.attention_maps:
            # attn: (1, num_heads, seq_len, seq_len) → average over heads
            attn_avg = attn.mean(dim=1).squeeze(0)  # (seq_len, seq_len)

            # Guard against a collapsed head axis: a 1D result would silently
            # broadcast against the identity below and produce a bogus map.
            if attn_avg.dim() != 2 or attn_avg.size(0) != attn_avg.size(1):
                raise RuntimeError(
                    "Expected a square (seq_len, seq_len) attention matrix, got "
                    f"{tuple(attn_avg.shape)}. The attention weights were captured "
                    "with the head axis already averaged."
                )

            # Add residual connection (identity matrix)
            attn_avg = 0.5 * attn_avg + 0.5 * torch.eye(attn_avg.size(0), device=attn_avg.device)

            # Re-normalize rows
            attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)

            # Multiply across layers (rollout)
            if result is None:
                result = attn_avg
            else:
                result = attn_avg @ result

        # Extract [CLS] token attention to all patches (skip [CLS] itself)
        cls_attention = result[0, 1:]  # (num_patches,)

        # Reshape to 2D grid
        num_patches = cls_attention.shape[0]
        grid_size = int(num_patches ** 0.5)
        cls_attention = cls_attention.reshape(grid_size, grid_size).cpu().numpy()

        # Upsample to input resolution
        heatmap = np.array(
            Image.fromarray(cls_attention).resize(
                (image.shape[3], image.shape[2]), resample=Image.BILINEAR
            )
        )

        # Normalize to [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        return heatmap


# =============================================================================
# Visualization helper
# =============================================================================
def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalization and convert to (H, W, 3) uint8 array."""
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
    original = _denormalize(image_tensor)

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
# End-to-end XAI generation
# =============================================================================
def generate_explanations(
    model: nn.Module,
    test_loader: DataLoader,
    model_name: str,
    thresholds: np.ndarray | None = None,
    num_samples: int = config.XAI_NUM_SAMPLES,
) -> None:
    """
    Generate and save XAI visualizations for a trained model.

    For DenseNet-121: uses Grad-CAM.
    For ViT-B/16: uses Attention Rollout.

    Args:
        model: Trained model (already on the right device, in eval mode).
        test_loader: Test DataLoader.
        model_name: "densenet121" or "vit_b_16".
        thresholds: Per-class calibrated decision thresholds.
        num_samples: Number of sample images to visualize.
    """
    device = next(model.parameters()).device
    model.eval()

    # Use default thresholds if not provided
    if thresholds is None:
        thresholds = np.full(config.NUM_CLASSES, config.DEFAULT_THRESHOLD, dtype=np.float32)
    else:
        thresholds = np.asarray(thresholds, dtype=np.float32)

    # Output directory
    xai_dir = config.XAI_DIR / model_name
    xai_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the appropriate explainer
    if model_name == "densenet121":
        explainer = GradCAM(model, model_name="densenet121")
        method_name = "Grad-CAM"
    elif model_name == "efficientnet_b4":
        explainer = GradCAM(model, model_name="efficientnet_b4")
        method_name = "Grad-CAM"
    elif model_name == "vit_b_16":
        explainer = AttentionRollout(model)
        method_name = "Attention Rollout"
    elif model_name == "swin_v2_b":
        print("[xai] SwinV2 explainability not implemented.")
        return
    else:
        print(f"[xai] No XAI method defined for '{model_name}' — skipping.")
        return

    print(f"[xai] Generating {method_name} visualizations for {model_name} …")

    # Iterate over test dataset
    count = 0
    for batch in test_loader:
        images = batch["image"]
        labels = batch["label"]
        filenames = batch["filename"]

        for i in range(images.size(0)):
            if count >= num_samples:
                return

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

            # Generate heatmap
            if model_name in ("densenet121", "efficientnet_b4"):
                heatmap = explainer.generate(img, top_class_idx)
            else:
                heatmap = explainer.generate(img)

            # Save visualization
            title = (
                f"{method_name} — {model_name}\n"
                f"{selection_note}: {top_class_name} ({top_prob:.2f}, thr={top_threshold:.2f})"
                f"  |  GT: {gt_str}"
            )
            save_path = str(xai_dir / f"sample_{count:03d}_{filenames[i].replace('.png', '')}.png")
            overlay_heatmap(images[i], heatmap, title, save_path)

            count += 1

    print(f"[xai] Saved {count} visualizations → {xai_dir}")
