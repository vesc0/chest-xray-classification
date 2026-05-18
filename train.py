"""
Training loop

Features:
  - Long-tail aware loss functions (ASL / weighted BCE / BCE)
  - AdamW optimizer with warmup + cosine decay
  - Early stopping and checkpointing on an imbalance-sensitive metric
  - Mixed precision on CUDA
  - Training history logging (loss / AUROC / AUPRC per epoch)
"""

import time

import numpy as np

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from sklearn.metrics import average_precision_score, roc_auc_score

import config


# =============================================================================
# Device utilities
# =============================================================================
def _get_device() -> torch.device:
    """Select the best available device: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _has_both_labels(labels: np.ndarray) -> bool:
    """AUROC is defined only when both positive and negative labels exist."""
    positives = labels.sum()
    return 0 < positives < len(labels)


# =============================================================================
# Metric computation
# =============================================================================
def _compute_macro_auroc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Compute macro AUROC, skipping undefined classes."""
    aurocs = []
    for class_idx in range(labels.shape[1]):
        targets = labels[:, class_idx]
        if _has_both_labels(targets):
            aurocs.append(roc_auc_score(targets, probs[:, class_idx]))
    return float(np.mean(aurocs)) if aurocs else 0.0


def _compute_macro_auprc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Compute macro average precision, skipping classes with no positives."""
    auprcs = []
    for class_idx in range(labels.shape[1]):
        targets = labels[:, class_idx]
        if targets.sum() > 0:
            auprcs.append(average_precision_score(targets, probs[:, class_idx]))
    return float(np.mean(auprcs)) if auprcs else 0.0


def _compute_epoch_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Summarize threshold-free ranking metrics for one epoch."""
    return {
        "auroc": _compute_macro_auroc(labels, probs),
        "auprc": _compute_macro_auprc(labels, probs),
    }


# =============================================================================
# Loss functions
# =============================================================================
class AsymmetricLoss(nn.Module):
    """
    Asymmetric loss for long-tailed multi-label classification.

    Based on the common ASL formulation where negatives are down-weighted more
    aggressively than positives. This is a strong default for multi-label
    medical classification because it reduces the "predict-all-negatives" bias.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_pos = probs.clamp(min=self.eps, max=1.0 - self.eps)
        probs_neg = 1.0 - probs

        # Prevent extreme negative probabilities
        if self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1.0)
        probs_neg = probs_neg.clamp(min=self.eps, max=1.0 - self.eps)

        loss_pos = targets * torch.log(probs_pos)
        loss_neg = (1.0 - targets) * torch.log(probs_neg)

        # Focus more on hard examples
        pos_weight = torch.pow(1.0 - probs_pos, self.gamma_pos) * targets
        neg_weight = torch.pow(1.0 - probs_neg, self.gamma_neg) * (1.0 - targets)
        asymmetric_weight = pos_weight + neg_weight

        loss = -(loss_pos + loss_neg) * asymmetric_weight
        return loss.mean()


def _compute_pos_weight(train_loader: DataLoader, device: torch.device) -> torch.Tensor:
    """Compute clipped class-wise positive weights from the training labels."""
    label_matrix = train_loader.dataset.get_label_matrix()
    positives = label_matrix.sum(axis=0)
    negatives = len(label_matrix) - positives
    pos_weight = (negatives + 1.0) / (positives + 1.0)
    # Prevent extremely large weights
    pos_weight = np.clip(pos_weight, 1.0, config.MAX_POS_WEIGHT)
    return torch.tensor(pos_weight, dtype=torch.float32, device=device)


def _build_loss(train_loader: DataLoader, device: torch.device) -> nn.Module:
    """Build the configured loss function."""
    loss_name = config.LOSS_NAME.lower()
    if loss_name == "asymmetric":
        return AsymmetricLoss(
            gamma_neg=config.ASL_GAMMA_NEG,
            gamma_pos=config.ASL_GAMMA_POS,
            clip=config.ASL_CLIP,
        )
    if loss_name == "weighted_bce":
        return nn.BCEWithLogitsLoss(pos_weight=_compute_pos_weight(train_loader, device))
    if loss_name == "bce":
        return nn.BCEWithLogitsLoss()
    raise ValueError(
        f"Unsupported loss '{config.LOSS_NAME}'. "
        "Use one of: asymmetric, weighted_bce, bce."
    )


# =============================================================================
# Optimizer and scheduler
# =============================================================================
def _build_optimizer(
    backbone_params: list[nn.Parameter],
    head_params: list[nn.Parameter],
) -> torch.optim.Optimizer:
    """Build AdamW with optional separate parameter groups for head/backbone."""
    param_groups = []
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": config.LEARNING_RATE * config.BACKBONE_LR_MULTIPLIER,
            }
        )
    if head_params:
        param_groups.append(
            {
                "params": head_params,
                "lr": config.LEARNING_RATE * config.HEAD_LR_MULTIPLIER,
            }
        )

    if not param_groups:
        param_groups = [{"params": list(backbone_params) + list(head_params), "lr": config.LEARNING_RATE}]

    return torch.optim.AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)


def _build_scheduler(optimizer: torch.optim.Optimizer):
    """Warm up the optimizer before cosine decay."""
    if config.NUM_EPOCHS <= 1:
        return None

    if config.WARMUP_EPOCHS > 0 and config.NUM_EPOCHS > config.WARMUP_EPOCHS:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.2,
            end_factor=1.0,
            total_iters=config.WARMUP_EPOCHS,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(config.NUM_EPOCHS - config.WARMUP_EPOCHS, 1),
            eta_min=config.MIN_LEARNING_RATE,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[config.WARMUP_EPOCHS],
        )

    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.NUM_EPOCHS, 1),
        eta_min=config.MIN_LEARNING_RATE,
    )


# =============================================================================
# Fine-tuning utilities
# =============================================================================
def _split_backbone_head_params(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split parameters into backbone and classifier-head groups."""
    if not hasattr(model, "backbone"):
        return list(model.parameters()), []

    backbone = model.backbone
    head_module = None

    # DenseNet
    if hasattr(backbone, "classifier"):
        head_module = backbone.classifier

    # ViT
    elif hasattr(backbone, "heads"):
        head_module = backbone.heads

    if head_module is None:
        return list(model.parameters()), []

    head_param_ids = {id(param) for param in head_module.parameters()}
    backbone_params, head_params = [], []
    for param in model.parameters():
        if id(param) in head_param_ids:
            head_params.append(param)
        else:
            backbone_params.append(param)
    return backbone_params, head_params


def _set_requires_grad(parameters: list[nn.Parameter], flag: bool) -> None:
    """Enable or disable gradient computation."""
    for param in parameters:
        param.requires_grad = flag


def _apply_tuning_mode(
    model: nn.Module,
    epoch: int,
    backbone_params: list[nn.Parameter],
    head_params: list[nn.Parameter],
) -> str:
    """Set trainable params for the current epoch and return stage label."""
    mode = config.TUNING_MODE.lower()

    if mode == "full":
        _set_requires_grad(backbone_params, True)
        _set_requires_grad(head_params, True)
        return "full"

    if mode == "head_only":
        _set_requires_grad(backbone_params, False)
        _set_requires_grad(head_params, True)
        return "head_only"

    if mode == "partial":
        freeze_epochs = max(int(config.FREEZE_EPOCHS), 0)

        # Always train classifier head
        _set_requires_grad(head_params, True)

        # Warmup phase: train only head
        if epoch <= freeze_epochs:
            _set_requires_grad(backbone_params, False)
            return "partial_head_warmup"
        
        # Unfreeze part of the backbone
        _set_requires_grad(backbone_params, False)

        fraction = float(config.PARTIAL_UNFREEZE_FRACTION)
        fraction = min(max(fraction, 0.0), 1.0)

        n_unfrozen = int(len(backbone_params) * fraction)
        n_unfrozen = max(n_unfrozen, 1) if backbone_params else 0

        # Unfreeze final layers
        for param in backbone_params[-n_unfrozen:]:
            param.requires_grad = True
        return "partial_unfrozen"

    raise ValueError(
        f"Unsupported tuning mode '{config.TUNING_MODE}'. "
        "Use one of: full, head_only, partial."
    )


def _count_trainable_params(model: nn.Module) -> int:
    """Count trainable model parameters."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


# =============================================================================
# Training and validation loops
# =============================================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None,
) -> dict[str, float]:
    """Run one training epoch and return loss plus threshold-free metrics."""
    model.train()
    running_loss = 0.0
    all_labels, all_probs = [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        # Clear old gradients
        optimizer.zero_grad(set_to_none=True)

        use_amp = scaler is not None and device.type == "cuda"

        # Forward pass
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)

        all_labels.append(labels.detach().cpu().numpy())
        all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())

    labels_np = np.concatenate(all_labels)
    probs_np = np.concatenate(all_probs)

    metrics = _compute_epoch_metrics(labels_np, probs_np)
    metrics["loss"] = running_loss / max(len(loader.dataset), 1)

    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Run validation and return loss plus threshold-free metrics."""
    model.eval()

    running_loss = 0.0
    all_labels, all_probs = [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)

        all_labels.append(labels.cpu().numpy())
        all_probs.append(torch.sigmoid(logits).cpu().numpy())

    labels_np = np.concatenate(all_labels)
    probs_np = np.concatenate(all_probs)

    metrics = _compute_epoch_metrics(labels_np, probs_np)
    metrics["loss"] = running_loss / max(len(loader.dataset), 1)

    return metrics


def _get_monitored_metric(metrics: dict[str, float]) -> tuple[float, bool]:
    """Return the metric used for checkpointing and whether smaller is better."""
    metric_name = config.CHECKPOINT_METRIC.lower()
    if metric_name == "val_loss":
        return metrics["loss"], True
    if metric_name == "val_auroc":
        return metrics["auroc"], False
    if metric_name == "val_auprc":
        return metrics["auprc"], False
    raise ValueError(
        f"Unsupported checkpoint metric '{config.CHECKPOINT_METRIC}'. "
        "Use one of: val_loss, val_auroc, val_auprc."
    )


# =============================================================================
# Training pipeline
# =============================================================================
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str,
) -> dict:
    """
    Train a model end-to-end with early stopping.

    Returns a history dict for plotting and experiment tracking.
    """
    device = _get_device()
    print(f"[train] Using device: {device}")
    model = model.to(device)

    criterion = _build_loss(train_loader, device)
    backbone_params, head_params = _split_backbone_head_params(model)
    optimizer = _build_optimizer(backbone_params, head_params)
    scheduler = _build_scheduler(optimizer)
    scaler = GradScaler("cuda") if device.type == "cuda" else None

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = config.CHECKPOINT_DIR / f"{model_name}_best.pt"

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_auroc": [],
        "train_auprc": [],
        "val_loss": [],
        "val_auroc": [],
        "val_auprc": [],
        "lr": [],
    }

    best_value = float("inf") if config.CHECKPOINT_METRIC == "val_loss" else float("-inf")
    patience_counter = 0
    last_stage = None

    for epoch in range(1, config.NUM_EPOCHS + 1):
        t0 = time.time()

        # Configure trainable layers
        current_stage = _apply_tuning_mode(model, epoch, backbone_params, head_params)
        trainable_params = _count_trainable_params(model)
        if current_stage != last_stage:
            print(
                f"  [train] Tuning stage -> {current_stage} "
                f"(trainable params: {trainable_params:,})"
            )
            last_stage = current_stage

        # Training + validation
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
        )
        val_metrics = validate(model, val_loader, criterion, device)

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[-1]["lr"]
        elapsed = time.time() - t0

        print(
            f"  Epoch {epoch:02d}/{config.NUM_EPOCHS} "
            f"| train_loss={train_metrics['loss']:.4f} "
            f"train_AUROC={train_metrics['auroc']:.4f} "
            f"train_AUPRC={train_metrics['auprc']:.4f} "
            f"| val_loss={val_metrics['loss']:.4f} "
            f"val_AUROC={val_metrics['auroc']:.4f} "
            f"val_AUPRC={val_metrics['auprc']:.4f} "
            f"| lr={current_lr:.2e} | {elapsed:.1f}s"
        )

        # Store history
        history["train_loss"].append(train_metrics["loss"])
        history["train_auroc"].append(train_metrics["auroc"])
        history["train_auprc"].append(train_metrics["auprc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_auroc"].append(val_metrics["auroc"])
        history["val_auprc"].append(val_metrics["auprc"])
        history["lr"].append(current_lr)

        # Check improvement
        current_value, smaller_is_better = _get_monitored_metric(val_metrics)
        is_improved = (
            current_value < best_value if smaller_is_better else current_value > best_value
        )

        if is_improved:
            best_value = current_value
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt_path)
            print(
                f"  [train] Saved best checkpoint based on "
                f"{config.CHECKPOINT_METRIC}={current_value:.4f}"
            )
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(
                f"  [train] Early stopping at epoch {epoch} "
                f"(patience={config.EARLY_STOPPING_PATIENCE})"
            )
            break

    # Load best model weights
    model.load_state_dict(
        torch.load(best_ckpt_path, map_location=device, weights_only=True)
    )
    print(f"[train] Loaded best checkpoint ({config.CHECKPOINT_METRIC}={best_value:.4f})")

    return history
