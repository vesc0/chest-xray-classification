"""
Evaluation module

Computes and reports:
  - Per-class AUROC / AUPRC
  - Threshold-calibrated precision / recall / F1
  - Micro/macro aggregate metrics
  - Calibration metrics (Brier score, ECE)
  - Validation-set threshold tuning for long-tailed labels
"""

import json
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    fbeta_score,
    hamming_loss as sk_hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config
from device import synchronize
from metrics import (
    class_auprc,
    class_auroc,
    has_both_labels,
    macro_average,
)


# =============================================================================
# Inference
# =============================================================================
@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Run inference on an entire loader.

    Returns:
        labels: (N, NUM_CLASSES) ground-truth multi-hot
        probs:  (N, NUM_CLASSES) predicted probabilities
        timing: wall-clock breakdown

    Timing separates two figures that are easy to conflate. End-to-end includes
    image decode and transforms, so it reflects what a deployment would feel but
    is dominated by the DataLoader for small models; model-only isolates the
    forward pass and is the fair architecture comparison. The first batch is
    excluded from model-only time as warm-up, since it carries lazy allocation
    and kernel selection.
    """
    model.eval()
    all_labels, all_probs = [], []

    model_seconds = 0.0
    warmup_images = 0
    total_images = 0

    synchronize(device)
    start = time.perf_counter()

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)

        # Labels are kept on CPU to avoid GPU overhead
        labels = batch["label"].cpu().numpy()

        synchronize(device)
        forward_start = time.perf_counter()
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(images)
        synchronize(device)
        forward_seconds = time.perf_counter() - forward_start

        if batch_idx == 0:
            warmup_images = images.size(0)
        else:
            model_seconds += forward_seconds

        total_images += images.size(0)

        probs = torch.sigmoid(logits).cpu().numpy()
        all_labels.append(labels)
        all_probs.append(probs)

    synchronize(device)
    total_seconds = time.perf_counter() - start

    timed_images = max(total_images - warmup_images, 0)
    timing = {
        "num_images": total_images,
        "total_seconds": round(total_seconds, 3),
        "images_per_second": round(total_images / total_seconds, 1) if total_seconds > 0 else None,
        "ms_per_image": round(1000.0 * total_seconds / total_images, 3) if total_images else None,
        "model_seconds": round(model_seconds, 3),
        "model_images": timed_images,
        "model_images_per_second": (
            round(timed_images / model_seconds, 1) if model_seconds > 0 else None
        ),
        "model_ms_per_image": (
            round(1000.0 * model_seconds / timed_images, 3) if timed_images else None
        ),
    }

    labels_np = np.concatenate(all_labels)
    probs_np = np.concatenate(all_probs)
    return labels_np, probs_np, timing


# =============================================================================
# Thresholding
# =============================================================================
def apply_thresholds(probs: np.ndarray, thresholds: np.ndarray | float | None) -> np.ndarray:
    """Convert probabilities into binary predictions using per-class thresholds."""
    if thresholds is None:
        thresholds = config.DEFAULT_THRESHOLD
    return (probs >= thresholds).astype(np.int32)


def _threshold_objective(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> tuple[float, float, float]:
    """Score a decision threshold and return (objective, precision, recall)."""
    preds = (probs >= threshold).astype(np.int32)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)

    metric_name = config.THRESHOLD_METRIC.lower()
    if metric_name == "f1":
        score = f1_score(labels, preds, zero_division=0)
    elif metric_name == "fbeta":
        score = fbeta_score(labels, preds, beta=config.THRESHOLD_BETA, zero_division=0)
    elif metric_name == "youden":
        negatives = len(labels) - labels.sum()
        tn = int(((preds == 0) & (labels == 0)).sum())
        specificity = tn / negatives if negatives > 0 else 0.0
        score = recall + specificity - 1.0
    else:
        raise ValueError(
            f"Unsupported threshold metric '{config.THRESHOLD_METRIC}'. "
            "Use one of: f1, fbeta, youden."
        )

    return float(score), float(precision), float(recall)


def tune_thresholds(
    labels: np.ndarray,
    probs: np.ndarray,
) -> tuple[np.ndarray, dict[str, dict[str, float | int | str]]]:
    """
    Tune one threshold per class using the validation set.

    Classes with too few positives fall back to config.DEFAULT_THRESHOLD to
    avoid unstable operating points.
    """
    thresholds = np.full(labels.shape[1], config.DEFAULT_THRESHOLD, dtype=np.float32)
    summary: dict[str, dict[str, float | int | str]] = {}
    threshold_grid = np.linspace(
        config.THRESHOLD_MIN,
        config.THRESHOLD_MAX,
        config.THRESHOLD_STEPS,
    )

    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        y_true = labels[:, class_idx]
        y_prob = probs[:, class_idx]
        support = int(y_true.sum())

        if support < config.THRESHOLD_MIN_SUPPORT or not has_both_labels(y_true):
            summary[class_name] = {
                "threshold": float(config.DEFAULT_THRESHOLD),
                "objective": None,
                "support": support,
                "status": "default",
            }
            continue

        best_threshold = float(config.DEFAULT_THRESHOLD)
        best_score = float("-inf")
        best_precision = 0.0
        best_recall = 0.0

        for threshold in threshold_grid:
            score, precision, recall = _threshold_objective(y_true, y_prob, float(threshold))

            # Small tie-break bias toward recall
            if (
                score > best_score + 1e-8
                or (
                    abs(score - best_score) <= 1e-8
                    and recall > best_recall + 1e-8
                )
            ):
                best_threshold = float(threshold)
                best_score = score
                best_precision = precision
                best_recall = recall

        thresholds[class_idx] = best_threshold
        summary[class_name] = {
            "threshold": round(best_threshold, 4),
            "objective": round(best_score, 4),
            "precision": round(best_precision, 4),
            "recall": round(best_recall, 4),
            "support": support,
            "status": "tuned",
        }

    return thresholds, summary


def save_thresholds(
    thresholds: np.ndarray,
    tuning_summary: dict[str, dict[str, float | int | str]],
    model_name: str,
) -> Path:
    """Persist tuned thresholds for reproducible evaluation and XAI."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / f"{model_name}_thresholds.json"
    payload = {
        "threshold_metric": config.THRESHOLD_METRIC,
        "default_threshold": config.DEFAULT_THRESHOLD,
        "thresholds": {
            class_name: round(float(thresholds[idx]), 4)
            for idx, class_name in enumerate(config.CLASS_NAMES)
        },
        "tuning_summary": tuning_summary,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[evaluate] Thresholds saved -> {path}")
    return path


def calibrate_thresholds(
    model: nn.Module,
    val_loader: DataLoader,
    model_name: str,
) -> np.ndarray:
    """Tune per-class thresholds on the validation set and save them."""
    device = next(model.parameters()).device
    print(f"[evaluate] Calibrating thresholds on the validation set ({len(val_loader.dataset)} samples) ...")
    labels, probs, _ = collect_predictions(model, val_loader, device)
    thresholds, summary = tune_thresholds(labels, probs)
    save_thresholds(thresholds, summary, model_name)
    return thresholds


def _expected_calibration_error(
    labels: np.ndarray,
    probs: np.ndarray,
    num_bins: int = config.ECE_BINS,
) -> float:
    """Compute binary expected calibration error for one class."""
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0

    for start, end in zip(bins[:-1], bins[1:]):
        if end == 1.0:
            mask = (probs >= start) & (probs <= end)
        else:
            mask = (probs >= start) & (probs < end)

        if not np.any(mask):
            continue

        confidence = probs[mask].mean()
        accuracy = labels[mask].mean()
        ece += np.abs(confidence - accuracy) * (mask.sum() / len(probs))

    return float(ece)


def _normal_vs_abnormal_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    """
    Screening metric derived from the 14 pathology outputs.

    "No Finding" is not a trained class, so normality is read off the pathology
    predictions: a study is abnormal when any of the 14 is present. Two standard
    aggregations are reported — the strongest single finding (max) and the
    probabilistic OR under an independence assumption (noisy-or). This keeps the
    clinically useful normal/abnormal signal without letting a 54%-prevalence
    class into the macro and micro averages.
    """
    y_true = (labels.sum(axis=1) > 0).astype(np.int32)
    scores = {
        "max": probs.max(axis=1),
        "noisy_or": 1.0 - np.prod(1.0 - probs, axis=1),
    }

    results = {"prevalence": round(float(y_true.mean()), 4)}
    for name, score in scores.items():
        if has_both_labels(y_true):
            results[f"{name}_auroc"] = round(float(roc_auc_score(y_true, score)), 4)
            results[f"{name}_auprc"] = round(float(average_precision_score(y_true, score)), 4)
        else:
            results[f"{name}_auroc"] = None
            results[f"{name}_auprc"] = None

    return results


# =============================================================================
# Metrics computation
# =============================================================================
def compute_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    thresholds: np.ndarray | float | None = None,
) -> dict:
    """
    Compute a comprehensive set of evaluation metrics.

    Returns a dict containing per-class, macro, micro, and calibration figures.
    """
    if thresholds is None:
        thresholds = np.full(labels.shape[1], config.DEFAULT_THRESHOLD, dtype=np.float32)
    elif np.isscalar(thresholds):
        thresholds = np.full(labels.shape[1], float(thresholds), dtype=np.float32)
    else:
        thresholds = np.asarray(thresholds, dtype=np.float32)

    results: dict = {
        # Recorded so a comparison across runs can verify the class sets match
        "class_names": list(config.CLASS_NAMES),
        "num_classes": config.NUM_CLASSES,
        "per_class": {},
        "macro": {},
        "micro": {},
        "normal_vs_abnormal": _normal_vs_abnormal_metrics(labels, probs),
        "calibration": {},
        "thresholds": {
            class_name: round(float(thresholds[idx]), 4)
            for idx, class_name in enumerate(config.CLASS_NAMES)
        },
    }

    per_class_auroc = []
    per_class_auprc = []
    per_class_brier = []
    per_class_ece = []

    # Per-class metrics
    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        targets = labels[:, class_idx]
        scores = probs[:, class_idx]
        predictions = preds[:, class_idx]
        support = int(targets.sum())

        # Same helpers the per-epoch training metrics use, so the validation
        # curve and the number reported here mean the same thing
        auroc = class_auroc(targets, scores)
        auprc = class_auprc(targets, scores)
        brier = float(np.mean((scores - targets) ** 2))
        ece = _expected_calibration_error(targets, scores)

        per_class_auroc.append(auroc)
        per_class_auprc.append(auprc)
        per_class_brier.append(brier)
        per_class_ece.append(ece)

        results["per_class"][class_name] = {
            "threshold": round(float(thresholds[class_idx]), 4),
            "auroc": round(float(auroc), 4) if not np.isnan(auroc) else None,
            "auprc": round(float(auprc), 4) if not np.isnan(auprc) else None,
            "precision": round(float(precision_score(targets, predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(targets, predictions, zero_division=0)), 4),
            "f1": round(float(f1_score(targets, predictions, zero_division=0)), 4),
            "brier": round(brier, 4),
            "ece": round(ece, 4),
            "support": support,
            "prevalence": round(float(targets.mean()), 4),
        }

    # Macro metrics — macro_average skips the classes that were not scorable
    macro_auroc = macro_average(per_class_auroc)
    macro_auprc = macro_average(per_class_auprc)
    results["macro"]["auroc"] = round(macro_auroc, 4) if macro_auroc is not None else None
    results["macro"]["auprc"] = round(macro_auprc, 4) if macro_auprc is not None else None
    results["macro"]["precision"] = round(
        float(precision_score(labels, preds, average="macro", zero_division=0)),
        4,
    )
    results["macro"]["recall"] = round(
        float(recall_score(labels, preds, average="macro", zero_division=0)),
        4,
    )
    results["macro"]["f1"] = round(
        float(f1_score(labels, preds, average="macro", zero_division=0)),
        4,
    )
    results["macro"]["samples_f1"] = round(
        float(f1_score(labels, preds, average="samples", zero_division=0)),
        4,
    )
    results["macro"]["subset_accuracy"] = round(float(accuracy_score(labels, preds)), 4)
    results["macro"]["sample_accuracy"] = round(
        float(((preds == labels).sum(axis=1) / labels.shape[1]).mean()),
        4,
    )
    results["macro"]["all_negative_sample_accuracy_baseline"] = round(
        float((1.0 - labels.mean(axis=0)).mean()),
        4,
    )
    results["macro"]["hamming_loss"] = round(float(sk_hamming_loss(labels, preds)), 4)

    # Micro metrics
    flat_labels = labels.ravel()
    flat_probs = probs.ravel()
    flat_preds = preds.ravel()
    results["micro"]["auprc"] = round(
        float(average_precision_score(flat_labels, flat_probs)),
        4,
    ) if flat_labels.sum() > 0 else None
    results["micro"]["precision"] = round(
        float(precision_score(flat_labels, flat_preds, zero_division=0)),
        4,
    )
    results["micro"]["recall"] = round(
        float(recall_score(flat_labels, flat_preds, zero_division=0)),
        4,
    )
    results["micro"]["f1"] = round(
        float(f1_score(flat_labels, flat_preds, zero_division=0)),
        4,
    )

    # Calibration metrics
    results["calibration"]["macro_brier"] = round(float(np.mean(per_class_brier)), 4)
    results["calibration"]["macro_ece"] = round(float(np.mean(per_class_ece)), 4)

    return results


# =============================================================================
# Output utilities
# =============================================================================
def print_results(results: dict, model_name: str) -> None:
    """Print a formatted summary table of evaluation results."""
    print(f"\n{'=' * 96}")
    print(f"  Evaluation Results - {model_name}")
    print(f"{'=' * 96}")

    header = (
        f"{'Class':<22} {'Thr':>6} {'AUROC':>8} {'AUPRC':>8} "
        f"{'Prec':>7} {'Recall':>7} {'F1':>7} {'Support':>8}"
    )
    print(header)
    print("-" * len(header))

    for class_name, metrics in results["per_class"].items():
        auroc = f"{metrics['auroc']:.4f}" if metrics["auroc"] is not None else "N/A"
        auprc = f"{metrics['auprc']:.4f}" if metrics["auprc"] is not None else "N/A"
        print(
            f"{class_name:<22} {metrics['threshold']:>6.2f} {auroc:>8} {auprc:>8} "
            f"{metrics['precision']:>7.4f} {metrics['recall']:>7.4f} "
            f"{metrics['f1']:>7.4f} {metrics['support']:>8d}"
        )

    print("-" * len(header))
    macro = results["macro"]
    micro = results["micro"]
    calibration = results["calibration"]

    macro_auroc = f"{macro['auroc']:.4f}" if macro["auroc"] is not None else "N/A"
    macro_auprc = f"{macro['auprc']:.4f}" if macro["auprc"] is not None else "N/A"
    micro_auprc = f"{micro['auprc']:.4f}" if micro["auprc"] is not None else "N/A"

    print(
        f"{'Macro average':<22} {'-':>6} {macro_auroc:>8} {macro_auprc:>8} "
        f"{macro['precision']:>7.4f} {macro['recall']:>7.4f} {macro['f1']:>7.4f}"
    )
    print(
        f"{'Micro average':<22} {'-':>6} {'-':>8} {micro_auprc:>8} "
        f"{micro['precision']:>7.4f} {micro['recall']:>7.4f} {micro['f1']:>7.4f}"
    )
    print(f"  Samples F1:                      {macro['samples_f1']:.4f}")
    print(f"  Subset accuracy:                 {macro['subset_accuracy']:.4f}")
    print(f"  Sample accuracy:                 {macro['sample_accuracy']:.4f}")
    print(
        f"  All-negative sample baseline:    "
        f"{macro['all_negative_sample_accuracy_baseline']:.4f}"
    )
    print(f"  Hamming loss:                    {macro['hamming_loss']:.4f}")
    print(f"  Macro Brier score:               {calibration['macro_brier']:.4f}")
    print(f"  Macro ECE:                       {calibration['macro_ece']:.4f}")

    # Derived screening metric — not part of the 14-class averages above
    screening = results.get("normal_vs_abnormal", {})
    if screening:
        def _fmt(value):
            return f"{value:.4f}" if value is not None else "N/A"

        print(
            f"  Normal vs abnormal AUROC:        "
            f"{_fmt(screening.get('max_auroc'))} (max) / "
            f"{_fmt(screening.get('noisy_or_auroc'))} (noisy-or)"
            f"   [abnormal prevalence {screening.get('prevalence', 0):.4f}]"
        )

    timing = results.get("timing", {})
    run = results.get("run", {})
    training = timing.get("training") or {}
    inference = timing.get("inference") or {}
    if run or training or inference:
        print(f"{'-' * 96}")
        trainable = run.get("trainable_params")
        trainable_str = (
            f"{trainable:,} trainable ({run.get('trainable_fraction') or 0:.2%})"
            if trainable else "trainable unknown"
        )
        origin = "" if run.get("trained_this_run", True) else "  [recovered]"
        print(
            f"  Device: {run.get('device')}  |  batch {run.get('batch_size')}  |  "
            f"workers {run.get('num_workers')}  |  AMP {run.get('amp')}"
        )
        print(
            f"  Model:  {run.get('total_params', 0):,} params, {trainable_str}  |  "
            f"tuning {run.get('tuning_mode') or 'unknown'}  |  "
            f"loss {run.get('loss') or 'unknown'}  |  "
            f"lr {run.get('learning_rate') if run.get('learning_rate') is not None else 'unknown'}"
            f"{origin}"
        )
    if training:
        origin = " [from the original training run]" if timing.get(
            "training_timing_carried_over"
        ) else ""
        print(
            f"  Training:  {training['total_seconds'] / 60:.1f} min over "
            f"{training['epochs_run']} epochs "
            f"({training['mean_epoch_seconds']:.1f}s/epoch, "
            f"best at epoch {training['best_epoch']} "
            f"after {training['seconds_to_best_epoch'] / 60:.1f} min){origin}"
        )
    if inference:
        print(
            f"  Inference: {inference['total_seconds']:.1f}s for "
            f"{inference['num_images']:,} images  |  "
            f"end-to-end {inference['images_per_second']:.1f} img/s "
            f"({inference['ms_per_image']:.2f} ms/img)  |  "
            f"model-only {inference['model_images_per_second']:.1f} img/s "
            f"({inference['model_ms_per_image']:.2f} ms/img)"
        )
    print(f"{'=' * 96}\n")


def _load_previous_results(model_name: str) -> dict:
    """
    Read this experiment's existing results file, if any.

    --eval-only does not train, so figures that only a training run can know
    (training timing, how many parameters were actually unfrozen) would
    otherwise be nulled out on every re-run.
    """
    path = config.RESULTS_DIR / f"{model_name}_results.json"
    if not path.exists():
        return {}

    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def save_results(results: dict, model_name: str) -> Path:
    """Save results to a JSON file."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / f"{model_name}_results.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"[evaluate] Results saved -> {path}")
    return path


# =============================================================================
# Main evaluation pipeline
# =============================================================================
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    model_name: str,
    thresholds: np.ndarray | None = None,
    training_timing: dict | None = None,
    trainable_params: int | None = None,
) -> dict:
    """
    Full evaluation pipeline: inference -> thresholding -> metrics -> save.

    Returns the results dict.
    """
    device = next(model.parameters()).device

    # Anything that describes how the checkpoint was *trained* is unknowable on
    # an --eval-only run: the current config holds whatever the CLI happened to
    # pass this time, which is not what produced these weights.
    trained_this_run = training_timing is not None
    previous = _load_previous_results(model_name)

    training_fields = {
        "tuning_mode": config.TUNING_MODE,
        "loss": config.LOSS_NAME,
        "learning_rate": config.LEARNING_RATE,
        "epochs_configured": config.NUM_EPOCHS,
        "subset_size": config.SUBSET_SIZE or 0,
        "checkpoint_metric": config.CHECKPOINT_METRIC,
        "seed": config.SEED,
        "trainable_params": trainable_params,
    }

    carried_over = False
    if not trained_this_run:
        previous_run = previous.get("run") or {}
        recovered = {key: previous_run[key] for key in training_fields if key in previous_run}
        training_timing = (previous.get("timing") or {}).get("training")
        carried_over = bool(recovered) or training_timing is not None

        if recovered:
            training_fields.update(recovered)
            print(
                "[evaluate] Reusing training configuration and timing from the "
                "existing results file"
            )
        else:
            # Nothing to recover: report unknown rather than the current config
            training_fields = dict.fromkeys(training_fields)
            print(
                "[evaluate] WARNING: no previous results to recover the training "
                "configuration from; it will be recorded as unknown."
            )

    trainable_params = training_fields["trainable_params"]

    print(f"[evaluate] Running inference on the test set ({len(test_loader.dataset)} samples) ...")
    labels, probs, inference_timing = collect_predictions(model, test_loader, device)
    preds = apply_thresholds(probs, thresholds)

    print("[evaluate] Computing calibrated metrics ...")
    results = compute_metrics(labels, probs, preds, thresholds=thresholds)

    # Conditions the numbers above were produced under. total_params alone is
    # misleading: a head_only probe trains 25k of EfficientNet-B4's 17.6M.
    total_params = sum(p.numel() for p in model.parameters())
    results["run"] = {
        # How the checkpoint was trained (carried over on --eval-only)
        **training_fields,
        "trainable_fraction": (
            round(trainable_params / total_params, 6) if trainable_params else None
        ),
        "total_params": total_params,
        # Conditions of *this* evaluation, always current
        "threshold_metric": config.THRESHOLD_METRIC,
        "batch_size": config.BATCH_SIZE,
        "num_workers": config.NUM_WORKERS,
        "device": device.type,
        "amp": device.type == "cuda",
        "trained_this_run": trained_this_run,
    }

    results["timing"] = {
        "training": training_timing,
        # True when the training block came from an earlier run, not this one
        "training_timing_carried_over": carried_over,
        "inference": inference_timing,
    }

    print_results(results, model_name)
    save_results(results, model_name)

    return results
