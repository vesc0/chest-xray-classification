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


def _has_both_labels(labels: np.ndarray) -> bool:
    """AUROC is defined only when both positive and negative labels exist."""
    positives = labels.sum()
    return 0 < positives < len(labels)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference on an entire loader.

    Returns:
        labels: (N, NUM_CLASSES) ground-truth multi-hot
        probs:  (N, NUM_CLASSES) predicted probabilities
    """
    model.eval()
    all_labels, all_probs = [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].cpu().numpy()

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(images)

        probs = torch.sigmoid(logits).cpu().numpy()
        all_labels.append(labels)
        all_probs.append(probs)

    labels_np = np.concatenate(all_labels)
    probs_np = np.concatenate(all_probs)
    return labels_np, probs_np


def apply_thresholds(probs: np.ndarray, thresholds: np.ndarray | float | None) -> np.ndarray:
    """Convert probabilities into binary predictions using per-class thresholds."""
    if thresholds is None:
        thresholds = config.DEFAULT_THRESHOLD
    return (probs >= thresholds).astype(np.int32)


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

        if support < config.THRESHOLD_MIN_SUPPORT or not _has_both_labels(y_true):
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
    labels, probs = collect_predictions(model, val_loader, device)
    thresholds, summary = tune_thresholds(labels, probs)
    save_thresholds(thresholds, summary, model_name)
    return thresholds


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
        "per_class": {},
        "macro": {},
        "micro": {},
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

    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        targets = labels[:, class_idx]
        scores = probs[:, class_idx]
        predictions = preds[:, class_idx]
        support = int(targets.sum())

        auroc = roc_auc_score(targets, scores) if _has_both_labels(targets) else float("nan")
        auprc = average_precision_score(targets, scores) if support > 0 else float("nan")
        brier = float(np.mean((scores - targets) ** 2))
        ece = _expected_calibration_error(targets, scores)

        if not np.isnan(auroc):
            per_class_auroc.append(auroc)
        if not np.isnan(auprc):
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

    results["macro"]["auroc"] = round(float(np.mean(per_class_auroc)), 4) if per_class_auroc else None
    results["macro"]["auprc"] = round(float(np.mean(per_class_auprc)), 4) if per_class_auprc else None
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

    results["calibration"]["macro_brier"] = round(float(np.mean(per_class_brier)), 4)
    results["calibration"]["macro_ece"] = round(float(np.mean(per_class_ece)), 4)

    return results


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
    print(f"{'=' * 96}\n")


def save_results(results: dict, model_name: str) -> Path:
    """Save results to a JSON file."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / f"{model_name}_results.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"[evaluate] Results saved -> {path}")
    return path


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    model_name: str,
    thresholds: np.ndarray | None = None,
) -> dict:
    """
    Full evaluation pipeline: inference -> thresholding -> metrics -> save.

    Returns the results dict.
    """
    device = next(model.parameters()).device

    print(f"[evaluate] Running inference on the test set ({len(test_loader.dataset)} samples) ...")
    labels, probs = collect_predictions(model, test_loader, device)
    preds = apply_thresholds(probs, thresholds)

    print("[evaluate] Computing calibrated metrics ...")
    results = compute_metrics(labels, probs, preds, thresholds=thresholds)

    print_results(results, model_name)
    save_results(results, model_name)

    return results
