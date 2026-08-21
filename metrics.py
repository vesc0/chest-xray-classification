"""
Threshold-free ranking metrics shared by training and evaluation.

Only the primitives both sides need: per-epoch validation AUROC and the AUROC
reported on test have to be the same number computed the same way, or the
training curve does not predict the final result. Stage-specific metrics
(thresholds, Brier, ECE) stay in evaluate.py.

Undefined classes are NaN, not 0.0, so a class nobody could score stays
distinguishable from one scored at chance and never drags a macro down.
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def has_both_labels(labels: np.ndarray) -> bool:
    """AUROC is defined only when both positive and negative labels exist."""
    positives = labels.sum()
    return 0 < positives < len(labels)


def class_auroc(targets: np.ndarray, scores: np.ndarray) -> float:
    """AUROC for one class; NaN when the class has only one outcome present."""
    if not has_both_labels(targets):
        return float("nan")
    return float(roc_auc_score(targets, scores))


def class_auprc(targets: np.ndarray, scores: np.ndarray) -> float:
    """Average precision for one class; NaN when the class has no positives."""
    if targets.sum() <= 0:
        return float("nan")
    return float(average_precision_score(targets, scores))


def per_class_auroc(labels: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """AUROC per class, NaN where undefined."""
    return np.array(
        [class_auroc(labels[:, i], probs[:, i]) for i in range(labels.shape[1])],
        dtype=np.float64,
    )


def per_class_auprc(labels: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Average precision per class, NaN where undefined."""
    return np.array(
        [class_auprc(labels[:, i], probs[:, i]) for i in range(labels.shape[1])],
        dtype=np.float64,
    )


def macro_average(values: np.ndarray) -> float | None:
    """Mean over the classes that were scorable; None when none of them were."""
    values = np.asarray(values, dtype=np.float64)
    defined = values[~np.isnan(values)]
    return float(defined.mean()) if defined.size else None


def epoch_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """
    One epoch's ranking performance, falling back to 0.0 rather than None so it
    can go straight into the history arrays behind plotting and early stopping.
    """
    return {
        "auroc": macro_average(per_class_auroc(labels, probs)) or 0.0,
        "auprc": macro_average(per_class_auprc(labels, probs)) or 0.0,
    }
