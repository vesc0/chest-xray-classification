"""
Inference, per-class threshold calibration, metrics, and bootstrap intervals.

Thresholds are fitted on validation only and frozen before the test set is
read. Every reported metric carries a patient-level bootstrap interval, since
one seed per configuration cannot support a ranking claim on its own.
"""

import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
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


# --- Inference ----------------------------------------------------------------
@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Run inference over a loader, returning (labels, probs, timing).

    Timing separates two figures that are easy to conflate: end-to-end includes
    decode and transforms, so it reflects deployment but is DataLoader-bound for
    small models, while model-only isolates the forward pass and is the fair
    architecture comparison. The first batch is excluded from model-only as
    warm-up, since it carries lazy allocation and kernel selection.
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


# --- Thresholding -------------------------------------------------------------
def apply_thresholds(probs: np.ndarray, thresholds: np.ndarray | float | None) -> np.ndarray:
    """Convert probabilities into binary predictions using per-class thresholds."""
    if thresholds is None:
        thresholds = config.DEFAULT_THRESHOLD
    return (probs >= thresholds).astype(np.int32)


THRESHOLD_METRICS = ("f1", "fbeta", "youden", "sensitivity")


def resolve_threshold_metric() -> str:
    """Validate config.THRESHOLD_METRIC up front and return it normalized."""
    metric_name = str(config.THRESHOLD_METRIC).lower()
    if metric_name not in THRESHOLD_METRICS:
        raise ValueError(
            f"Unsupported threshold metric '{config.THRESHOLD_METRIC}'. "
            f"Use one of: {', '.join(THRESHOLD_METRICS)}."
        )
    return metric_name


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Elementwise ratio, zero wherever the denominator is zero."""
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros(np.broadcast(numerator, denominator).shape, dtype=np.float64),
        where=denominator > 0,
    )


def confusion_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Confusion counts at every distinct predicted score.

    Candidates are the scores the model produced, so every achievable operating
    point is reachable and no unachievable one is. A fixed grid fails both ways:
    it cannot express an optimum outside its bounds, and on a rare class scored
    entirely below its floor every point scores zero, so the search returns the
    floor with all-negative predictions.

    Returns (thresholds, tp, fp, fn, tn) in descending threshold order, matching
    apply_thresholds()'s `probs >= threshold`: index 0 is strictest and the last
    predicts everything positive, so recall is non-decreasing in the index.
    """
    order = np.argsort(-y_prob, kind="mergesort")
    sorted_true = y_true[order].astype(np.int64)
    sorted_prob = y_prob[order]

    # Tied scores have to enter the positive set together, or the counts
    # describe a split no threshold can produce.
    last_of_run = np.r_[np.flatnonzero(np.diff(sorted_prob)), sorted_prob.size - 1]

    true_positive = np.cumsum(sorted_true)[last_of_run]
    false_positive = (last_of_run + 1) - true_positive

    positives = int(sorted_true.sum())
    negatives = sorted_true.size - positives

    return (
        sorted_prob[last_of_run],
        true_positive,
        false_positive,
        positives - true_positive,
        negatives - false_positive,
    )


def _fbeta_curve(precision: np.ndarray, recall: np.ndarray, beta: float) -> np.ndarray:
    """F-beta at every candidate, zero where precision and recall are both zero."""
    beta_squared = beta * beta
    return _safe_divide(
        (1.0 + beta_squared) * precision * recall,
        beta_squared * precision + recall,
    )


def select_candidate(
    true_positive: np.ndarray,
    false_positive: np.ndarray,
    false_negative: np.ndarray,
    true_negative: np.ndarray,
    metric_name: str | None = None,
) -> dict[str, float | int]:
    """
    Pick one candidate from a confusion sweep under the configured objective.

    Returns the chosen index alongside the rates there, so the caller can record
    what the operating point actually costs rather than only its score.
    """
    metric_name = resolve_threshold_metric() if metric_name is None else metric_name

    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    specificity = _safe_divide(true_negative, true_negative + false_positive)

    if metric_name == "sensitivity":
        target = float(config.THRESHOLD_TARGET_SENSITIVITY)
        reaches_target = recall >= target
        # Recall grows with the index, so the first candidate to clear the
        # target is the most specific one that does. The last index has recall
        # 1.0, so the fallback below is defensive only.
        index = int(np.argmax(reaches_target)) if reaches_target.any() else recall.size - 1
        score = float(specificity[index])
    else:
        if metric_name == "youden":
            objective = recall + specificity - 1.0
        else:
            beta = 1.0 if metric_name == "f1" else float(config.THRESHOLD_BETA)
            objective = _fbeta_curve(precision, recall, beta)

        # Ties break toward recall, which grows with the index.
        tied = np.flatnonzero(objective >= objective.max() - 1e-12)
        index = int(tied[-1])
        score = float(objective[index])

    return {
        "index": index,
        "objective": score,
        "precision": float(precision[index]),
        "recall": float(recall[index]),
        "specificity": float(specificity[index]),
    }


def tune_thresholds(
    labels: np.ndarray,
    probs: np.ndarray,
) -> tuple[np.ndarray, dict[str, dict[str, float | int | str | None]]]:
    """
    Fit one threshold per class on the validation set.

    Two situations fall back to config.DEFAULT_THRESHOLD and record why:
    "default" (too few positives, or only one outcome present) and "degenerate"
    (no candidate scored above zero). The second is unreachable for f1 and
    fbeta by construction, but youden and the fixed-sensitivity rule can both
    legitimately reach it on a class the model cannot rank at all.
    """
    metric_name = resolve_threshold_metric()
    thresholds = np.full(labels.shape[1], config.DEFAULT_THRESHOLD, dtype=np.float32)
    summary: dict[str, dict[str, float | int | str | None]] = {}

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
                "reason": (
                    f"support {support} below THRESHOLD_MIN_SUPPORT "
                    f"{config.THRESHOLD_MIN_SUPPORT}"
                    if support < config.THRESHOLD_MIN_SUPPORT
                    else "only one outcome present in validation"
                ),
            }
            continue

        candidates, *counts = confusion_sweep(y_true, y_prob)
        best = select_candidate(*counts, metric_name=metric_name)

        if not np.isfinite(best["objective"]) or best["objective"] <= 0.0:
            summary[class_name] = {
                "threshold": float(config.DEFAULT_THRESHOLD),
                "objective": round(float(best["objective"]), 4),
                "support": support,
                "status": "degenerate",
                "reason": (
                    f"no candidate threshold scored above zero under "
                    f"'{metric_name}'; max predicted score was "
                    f"{float(y_prob.max()):.4g}"
                ),
            }
            continue

        thresholds[class_idx] = candidates[best["index"]]
        summary[class_name] = {
            "threshold": round(float(candidates[best["index"]]), 4),
            "objective": round(float(best["objective"]), 4),
            "precision": round(float(best["precision"]), 4),
            "recall": round(float(best["recall"]), 4),
            "specificity": round(float(best["specificity"]), 4),
            "support": support,
            "num_candidates": int(candidates.size),
            "status": "tuned",
        }

    return thresholds, summary


def threshold_settings() -> dict:
    """
    Every setting that determines tune_thresholds' output, recorded next to the
    thresholds so the file is reproducible from its own contents.
    """
    metric_name = str(config.THRESHOLD_METRIC).lower()
    settings = {
        "threshold_metric": metric_name,
        "default_threshold": config.DEFAULT_THRESHOLD,
        "min_support": config.THRESHOLD_MIN_SUPPORT,
        "candidate_source": "distinct validation scores",
    }
    if metric_name == "fbeta":
        settings["beta"] = config.THRESHOLD_BETA
    if metric_name == "sensitivity":
        settings["target_sensitivity"] = config.THRESHOLD_TARGET_SENSITIVITY
    return settings


def save_thresholds(
    thresholds: np.ndarray,
    tuning_summary: dict[str, dict[str, float | int | str | None]],
    model_name: str,
    num_val_samples: int | None = None,
) -> Path:
    """Persist tuned thresholds for reproducible evaluation and XAI."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / f"{model_name}_thresholds.json"
    payload = {
        **threshold_settings(),
        "num_val_samples": num_val_samples,
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


def save_predictions(
    labels: np.ndarray,
    probs: np.ndarray,
    model_name: str,
    split: str,
    groups: np.ndarray | None = None,
) -> Path | None:
    """
    Persist the raw labels and probabilities behind one split's metrics.

    This is what makes every thresholding decision reversible: re-thresholding,
    subgroup analysis and ensembling all become post-hoc scripts over ~1 MB of
    arrays instead of another inference pass. Patient IDs travel with them so
    post-hoc code can resample by patient the way the in-run bootstrap does.
    """
    if not config.SAVE_PREDICTIONS:
        return None

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / f"{model_name}_{split}_predictions.npz"

    payload = {
        # Multi-hot labels and a float32 forward pass, so these dtypes lose
        # nothing and keep the file small enough to save for every model.
        "labels": labels.astype(np.int8),
        "probs": probs.astype(np.float32),
        "class_names": np.asarray(config.CLASS_NAMES),
    }
    if groups is not None:
        # A pandas column gives dtype=object, which np.load will not read back
        # without allow_pickle. Fixed-width text keeps the safe default, and
        # these are only ever compared for equality.
        groups = np.asarray(groups)
        if groups.dtype == object:
            groups = groups.astype(str)
        payload["groups"] = groups

    np.savez_compressed(path, **payload)
    print(f"[evaluate] {split.capitalize()} predictions saved -> {path}")
    return path


# So the legacy-format notice prints once per process, not once per file.
_LEGACY_GROUPS_REPORTED = False


def load_predictions(model_name: str, split: str) -> dict[str, np.ndarray]:
    """
    Read back what save_predictions() wrote.

    Files predating text patient IDs hold `groups` as an object array, which
    np.load will not touch under allow_pickle=False. Those are re-read
    permissively and converted, so a tree of finished runs stays readable. That
    path is entered only after the safe read fails for that specific reason.
    """
    path = config.RESULTS_DIR / f"{model_name}_{split}_predictions.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"No saved {split} predictions at {path}. Re-run evaluation with "
            "config.SAVE_PREDICTIONS enabled."
        )

    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    except ValueError as error:
        if "allow_pickle" not in str(error):
            raise

    global _LEGACY_GROUPS_REPORTED
    if not _LEGACY_GROUPS_REPORTED:
        # ensemble.py reads every run in outputs/, and one notice says what
        # twenty identical ones would.
        _LEGACY_GROUPS_REPORTED = True
        print(
            f"[evaluate] {path.name} stores patient IDs in the pre-text format; "
            f"converting on read. Later files of this format convert silently."
        )
    with np.load(path, allow_pickle=True) as archive:
        restored = {key: archive[key] for key in archive.files}

    if "groups" in restored and restored["groups"].dtype == object:
        restored["groups"] = restored["groups"].astype(str)
    return restored


def _loader_patient_groups(loader: DataLoader) -> np.ndarray | None:
    """Patient IDs for a loader, or None when they cannot be read off it."""
    try:
        return patient_groups(loader)
    except (ValueError, KeyError, AttributeError):
        return None


def calibrate_thresholds(
    model: nn.Module,
    val_loader: DataLoader,
    model_name: str,
) -> np.ndarray:
    """Tune per-class thresholds on the validation set and save them."""
    device = next(model.parameters()).device
    print(f"[evaluate] Calibrating thresholds on the validation set ({len(val_loader.dataset)} samples) ...")
    labels, probs, _ = collect_predictions(model, val_loader, device)

    save_predictions(labels, probs, model_name, "val", groups=_loader_patient_groups(val_loader))

    thresholds, summary = tune_thresholds(labels, probs)

    fallbacks = [name for name, entry in summary.items() if entry["status"] != "tuned"]
    if fallbacks:
        print(
            f"[evaluate] {len(fallbacks)} class(es) left at the default threshold "
            f"({config.DEFAULT_THRESHOLD}): {', '.join(fallbacks)}"
        )

    save_thresholds(thresholds, summary, model_name, num_val_samples=int(labels.shape[0]))
    return thresholds


def _calibration_bin_edges(probs: np.ndarray, num_bins: int, strategy: str) -> np.ndarray:
    """Bin boundaries for one class's predictions under the chosen strategy."""
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, num_bins + 1)

    if strategy != "quantile":
        raise ValueError(
            f"Unsupported ECE bin strategy '{strategy}'. Use one of: quantile, uniform."
        )

    # Heavy ties collapse neighbouring quantiles onto one value; merge them
    # rather than leaving empty bins.
    edges = np.unique(np.quantile(probs, np.linspace(0.0, 1.0, num_bins + 1)))
    if edges.size < 2:
        # Every prediction is the same number: one bin that contains them all.
        return np.array([edges[0], np.nextafter(edges[0], np.inf)])
    return edges


def _expected_calibration_error(
    labels: np.ndarray,
    probs: np.ndarray,
    num_bins: int | None = None,
    strategy: str | None = None,
) -> float:
    """
    Binary expected calibration error for one class.

    Config is read at call time so a runtime override reaches this. Uniform
    bins are the textbook definition and near-useless on rare classes, where
    almost every prediction lands in a trivially well-calibrated first bin that
    carries almost all the weight; quantile bins spread that mass out.
    """
    if probs.size == 0:
        return 0.0

    num_bins = config.ECE_BINS if num_bins is None else num_bins
    strategy = (config.ECE_BIN_STRATEGY if strategy is None else strategy).lower()
    edges = _calibration_bin_edges(probs, int(num_bins), strategy)

    # Bins are (lower, upper], the first extended down to its own edge. Closing
    # the top bin keeps a p = 1.0 prediction from being dropped entirely.
    assignment = np.clip(np.searchsorted(edges, probs, side="left") - 1, 0, edges.size - 2)

    ece = 0.0
    for bin_index in range(edges.size - 1):
        mask = assignment == bin_index
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = probs[mask].mean()
        accuracy = labels[mask].mean()
        ece += abs(confidence - accuracy) * (count / probs.size)

    return float(ece)


def _normal_vs_abnormal_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    """
    Screening metric derived from the 14 pathology outputs.

    "No Finding" is not a trained class, so a study is abnormal when any of the
    14 is present, aggregated two ways: strongest single finding (max) and
    probabilistic OR (noisy-or). This keeps the normal/abnormal signal without
    letting a 54%-prevalence class into the macro and micro averages.
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


# --- Bootstrap confidence intervals -------------------------------------------
def patient_groups(loader: DataLoader) -> np.ndarray:
    """
    Patient ID per row, aligned with what collect_predictions returns.

    Alignment is positional, so this is valid only for an unshuffled loader; a
    shuffled one would pair each prediction with some other patient's ID and
    yield confident, wrong intervals. It raises rather than guessing.
    """
    sampler = getattr(loader, "sampler", None)
    if not isinstance(sampler, SequentialSampler):
        raise ValueError(
            "Patient grouping needs an unshuffled loader; predictions and "
            f"patient IDs are matched by position. Got sampler {type(sampler).__name__}."
        )

    frame = loader.dataset.df
    if config.PATIENT_ID_COLUMN not in frame.columns:
        raise KeyError(
            f"No '{config.PATIENT_ID_COLUMN}' column to group the bootstrap by."
        )
    return frame[config.PATIENT_ID_COLUMN].to_numpy()


def _percentile_ci(values: list[float]) -> list[float] | None:
    """Two-sided percentile interval, or None if too little of it was scorable."""
    finite = np.asarray([v for v in values if v is not None and np.isfinite(v)])

    # Degenerate resamples are expected on rare classes; an interval built from
    # a handful of points is not worth reporting.
    if finite.size < 0.5 * max(len(values), 1) or finite.size < 2:
        return None

    tail = (1.0 - config.BOOTSTRAP_CI) / 2.0
    lower, upper = np.percentile(finite, [100 * tail, 100 * (1 - tail)])
    return [round(float(lower), 4), round(float(upper), 4)]


def _rows_by_group(groups: np.ndarray) -> list[np.ndarray]:
    """
    Row indices per group, precomputed once.

    Sorting and splitting is O(n log n) against O(groups x rows) for the
    obvious loop — 2.8k patients x 25.6k rows on the test split.
    """
    _, inverse = np.unique(groups, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    boundaries = np.flatnonzero(np.diff(inverse[order])) + 1
    return np.split(order, boundaries)


def _thresholded_rates(
    labels: np.ndarray,
    preds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-class precision, recall and F1 from the confusion counts.

    sklearn's zero_division=0 convention, vectorized over classes so it costs
    nothing inside the bootstrap loop.
    """
    true_positive = np.logical_and(preds, labels).sum(axis=0)
    precision = _safe_divide(true_positive, preds.sum(axis=0))
    recall = _safe_divide(true_positive, labels.sum(axis=0))
    return precision, recall, _safe_divide(2.0 * precision * recall, precision + recall)


def bootstrap_cis(
    labels: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray | None = None,
    groups: np.ndarray | None = None,
) -> dict:
    """
    Percentile confidence intervals for the per-class and macro metrics.

    With `groups`, whole patients are resampled: several correlated studies per
    patient make an image-level bootstrap treat them as independent and return
    an interval that is too narrow.

    `preds` adds precision/recall/F1 intervals, which is where they are needed
    most — Hernia's test F1 rests on 86 positives. Thresholds were frozen on
    validation first, so this is an ordinary sampling interval at a fixed
    operating point; uncertainty in the threshold itself needs the validation
    set resampled, which threshold_analysis.py does.

    Returns a dict with "per_class", "macro", and the settings used.
    """
    num_samples = int(config.BOOTSTRAP_SAMPLES)
    num_rows, num_classes = labels.shape
    rng = np.random.default_rng(config.SEED)

    if groups is not None:
        rows_by_group = _rows_by_group(groups)
        num_groups = len(rows_by_group)
        unit = "patient"
    else:
        rows_by_group = None
        num_groups = num_rows
        unit = "image"

    auroc_draws: list[list[float]] = [[] for _ in range(num_classes)]
    auprc_draws: list[list[float]] = [[] for _ in range(num_classes)]
    macro_auroc_draws: list[float] = []
    macro_auprc_draws: list[float] = []

    point_metrics = ("precision", "recall", "f1")
    point_draws: dict[str, list[list[float]]] = {
        name: [[] for _ in range(num_classes)] for name in point_metrics
    }
    macro_point_draws: dict[str, list[float]] = {name: [] for name in point_metrics}

    for _ in range(num_samples):
        if rows_by_group is None:
            index = rng.integers(0, num_rows, num_rows)
        else:
            picked = rng.integers(0, num_groups, num_groups)
            index = np.concatenate([rows_by_group[g] for g in picked])

        sample_labels = labels[index]
        sample_probs = probs[index]

        sample_auroc, sample_auprc = [], []
        for class_idx in range(num_classes):
            targets = sample_labels[:, class_idx]
            scores = sample_probs[:, class_idx]
            auroc = class_auroc(targets, scores)
            auprc = class_auprc(targets, scores)
            auroc_draws[class_idx].append(auroc)
            auprc_draws[class_idx].append(auprc)
            sample_auroc.append(auroc)
            sample_auprc.append(auprc)

        macro_auroc = macro_average(sample_auroc)
        macro_auprc = macro_average(sample_auprc)
        if macro_auroc is not None:
            macro_auroc_draws.append(macro_auroc)
        if macro_auprc is not None:
            macro_auprc_draws.append(macro_auprc)

        if preds is not None:
            rates = dict(zip(point_metrics, _thresholded_rates(sample_labels, preds[index])))
            for name, values in rates.items():
                for class_idx in range(num_classes):
                    point_draws[name][class_idx].append(float(values[class_idx]))
                # Matches f1_score(average="macro"): every class, zeros too.
                macro_point_draws[name].append(float(values.mean()))

    per_class = {}
    for idx, class_name in enumerate(config.CLASS_NAMES):
        entry = {
            "auroc": _percentile_ci(auroc_draws[idx]),
            "auprc": _percentile_ci(auprc_draws[idx]),
        }
        if preds is not None:
            entry.update(
                {name: _percentile_ci(point_draws[name][idx]) for name in point_metrics}
            )
        per_class[class_name] = entry

    macro = {
        "auroc": _percentile_ci(macro_auroc_draws),
        "auprc": _percentile_ci(macro_auprc_draws),
    }
    if preds is not None:
        macro.update({name: _percentile_ci(macro_point_draws[name]) for name in point_metrics})

    return {
        "settings": {
            "samples": num_samples,
            "ci": config.BOOTSTRAP_CI,
            "resampling_unit": unit,
            "num_units": int(num_groups),
            "seed": config.SEED,
            "includes_thresholded_metrics": preds is not None,
        },
        "per_class": per_class,
        "macro": macro,
    }


def compute_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    thresholds: np.ndarray | float | None = None,
    groups: np.ndarray | None = None,
    threshold_status: dict[str, str] | None = None,
) -> dict:
    """
    Per-class, macro, micro and calibration metrics, with bootstrap intervals.

    `groups` carries patient IDs so correlated studies resample together.
    `threshold_status` carries tune_thresholds' per-class outcome into the
    report, so a class left at the default is not read as a fitted operating
    point that scored badly — in the table those look identical.
    """
    if thresholds is None:
        thresholds = np.full(labels.shape[1], config.DEFAULT_THRESHOLD, dtype=np.float32)
    elif np.isscalar(thresholds):
        thresholds = np.full(labels.shape[1], float(thresholds), dtype=np.float32)
    else:
        thresholds = np.asarray(thresholds, dtype=np.float32)

    results: dict = {
        # So a cross-run comparison can verify the class sets match.
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

    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        targets = labels[:, class_idx]
        scores = probs[:, class_idx]
        predictions = preds[:, class_idx]
        support = int(targets.sum())

        # Same helpers as the per-epoch training metrics, so the validation
        # curve and the number reported here mean the same thing.
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
            "threshold_status": (threshold_status or {}).get(class_name),
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

    # macro_average skips the classes that were not scorable.
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
    # zero_division=1, not 0: a correctly predicted No Finding study leaves
    # precision and recall both 0/0, and scoring it 0.0 punishes the right
    # answer. At 38.5% No Finding a *perfect* model would score 0.615. The
    # price is that an all-negative model collects those rows free, which is
    # what the baseline below is for.
    all_negative_rows = float((labels.sum(axis=1) == 0).mean())
    results["macro"]["samples_f1"] = round(
        float(f1_score(labels, preds, average="samples", zero_division=1)),
        4,
    )
    results["macro"]["all_negative_samples_f1_baseline"] = round(all_negative_rows, 4)
    results["macro"]["subset_accuracy"] = round(float(accuracy_score(labels, preds)), 4)
    results["macro"]["all_negative_subset_accuracy_baseline"] = round(all_negative_rows, 4)
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
    # The binning changes what the number means, so it travels with it.
    results["calibration"]["ece_bins"] = int(config.ECE_BINS)
    results["calibration"]["ece_bin_strategy"] = str(config.ECE_BIN_STRATEGY).lower()

    if config.BOOTSTRAP_ENABLED:
        print(
            f"[evaluate] Bootstrapping {config.BOOTSTRAP_SAMPLES} resamples "
            f"({'patient' if groups is not None else 'image'}-level) ..."
        )
        intervals = bootstrap_cis(labels, probs, preds=preds, groups=groups)
        results["bootstrap"] = intervals["settings"]

        for class_name, class_cis in intervals["per_class"].items():
            for metric_name, bounds in class_cis.items():
                results["per_class"][class_name][f"{metric_name}_ci"] = bounds

        for metric_name, bounds in intervals["macro"].items():
            results["macro"][f"{metric_name}_ci"] = bounds
    else:
        results["bootstrap"] = None

    return results


# --- Output utilities ---------------------------------------------------------
def print_results(results: dict, model_name: str) -> None:
    """Print a formatted summary table of evaluation results."""
    bootstrap = results.get("bootstrap")
    ci_width = 16
    ci_pad = " " * ci_width if bootstrap else ""

    def interval(metrics: dict, key: str) -> str:
        """Render a CI as [lo, hi], padded, or blank when there isn't one."""
        if not bootstrap:
            return ""
        bounds = metrics.get(key)
        text = f"[{bounds[0]:.3f}, {bounds[1]:.3f}]" if bounds else ""
        return f"{text:>{ci_width}}"

    def ci_label(metric_name: str) -> str:
        if not bootstrap:
            return ""
        return f"{metric_name + ' ' + f'{config.BOOTSTRAP_CI:.0%}'.strip() + ' CI':>{ci_width}}"

    header = (
        f"{'Class':<22} {'Thr':>7} {'AUROC':>8}{ci_label('AUROC')} {'AUPRC':>8} "
        f"{'Prec':>7} {'Recall':>7} {'F1':>7}{ci_label('F1')} {'ECE':>7} {'Support':>8}"
    )
    rule = "=" * len(header)

    print(f"\n{rule}")
    print(f"  Evaluation Results - {model_name}")
    print(rule)
    print(header)
    print("-" * len(header))

    # A class left at the default is not a fitted operating point that scored
    # badly, and in the table those two are indistinguishable.
    not_tuned = []
    for class_name, metrics in results["per_class"].items():
        auroc = f"{metrics['auroc']:.4f}" if metrics["auroc"] is not None else "N/A"
        auprc = f"{metrics['auprc']:.4f}" if metrics["auprc"] is not None else "N/A"
        status = metrics.get("threshold_status")
        marker = "" if status in (None, "tuned") else "*"
        if marker:
            not_tuned.append(f"{class_name} ({status})")
        threshold_text = f"{metrics['threshold']:.2f}{marker}"
        print(
            f"{class_name:<22} {threshold_text:>7} "
            f"{auroc:>8}{interval(metrics, 'auroc_ci')} {auprc:>8} "
            f"{metrics['precision']:>7.4f} {metrics['recall']:>7.4f} "
            f"{metrics['f1']:>7.4f}{interval(metrics, 'f1_ci')} "
            f"{metrics['ece']:>7.4f} {metrics['support']:>8d}"
        )

    print("-" * len(header))
    macro = results["macro"]
    micro = results["micro"]
    calibration = results["calibration"]

    macro_auroc = f"{macro['auroc']:.4f}" if macro["auroc"] is not None else "N/A"
    macro_auprc = f"{macro['auprc']:.4f}" if macro["auprc"] is not None else "N/A"
    micro_auprc = f"{micro['auprc']:.4f}" if micro["auprc"] is not None else "N/A"

    print(
        f"{'Macro average':<22} {'-':>7} "
        f"{macro_auroc:>8}{interval(macro, 'auroc_ci')} {macro_auprc:>8} "
        f"{macro['precision']:>7.4f} {macro['recall']:>7.4f} "
        f"{macro['f1']:>7.4f}{interval(macro, 'f1_ci')} "
        f"{calibration['macro_ece']:>7.4f}"
    )
    print(
        f"{'Micro average':<22} {'-':>7} {'-':>8}{ci_pad} {micro_auprc:>8} "
        f"{micro['precision']:>7.4f} {micro['recall']:>7.4f} "
        f"{micro['f1']:>7.4f}{ci_pad}"
    )

    if not_tuned:
        print(
            f"\n  * threshold not fitted, left at the default "
            f"{config.DEFAULT_THRESHOLD}: {', '.join(not_tuned)}. Precision, "
            f"recall and F1 for these classes are not tuned results — AUROC and "
            f"AUPRC are threshold-free and unaffected."
        )

    if bootstrap:
        thresholded = (
            " Precision/recall/F1 intervals hold the calibrated threshold fixed, so "
            "they carry test-set sampling noise only."
            if bootstrap.get("includes_thresholded_metrics")
            else ""
        )
        print(
            f"\n  {config.BOOTSTRAP_CI:.0%} CI from {bootstrap['samples']} bootstrap "
            f"resamples over {bootstrap['num_units']:,} {bootstrap['resampling_unit']}s. "
            f"Overlapping intervals between two models mean the gap is not resolved."
            f"{thresholded}"
        )

    # Next to the score an all-negative model would get: on labels this sparse
    # the bare number reads far better than it is.
    print(
        f"  Samples F1:                      {macro['samples_f1']:.4f}"
        f"   (all-negative baseline {macro['all_negative_samples_f1_baseline']:.4f})"
    )
    print(
        f"  Subset accuracy:                 {macro['subset_accuracy']:.4f}"
        f"   (all-negative baseline {macro['all_negative_subset_accuracy_baseline']:.4f})"
    )
    print(
        f"  Sample accuracy:                 {macro['sample_accuracy']:.4f}"
        f"   (all-negative baseline {macro['all_negative_sample_accuracy_baseline']:.4f})"
    )
    print(f"  Hamming loss:                    {macro['hamming_loss']:.4f}")
    print(f"  Macro Brier score:               {calibration['macro_brier']:.4f}")
    print(
        f"  Macro ECE:                       {calibration['macro_ece']:.4f}"
        f"   ({calibration['ece_bins']} {calibration['ece_bin_strategy']} bins; "
        f"read the per-class column instead — the macro figure is dominated by "
        f"the rare classes, where it means little)"
    )

    # Derived, not part of the 14-class averages above.
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
        print("-" * len(header))

    # A post-hoc run scored saved arrays, not a model, so it has no device or
    # parameter count; printing the block anyway reads as a broken run.
    if run.get("source"):
        print(f"  Source: {run['source']}")
        members = (results.get("ensemble") or {}).get("members") or []
        if members:
            print(
                "  Members: "
                + ", ".join(f"{entry['experiment']}:{entry['model']}" for entry in members)
            )
    elif run or training or inference:
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
    print(f"{rule}\n")


# Distribution names (not import names) of every package that can move a
# reported number: the two weight registries, the metrics, the image decoder,
# and the arithmetic underneath.
_PROVENANCE_PACKAGES = (
    "torch",
    "torchvision",
    "timm",
    # Pins the weight URLs behind densenet121_xrv, so a release that repoints
    # a tag shows up in the results file.
    "torchxrayvision",
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "pillow",
)


def _environment_versions() -> dict[str, str | None]:
    """
    Record the library versions this evaluation ran under.

    A torchvision upgrade can silently change what `.DEFAULT` resolves to, and
    a scikit-learn upgrade can shift a metric's edge cases. The lock file
    records the current environment; this records the one behind these numbers.
    Missing packages report None rather than raising.
    """
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in _PROVENANCE_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except Exception:
            versions[package] = None
    return versions


def _load_previous_results(model_name: str) -> dict:
    """
    Read this experiment's existing results file, if any, so that --eval-only
    does not null out figures only a training run can know.
    """
    path = config.RESULTS_DIR / f"{model_name}_results.json"
    if not path.exists():
        return {}

    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def load_threshold_status(model_name: str) -> dict[str, str]:
    """
    Per-class tuning outcome, read back from the thresholds file rather than
    passed down, so calibrate_thresholds keeps returning a plain array.
    """
    path = config.RESULTS_DIR / f"{model_name}_thresholds.json"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            summary = json.load(handle).get("tuning_summary") or {}
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        class_name: entry.get("status")
        for class_name, entry in summary.items()
        if isinstance(entry, dict)
    }


def save_results(results: dict, model_name: str) -> Path:
    """Save results to a JSON file."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / f"{model_name}_results.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"[evaluate] Results saved -> {path}")
    return path


# --- Main evaluation pipeline -------------------------------------------------
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    model_name: str,
    thresholds: np.ndarray | None = None,
    training_timing: dict | None = None,
    trainable_params: int | None = None,
) -> dict:
    """Inference -> thresholding -> metrics -> save. Returns the results."""
    device = next(model.parameters()).device

    # How the checkpoint was *trained* is unknowable on an --eval-only run:
    # the current config is whatever this invocation passed.
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
        # Resolved rather than echoed from config.WEIGHT_TAG, which is None on
        # an un-overridden run and identifies nothing. None for the torchvision
        # models, whose weights are pinned by the version under `versions`.
        "weight_tag": getattr(model, "weight_tag", None),
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
            # Nothing to recover: unknown beats the current config.
            training_fields = dict.fromkeys(training_fields)
            print(
                "[evaluate] WARNING: no previous results to recover the training "
                "configuration from; it will be recorded as unknown."
            )

    trainable_params = training_fields["trainable_params"]

    print(f"[evaluate] Running inference on the test set ({len(test_loader.dataset)} samples) ...")
    labels, probs, inference_timing = collect_predictions(model, test_loader, device)
    preds = apply_thresholds(probs, thresholds)

    # The same IDs group the bootstrap and travel into the saved predictions,
    # so post-hoc analysis groups the same way.
    test_patients = _loader_patient_groups(test_loader)
    save_predictions(labels, probs, model_name, "test", groups=test_patients)

    # Whole patients, so the interval accounts for correlated studies. Losing
    # the grouping is not worth losing the run over — fall back and say so.
    groups = None
    if config.BOOTSTRAP_ENABLED and config.BOOTSTRAP_GROUP_BY_PATIENT:
        groups = test_patients
        if groups is None:
            print(
                "[evaluate] WARNING: patient IDs unavailable; falling back to an "
                "image-level bootstrap. Intervals will be narrower than they should be."
            )

    print("[evaluate] Computing calibrated metrics ...")
    results = compute_metrics(
        labels,
        probs,
        preds,
        thresholds=thresholds,
        groups=groups,
        threshold_status=load_threshold_status(model_name),
    )

    # total_params alone is misleading: a head_only probe trains 14k of
    # DenseNet-121's 7.0M.
    total_params = sum(p.numel() for p in model.parameters())
    results["run"] = {
        # How the checkpoint was trained; carried over on --eval-only.
        **training_fields,
        "trainable_fraction": (
            round(trainable_params / total_params, 6) if trainable_params else None
        ),
        "total_params": total_params,
        # Conditions of *this* evaluation, always current.
        **threshold_settings(),
        "ece_bins": config.ECE_BINS,
        "ece_bin_strategy": str(config.ECE_BIN_STRATEGY).lower(),
        "batch_size": config.BATCH_SIZE,
        "num_workers": config.NUM_WORKERS,
        "device": device.type,
        "amp": device.type == "cuda",
        "trained_this_run": trained_this_run,
        # Not carried over on --eval-only: these describe the environment that
        # computed these metrics. Training may have run under other versions.
        "versions": _environment_versions(),
    }

    results["timing"] = {
        "training": training_timing,
        "training_timing_carried_over": carried_over,
        "inference": inference_timing,
    }

    print_results(results, model_name)
    save_results(results, model_name)

    return results
