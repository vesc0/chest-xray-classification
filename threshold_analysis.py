"""
Post-hoc threshold analysis over saved predictions.

Nothing here runs a model. It reads the val/test probability arrays that
evaluate.py writes and answers the questions that would otherwise cost a full
inference pass over every architecture:

  - What threshold does each objective pick, and how much does that threshold
    move between validation resamples? A threshold fitted on 125 Fibrosis
    positives is a point estimate like any other, and reporting it without an
    interval overstates how determined it is.
  - What do F1, Youden and a fixed-sensitivity operating point actually cost on
    this data, side by side, at the same frozen thresholds?

Thresholds are always fitted on validation and only then applied to test. The
test split is read to score an already-frozen decision rule and for nothing
else.

Usage:
  python threshold_analysis.py --experiment full_dataset --model densenet121
  python threshold_analysis.py --experiment full_dataset --model all \
      --metric f1 youden sensitivity
"""

import argparse
import json
from pathlib import Path

import numpy as np

import config
from evaluate import (
    THRESHOLD_METRICS,
    _percentile_ci,
    _rows_by_group,
    _thresholded_rates,
    apply_thresholds,
    confusion_sweep,
    load_predictions,
    resolve_threshold_metric,
    select_candidate,
    threshold_settings,
    tune_thresholds,
)
from metrics import has_both_labels


# =============================================================================
# Resampling
# =============================================================================
def _resample_indices(
    rng: np.random.Generator,
    num_rows: int,
    rows_by_group: list[np.ndarray] | None,
) -> np.ndarray:
    """One bootstrap draw, over whole patients when the grouping is known."""
    if rows_by_group is None:
        return rng.integers(0, num_rows, num_rows)
    picked = rng.integers(0, len(rows_by_group), len(rows_by_group))
    return np.concatenate([rows_by_group[group] for group in picked])


def bootstrap_thresholds(
    labels: np.ndarray,
    probs: np.ndarray,
    metric_name: str,
    groups: np.ndarray | None = None,
    num_samples: int | None = None,
) -> tuple[list[list[float]], list[list[float]]]:
    """
    Refit every class's threshold on resampled validation data.

    This is the uncertainty the in-run bootstrap cannot see. evaluate.py
    resamples the *test* set with the threshold held fixed, which answers "how
    precisely did we measure this operating point?". Resampling *validation* and
    refitting answers "how precisely did we locate it?" — and on the rare
    classes the second interval is much the wider of the two.

    Draws where a class falls below THRESHOLD_MIN_SUPPORT, or where no candidate
    separates anything, are recorded as NaN so they are dropped from the
    interval rather than being counted as a threshold of zero.
    """
    num_samples = int(config.BOOTSTRAP_SAMPLES if num_samples is None else num_samples)
    num_rows, num_classes = labels.shape
    rng = np.random.default_rng(config.SEED)
    rows_by_group = _rows_by_group(groups) if groups is not None else None

    threshold_draws: list[list[float]] = [[] for _ in range(num_classes)]
    objective_draws: list[list[float]] = [[] for _ in range(num_classes)]

    for _ in range(num_samples):
        index = _resample_indices(rng, num_rows, rows_by_group)
        sample_labels = labels[index]
        sample_probs = probs[index]

        for class_idx in range(num_classes):
            y_true = sample_labels[:, class_idx]
            y_prob = sample_probs[:, class_idx]

            if int(y_true.sum()) < config.THRESHOLD_MIN_SUPPORT or not has_both_labels(y_true):
                threshold_draws[class_idx].append(float("nan"))
                objective_draws[class_idx].append(float("nan"))
                continue

            candidates, *counts = confusion_sweep(y_true, y_prob)
            best = select_candidate(*counts, metric_name=metric_name)

            if not np.isfinite(best["objective"]) or best["objective"] <= 0.0:
                threshold_draws[class_idx].append(float("nan"))
                objective_draws[class_idx].append(float("nan"))
                continue

            threshold_draws[class_idx].append(float(candidates[best["index"]]))
            objective_draws[class_idx].append(float(best["objective"]))

    return threshold_draws, objective_draws


def bootstrap_test_rates(
    labels: np.ndarray,
    preds: np.ndarray,
    groups: np.ndarray | None = None,
    num_samples: int | None = None,
) -> dict[str, list[list[float]]]:
    """
    Precision/recall/F1 draws on the test set at a frozen operating point.

    Deliberately narrower than evaluate.bootstrap_cis: the ranking metrics are
    already intervalled in the results file, and recomputing 28,000 AUROCs to
    reach the same answer would dominate the runtime of this script.
    """
    num_samples = int(config.BOOTSTRAP_SAMPLES if num_samples is None else num_samples)
    num_rows, num_classes = labels.shape
    rng = np.random.default_rng(config.SEED)
    rows_by_group = _rows_by_group(groups) if groups is not None else None

    names = ("precision", "recall", "f1")
    draws: dict[str, list[list[float]]] = {
        name: [[] for _ in range(num_classes)] for name in names
    }

    for _ in range(num_samples):
        index = _resample_indices(rng, num_rows, rows_by_group)
        rates = dict(zip(names, _thresholded_rates(labels[index], preds[index])))
        for name, values in rates.items():
            for class_idx in range(num_classes):
                draws[name][class_idx].append(float(values[class_idx]))

    return draws


# =============================================================================
# One thresholding scheme, end to end
# =============================================================================
def analyze_scheme(
    val: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    metric_name: str,
    num_samples: int | None = None,
) -> dict:
    """Fit thresholds under one objective and score them on the test split."""
    config.THRESHOLD_METRIC = metric_name
    metric_name = resolve_threshold_metric()

    val_labels, val_probs = val["labels"], val["probs"]
    test_labels, test_probs = test["labels"], test["probs"]

    thresholds, summary = tune_thresholds(val_labels, val_probs)
    preds = apply_thresholds(test_probs, thresholds)
    precision, recall, f1 = _thresholded_rates(test_labels, preds)

    threshold_draws, objective_draws = bootstrap_thresholds(
        val_labels, val_probs, metric_name, groups=val.get("groups"), num_samples=num_samples
    )
    test_draws = bootstrap_test_rates(
        test_labels, preds, groups=test.get("groups"), num_samples=num_samples
    )

    per_class = {}
    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        entry = dict(summary[class_name])
        entry.update(
            {
                "threshold_ci": _percentile_ci(threshold_draws[class_idx]),
                "val_objective_ci": _percentile_ci(objective_draws[class_idx]),
                "test_support": int(test_labels[:, class_idx].sum()),
                "test_precision": round(float(precision[class_idx]), 4),
                "test_recall": round(float(recall[class_idx]), 4),
                "test_f1": round(float(f1[class_idx]), 4),
                "test_precision_ci": _percentile_ci(test_draws["precision"][class_idx]),
                "test_recall_ci": _percentile_ci(test_draws["recall"][class_idx]),
                "test_f1_ci": _percentile_ci(test_draws["f1"][class_idx]),
            }
        )
        per_class[class_name] = entry

    return {
        "settings": threshold_settings(),
        "per_class": per_class,
        "macro": {
            "test_precision": round(float(precision.mean()), 4),
            "test_recall": round(float(recall.mean()), 4),
            "test_f1": round(float(f1.mean()), 4),
        },
    }


# =============================================================================
# Reporting
# =============================================================================
def _interval(bounds: list[float] | None, width: int = 16) -> str:
    text = f"[{bounds[0]:.3f}, {bounds[1]:.3f}]" if bounds else ""
    return f"{text:>{width}}"


def print_scheme(model_name: str, metric_name: str, scheme: dict, unit: str) -> None:
    """One table per objective: where the threshold sits and what it buys."""
    settings = scheme["settings"]
    detail = ""
    if metric_name == "fbeta":
        detail = f" (beta={settings['beta']})"
    if metric_name == "sensitivity":
        detail = f" (target sensitivity={settings['target_sensitivity']})"

    header = (
        f"{'Class':<22} {'Thr':>7}{'threshold CI':>16} "
        f"{'val obj':>8} {'test P':>8} {'test R':>8} {'test F1':>8}"
        f"{'test F1 CI':>16} {'val pos':>8} {'test pos':>9}  status"
    )
    print(f"\n{'=' * len(header)}")
    print(f"  {model_name} - thresholds tuned on validation by '{metric_name}'{detail}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for class_name, entry in scheme["per_class"].items():
        objective = entry.get("objective")
        print(
            f"{class_name:<22} {entry['threshold']:>7.4f}"
            f"{_interval(entry['threshold_ci'])} "
            f"{(f'{objective:.4f}' if objective is not None else 'N/A'):>8} "
            f"{entry['test_precision']:>8.4f} {entry['test_recall']:>8.4f} "
            f"{entry['test_f1']:>8.4f}{_interval(entry['test_f1_ci'])} "
            f"{entry['support']:>8d} {entry['test_support']:>9d}  {entry['status']}"
        )

    macro = scheme["macro"]
    print("-" * len(header))
    print(
        f"{'Macro average':<22} {'-':>7}{'':>16} {'-':>8} "
        f"{macro['test_precision']:>8.4f} {macro['test_recall']:>8.4f} "
        f"{macro['test_f1']:>8.4f}"
    )
    print(
        f"\n  Threshold CIs come from refitting on {config.BOOTSTRAP_SAMPLES} "
        f"{unit}-level resamples of validation; test F1 CIs hold the threshold "
        f"fixed and resample test. A wide threshold CI on a narrow test CI means "
        f"the operating point is poorly located, not precisely measured."
    )


def print_comparison(schemes: dict[str, dict]) -> None:
    """Macro test performance across objectives, so the trade-off is visible."""
    if len(schemes) < 2:
        return

    title = "  Objectives compared at their own frozen thresholds"
    header = f"{'Objective':<16} {'macro P':>9} {'macro R':>9} {'macro F1':>9}"
    rule = "=" * max(len(header), len(title))
    print(f"\n{rule}")
    print(title)
    print(rule)
    print(header)
    print("-" * len(header))
    for metric_name, scheme in schemes.items():
        macro = scheme["macro"]
        print(
            f"{metric_name:<16} {macro['test_precision']:>9.4f} "
            f"{macro['test_recall']:>9.4f} {macro['test_f1']:>9.4f}"
        )
    print("-" * len(header))


# =============================================================================
# Entry point
# =============================================================================
def analyze_model(model_name: str, metrics: list[str], num_samples: int | None) -> Path:
    """Run every requested objective for one model and save the report."""
    val = load_predictions(model_name, "val")
    test = load_predictions(model_name, "test")

    saved_classes = list(val["class_names"])
    if saved_classes != list(config.CLASS_NAMES):
        raise ValueError(
            f"Saved predictions for {model_name} cover {saved_classes}, but the "
            f"current config declares {list(config.CLASS_NAMES)}."
        )

    unit = "patient" if val.get("groups") is not None else "image"
    report = {
        "model": model_name,
        "experiment": config.EXPERIMENT_NAME,
        "num_val_samples": int(val["labels"].shape[0]),
        "num_test_samples": int(test["labels"].shape[0]),
        "bootstrap": {
            "samples": int(config.BOOTSTRAP_SAMPLES if num_samples is None else num_samples),
            "ci": config.BOOTSTRAP_CI,
            "resampling_unit": unit,
            "seed": config.SEED,
        },
        "min_support": config.THRESHOLD_MIN_SUPPORT,
        "schemes": {},
    }

    for metric_name in metrics:
        print(f"\n[threshold-analysis] {model_name}: fitting by '{metric_name}' ...")
        scheme = analyze_scheme(val, test, metric_name, num_samples=num_samples)
        report["schemes"][metric_name] = scheme
        print_scheme(model_name, metric_name, scheme, unit)

    print_comparison(report["schemes"])

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / f"{model_name}_threshold_analysis.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\n[threshold-analysis] Saved -> {path}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Compare thresholding schemes over saved predictions, without re-running inference"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[*config.SUPPORTED_MODELS, "all"],
        help="Which model's saved predictions to analyze",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="full_dataset",
        help="Experiment folder under outputs/ holding the saved predictions",
    )
    parser.add_argument(
        "--metric",
        type=str,
        nargs="+",
        default=["f1"],
        choices=list(THRESHOLD_METRICS),
        help="One or more objectives to fit and compare",
    )
    parser.add_argument(
        "--target-sensitivity",
        type=float,
        default=None,
        help="Recall floor for the 'sensitivity' objective (default 0.90)",
    )
    parser.add_argument(
        "--threshold-beta",
        type=float,
        default=None,
        help="Beta for the 'fbeta' objective",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=None,
        help="Override config.THRESHOLD_MIN_SUPPORT",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=None,
        help="Resamples per interval (default config.BOOTSTRAP_SAMPLES)",
    )

    args = parser.parse_args()
    config.set_experiment(args.experiment)

    if args.target_sensitivity is not None:
        config.THRESHOLD_TARGET_SENSITIVITY = args.target_sensitivity
    if args.threshold_beta is not None:
        config.THRESHOLD_BETA = args.threshold_beta
    if args.min_support is not None:
        config.THRESHOLD_MIN_SUPPORT = args.min_support
    if args.bootstrap_samples is not None:
        config.BOOTSTRAP_SAMPLES = args.bootstrap_samples

    model_names = list(config.SWEEP_MODELS) if args.model == "all" else [args.model]

    for model_name in model_names:
        try:
            analyze_model(model_name, args.metric, args.bootstrap_samples)
        except FileNotFoundError as error:
            print(f"[threshold-analysis] Skipping {model_name}: {error}")


if __name__ == "__main__":
    main()
