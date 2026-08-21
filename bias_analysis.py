"""
Post-hoc subgroup analysis over saved predictions. Nothing here runs a model.

Re-attaches the patient metadata behind each saved test prediction and asks two
questions separately: does the model *rank* worse for a group (stratified
AUROC), and does the group get *missed* more at the deployed threshold
(stratified FNR, plus the share of abnormal studies where no class fires)? They
can disagree, which is the point — equal AUROC with unequal FNR puts the
disparity in the operating point rather than the features.

Thresholds stay global, because "what does the shipped threshold do to each
group?" is the question about a deployable system. Every gap is reported
against a reference group with a patient-level bootstrap interval on the
*difference*, both levels scored inside the same resample.

Framing follows Seyyed-Kalantari et al. (2021), restricted to the two
attributes CXR14 carries.

Usage:
  python bias_analysis.py --experiment s2_swint_full --model swin_t
  python bias_analysis.py --all
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from evaluate import _percentile_ci, _rows_by_group, apply_thresholds, load_predictions
from metrics import macro_average, per_class_auroc


# Upper bound inclusive. CXR14 carries impossible ages (the max is 414), so
# anything past the last band is dropped rather than pooled into it.
AGE_BANDS = ((0, 19), (20, 39), (40, 59), (60, 79), (80, 100))

# The largest level on each axis, so the reference is the best-measured one.
REFERENCE_LEVEL = {"sex": "M", "age": "40-59"}

# Positives a class needs *within a subgroup* to join its macro. Without a
# floor the long tail invents disparities: Hernia's single positive among the
# 6,908 images aged 20-39 moves that band's macro by five points.
MIN_CLASS_POSITIVES = 10

METRIC_KEYS = ("auroc", "fnr", "underdiagnosis")


# --- Metadata -----------------------------------------------------------------
def test_metadata() -> pd.DataFrame:
    """
    The official test split in loader order, straight from the CSV.

    Not dataset.load_metadata(), which resolves every image path to check the
    drive. Row order is the CSV's own, filtered by test_list.txt — the order
    the unshuffled loader walked — and align_metadata() verifies that.
    """
    frame = pd.read_csv(config.DATA_ENTRY_CSV)
    test_files = set(Path(config.TEST_LIST).read_text().strip().splitlines())
    return frame[frame["Image Index"].isin(test_files)].reset_index(drop=True)


def align_metadata(meta: pd.DataFrame, groups: np.ndarray | None) -> pd.DataFrame:
    """
    Check the metadata rows line up with the saved predictions.

    The join is positional, so a mismatch would pair each score with some other
    patient's demographics and produce confident, meaningless disparities. The
    saved patient IDs make that checkable.
    """
    if groups is None:
        raise ValueError(
            "Saved predictions carry no patient IDs, so metadata cannot be "
            "aligned to them. Re-run evaluation to write them."
        )
    saved = np.asarray(groups).astype(str)
    expected = meta[config.PATIENT_ID_COLUMN].astype(str).to_numpy()
    if not np.array_equal(saved, expected):
        raise ValueError(
            f"Saved predictions ({saved.size} rows) do not line up with the "
            f"official test split ({expected.size} rows) read from "
            f"{config.DATA_ENTRY_CSV.name}. The predictions were written over a "
            "different set of images."
        )
    return meta


def subgroup_masks(meta: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """Boolean row masks per level, for each protected attribute."""
    age = pd.to_numeric(meta["Patient Age"], errors="coerce").to_numpy()
    sex = meta["Patient Gender"].to_numpy()

    return {
        "sex": {"M": sex == "M", "F": sex == "F"},
        "age": {
            f"{low}-{high}": (age >= low) & (age <= high) for low, high in AGE_BANDS
        },
    }


# --- Subgroup metrics ---------------------------------------------------------
def shared_classes(
    labels: np.ndarray,
    level: np.ndarray,
    reference: np.ndarray,
    minimum: int = MIN_CLASS_POSITIVES,
) -> np.ndarray:
    """
    Classes well enough attested in *both* groups to be averaged over.

    A macro over whichever classes each group happens to score is not a
    comparison; the two averages cover different diseases. Per-pair rather than
    one global set, so a band thin on Hernia does not drop it everywhere.
    """
    in_level = labels[level].sum(axis=0)
    in_reference = labels[reference].sum(axis=0)
    return np.flatnonzero((in_level >= minimum) & (in_reference >= minimum))


def per_class_rates(
    labels: np.ndarray, probs: np.ndarray, preds: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Per-class AUROC and FNR for one subgroup, plus its underdiagnosis rate.

    Per-class so a macro over any subset is a mean of these rather than another
    pass over the rows, which is what makes the bootstrap affordable when every
    level averages over a different subset.

    'underdiagnosis' is the Seyyed-Kalantari framing read off all 14 heads: a
    study carrying a finding that fires no class at all was called healthy.
    """
    if labels.shape[0] == 0:
        empty = np.full(labels.shape[1], np.nan)
        return empty, empty, float("nan")

    positives = labels.sum(axis=0)
    true_positive = np.logical_and(preds, labels).sum(axis=0)
    fnr = np.where(
        positives > 0, 1.0 - true_positive / np.maximum(positives, 1), np.nan
    )

    abnormal = labels.sum(axis=1) > 0
    underdiagnosis = (
        float((preds[abnormal].sum(axis=1) == 0).mean()) if abnormal.any() else float("nan")
    )
    return per_class_auroc(labels, probs), fnr, underdiagnosis


def _macro(values: np.ndarray, classes: np.ndarray) -> float:
    """Mean over the scorable members of `classes`; NaN when there are none."""
    if classes.size == 0:
        return float("nan")
    averaged = macro_average(values[classes])
    return float("nan") if averaged is None else float(averaged)


def subgroup_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float]:
    """Ranking and miss rates for one subgroup, macro-averaged over `classes`."""
    auroc, fnr, underdiagnosis = per_class_rates(labels, probs, preds)
    return {
        "auroc": _macro(auroc, classes),
        "fnr": _macro(fnr, classes),
        "underdiagnosis": underdiagnosis,
    }


def bootstrap_gaps(
    labels: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    axes: dict[str, dict[str, np.ndarray]],
    class_sets: dict[str, dict[str, np.ndarray]],
    groups: np.ndarray,
    num_samples: int | None = None,
) -> dict[str, dict[str, dict[str, list[float] | None]]]:
    """
    Patient-level bootstrap intervals on each level's gap to its reference.

    Both sides are scored inside the same resample, over the same class set,
    and subtracted there, so the interval covers the difference itself. Class
    sets are fixed from the observed data rather than refit per draw — a
    resample that lost a class would silently change the estimand.
    """
    num_samples = int(config.BOOTSTRAP_SAMPLES if num_samples is None else num_samples)
    rows_by_group = _rows_by_group(groups)
    rng = np.random.default_rng(config.SEED)

    draws: dict[str, dict[str, dict[str, list[float]]]] = {
        axis: {level: {key: [] for key in METRIC_KEYS} for level in levels}
        for axis, levels in axes.items()
    }

    for _ in range(num_samples):
        picked = rng.integers(0, len(rows_by_group), len(rows_by_group))
        index = np.concatenate([rows_by_group[group] for group in picked])

        for axis, levels in axes.items():
            rates = {}
            for level, mask in levels.items():
                rows = index[mask[index]]
                rates[level] = per_class_rates(labels[rows], probs[rows], preds[rows])

            # Each level uses its own class set, so the reference is averaged
            # several ways from the single pass above.
            ref_auroc, ref_fnr, ref_underdiagnosis = rates[REFERENCE_LEVEL[axis]]
            for level in levels:
                classes = class_sets[axis][level]
                auroc, fnr, underdiagnosis = rates[level]
                draw = {
                    "auroc": _macro(auroc, classes) - _macro(ref_auroc, classes),
                    "fnr": _macro(fnr, classes) - _macro(ref_fnr, classes),
                    "underdiagnosis": underdiagnosis - ref_underdiagnosis,
                }
                for key in METRIC_KEYS:
                    draws[axis][level][key].append(draw[key])

    return {
        axis: {
            level: {key: _percentile_ci(values[key]) for key in METRIC_KEYS}
            for level, values in levels.items()
        }
        for axis, levels in draws.items()
    }


# --- One model, end to end ----------------------------------------------------
def analyze_model(
    model_name: str,
    num_samples: int | None = None,
    minimum: int = MIN_CLASS_POSITIVES,
) -> dict:
    """Score every subgroup for one saved run and write the report."""
    test = load_predictions(model_name, "test")
    labels, probs = test["labels"], test["probs"]

    saved_classes = list(test["class_names"])
    if saved_classes != list(config.CLASS_NAMES):
        raise ValueError(
            f"Saved predictions for {model_name} cover {saved_classes}, but the "
            f"current config declares {list(config.CLASS_NAMES)}."
        )

    meta = align_metadata(test_metadata(), test.get("groups"))
    axes = subgroup_masks(meta)

    threshold_path = config.RESULTS_DIR / f"{model_name}_thresholds.json"
    if not threshold_path.exists():
        raise FileNotFoundError(f"No tuned thresholds at {threshold_path}.")
    tuned = json.loads(threshold_path.read_text())
    thresholds = np.array(
        [tuned["thresholds"][name] for name in config.CLASS_NAMES], dtype=np.float64
    )
    preds = apply_thresholds(probs, thresholds)

    class_sets = {
        axis: {
            level: shared_classes(labels, mask, levels[REFERENCE_LEVEL[axis]], minimum)
            for level, mask in levels.items()
        }
        for axis, levels in axes.items()
    }

    all_classes = np.arange(labels.shape[1])
    overall = subgroup_metrics(labels, probs, preds, all_classes)
    intervals = bootstrap_gaps(
        labels, probs, preds, axes, class_sets, test["groups"], num_samples
    )

    report = {
        "model": model_name,
        "experiment": config.EXPERIMENT_NAME,
        "num_test_samples": int(labels.shape[0]),
        "threshold_metric": tuned.get("threshold_metric"),
        "min_class_positives": minimum,
        "bootstrap": {
            "samples": int(config.BOOTSTRAP_SAMPLES if num_samples is None else num_samples),
            "ci": config.BOOTSTRAP_CI,
            "resampling_unit": "patient",
            "seed": config.SEED,
        },
        "overall": {key: round(overall[key], 4) for key in METRIC_KEYS},
        "unassigned_age_rows": int(
            sum(~np.logical_or.reduce(list(axes["age"].values())))
        ),
        "axes": {},
    }

    for axis, levels in axes.items():
        reference = REFERENCE_LEVEL[axis]
        entries = {}
        for level, mask in levels.items():
            classes = class_sets[axis][level]
            values = subgroup_metrics(labels[mask], probs[mask], preds[mask], classes)
            reference_rows = levels[reference]
            against = subgroup_metrics(
                labels[reference_rows],
                probs[reference_rows],
                preds[reference_rows],
                classes,
            )
            entries[level] = {
                "n": int(mask.sum()),
                "n_patients": int(meta.loc[mask, config.PATIENT_ID_COLUMN].nunique()),
                "positives": int(labels[mask].sum()),
                "classes_scored": int(classes.size),
                "classes_dropped": [
                    name
                    for index, name in enumerate(config.CLASS_NAMES)
                    if index not in set(classes.tolist())
                ],
                **{key: round(values[key], 4) for key in METRIC_KEYS},
                **{
                    f"{key}_gap": round(values[key] - against[key], 4)
                    for key in METRIC_KEYS
                },
                **{f"{key}_gap_ci": intervals[axis][level][key] for key in METRIC_KEYS},
            }
        report["axes"][axis] = {"reference": reference, "levels": entries}

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / f"{model_name}_bias.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print_report(report)
    print(f"\n[bias] Saved -> {path}")
    return report


# --- Reporting ----------------------------------------------------------------
def _interval(bounds: list[float] | None) -> str:
    return f"[{bounds[0]:+.3f}, {bounds[1]:+.3f}]" if bounds else "n/a"


def _resolved(bounds: list[float] | None) -> bool:
    """A gap whose interval excludes zero is one this data can resolve."""
    return bool(bounds) and (bounds[0] > 0 or bounds[1] < 0)


def print_report(report: dict) -> None:
    header = (
        f"{'Group':<10} {'images':>7} {'pts':>6} {'cls':>4} {'AUROC':>7} {'gap':>8}"
        f"{'gap CI':>19} {'FNR':>7} {'gap':>8}{'gap CI':>19} "
        f"{'under':>7} {'gap':>8}{'gap CI':>19}  resolved"
    )
    overall = report["overall"]
    print(f"\n{'=' * len(header)}")
    print(
        f"  {report['experiment']}/{report['model']} - subgroup performance at "
        f"frozen thresholds  (overall AUROC {overall['auroc']:.4f}, "
        f"FNR {overall['fnr']:.4f}, underdiagnosis {overall['underdiagnosis']:.4f})"
    )
    print("=" * len(header))

    for axis, block in report["axes"].items():
        print(f"\n  {axis} - reference group '{block['reference']}'")
        print(header)
        print("-" * len(header))
        for level, entry in block["levels"].items():
            marker = ",".join(
                key for key in METRIC_KEYS if _resolved(entry[f"{key}_gap_ci"])
            )
            print(
                f"{level:<10} {entry['n']:>7,} {entry['n_patients']:>6,} "
                f"{entry['classes_scored']:>4} "
                f"{entry['auroc']:>7.4f} {entry['auroc_gap']:>+8.4f}"
                f"{_interval(entry['auroc_gap_ci']):>19} "
                f"{entry['fnr']:>7.4f} {entry['fnr_gap']:>+8.4f}"
                f"{_interval(entry['fnr_gap_ci']):>19} "
                f"{entry['underdiagnosis']:>7.4f} {entry['underdiagnosis_gap']:>+8.4f}"
                f"{_interval(entry['underdiagnosis_gap_ci']):>19}  {marker}"
            )

    print(
        f"\n  Gaps are level minus reference, with {report['bootstrap']['ci']:.0%} "
        f"patient-level bootstrap intervals on the difference "
        f"({report['bootstrap']['samples']} resamples); the last column names the "
        "gaps whose interval excludes zero. Negative AUROC gaps and positive FNR "
        "gaps both mean the group is served worse. 'cls' is how many of the 14 "
        f"classes clear {report['min_class_positives']} positives in both this "
        "group and the reference; AUROC and FNR are macro-averaged over exactly "
        "those, on both sides of the gap."
    )


def compare_experiments(output_root: Path = config.PROJECT_ROOT / "outputs") -> None:
    """
    One row per run: overall performance beside the gaps it leaves behind —
    whether the choices that buy macro AUROC also close the demographic ones.
    """
    report_files = sorted(output_root.glob("*/results/*_bias.json"))
    if not report_files:
        print("[bias] No bias reports found.")
        return

    rows = []
    for path in report_files:
        report = json.loads(path.read_text())
        row = {
            "experiment": path.parent.parent.name,
            "model": report["model"],
            "auroc": report["overall"]["auroc"],
            "fnr": report["overall"]["fnr"],
            "underdiagnosis": report["overall"]["underdiagnosis"],
        }
        for axis, block in report["axes"].items():
            for level, entry in block["levels"].items():
                if level == block["reference"]:
                    continue
                for key in METRIC_KEYS:
                    row[f"{axis}_{level}_{key}_gap"] = entry[f"{key}_gap"]
                    row[f"{axis}_{level}_{key}_resolved"] = _resolved(
                        entry[f"{key}_gap_ci"]
                    )
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    csv_path = output_root / "bias_comparison.csv"
    table.to_csv(csv_path, index=False)

    columns = [
        "experiment",
        "model",
        "auroc",
        "sex_F_auroc_gap",
        "sex_F_fnr_gap",
        "age_0-19_auroc_gap",
        "age_80-100_auroc_gap",
        "age_80-100_fnr_gap",
    ]
    print("\n  Overall performance vs the gaps it leaves (sorted by AUROC)")
    print(table[[c for c in columns if c in table]].to_string(index=False))
    print(f"\n[bias] Saved comparison table to {csv_path}")


# --- Entry point --------------------------------------------------------------
def _discover_models(experiment: str, output_root: Path) -> list[str]:
    results = output_root / experiment / "results"
    return sorted(
        path.name[: -len("_test_predictions.npz")]
        for path in results.glob("*_test_predictions.npz")
    )


def main():
    parser = argparse.ArgumentParser(
        description="Stratify saved test predictions by patient sex and age"
    )
    parser.add_argument("--model", type=str, default=None, help="Model to analyze")
    parser.add_argument(
        "--experiment", type=str, default="full_dataset", help="Experiment folder"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Every model of every experiment under outputs/",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Rebuild the cross-experiment table from existing reports",
    )
    parser.add_argument(
        "--min-class-positives",
        type=int,
        default=MIN_CLASS_POSITIVES,
        help="Positives a class needs in both groups to join a macro average",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    args = parser.parse_args()

    output_root = config.PROJECT_ROOT / "outputs"

    if args.compare_only:
        compare_experiments(output_root)
        return

    if args.all:
        experiments = sorted(path.parent.name for path in output_root.glob("*/results"))
    else:
        experiments = [args.experiment]

    for experiment in experiments:
        config.set_experiment(experiment)
        models = _discover_models(experiment, output_root) if args.all else [args.model]
        for model_name in models:
            if model_name is None:
                parser.error("--model is required unless --all is given")
            try:
                analyze_model(model_name, args.bootstrap_samples, args.min_class_positives)
            except (FileNotFoundError, ValueError) as error:
                print(f"[bias] Skipping {experiment}/{model_name}: {error}")

    compare_experiments(output_root)


if __name__ == "__main__":
    main()
