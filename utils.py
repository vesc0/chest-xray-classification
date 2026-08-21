"""Seeding, run logging, training curves, and cross-run comparison tables."""

import atexit
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")

import config


class _Tee:
    """Write to the terminal and a log file at once."""

    def __init__(self, stream, handle):
        self._stream = stream
        self._handle = handle

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._handle.write(data)
        # Flush per write: a crashed run should still leave a full log.
        self._handle.flush()
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._handle.flush()

    # Passthroughs so this still looks like a real stream.
    def isatty(self) -> bool:
        return self._stream.isatty()

    def fileno(self) -> int:
        return self._stream.fileno()


def start_run_log(tag: str = "run") -> Path:
    """
    Mirror stdout and stderr into outputs/<experiment>/logs/.

    stderr too, so a traceback survives the terminal. Each run gets its own
    timestamped file, so --eval-only never overwrites the training log.
    """
    log_dir = config.OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{tag}.log"

    handle = open(path, "w", encoding="utf-8")
    handle.write(f"# {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    handle.write(f"# command: {' '.join(sys.argv)}\n\n")
    handle.flush()

    sys.stdout = _Tee(sys.stdout, handle)
    sys.stderr = _Tee(sys.stderr, handle)

    def _close():
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        handle.close()

    atexit.register(_close)
    return path


def seed_everything(seed: int = config.SEED) -> None:
    """
    Seed the main-process RNGs and pin cuDNN to deterministic kernels.

    Not sufficient on its own: DataLoader workers are separate processes, which
    seed_worker / make_dataloader_generator cover.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """
    Seed one DataLoader worker (as worker_init_fn).

    PyTorch seeds torch per worker but leaves Python's `random` and NumPy
    alone, and the transforms here draw from both.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_dataloader_generator(seed: int = config.SEED) -> torch.Generator:
    """Build the RNG that drives shuffling and per-worker seeding."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def plot_training_curves(history: dict, model_name: str) -> Path:
    """Plot train/val loss, AUROC and AUPRC curves and save them."""
    
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=4)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", marker="s", markersize=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name} - Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_auroc"], label="Train AUROC", marker="o", markersize=4)
    axes[1].plot(epochs, history["val_auroc"], label="Val AUROC", marker="s", markersize=4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUROC")
    axes[1].set_title(f"{model_name} - AUROC")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["train_auprc"], label="Train AUPRC", marker="o", markersize=4)
    axes[2].plot(epochs, history["val_auprc"], label="Val AUPRC", marker="s", markersize=4)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("AUPRC")
    axes[2].set_title(f"{model_name} - AUPRC")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()

    path = config.RESULTS_DIR / f"{model_name}_training_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")

    plt.close(fig)

    print(f"[utils] Training curves saved -> {path}")

    return path


def compare_models(results_dir: Path | None = None) -> None:
    """Print a side-by-side table of macro metrics for one experiment."""

    results_dir = results_dir or config.RESULTS_DIR

    result_files = sorted(results_dir.glob("*_results.json"))
    if not result_files:
        print("[compare] No result files found.")
        return

    models = {}
    for result_file in result_files:
        name = result_file.stem.replace("_results", "")
        with open(result_file, encoding="utf-8") as handle:
            models[name] = json.load(handle)

    metric_keys = [
        "auroc",
        "auprc",
        "precision",
        "recall",
        "f1",
        "samples_f1",
        "subset_accuracy",
        "hamming_loss",
    ]

    header = f"{'Metric':<22}" + "".join(f"{name:>16}" for name in models)

    print(f"\n{'=' * 70}")
    print("  Model Comparison - Macro Metrics")
    print(f"{'=' * 70}")
    print(header)
    print("-" * len(header))

    for key in metric_keys:
        row = f"{key:<22}"
        for name, result in models.items():
            value = result.get("macro", {}).get(key)
            row += f"{value:>16.4f}" if value is not None else f"{'N/A':>16}"
        print(row)

    print(f"{'=' * 70}\n")


def compare_localization(results_dir: Path | None = None) -> None:
    """
    Side-by-side localization for every model scored in results_dir.

    Deliberately omits IoU and IoBB: both scale with the predicted box, and
    ViT's 14x14 CAM grid gets finer boxes for free. A caveat printed under a
    number does not survive that number being quoted, so the overlap metrics
    stay in the per-model JSONs beside their own settings. The pointing game
    and energy fraction are far less grid-sensitive, and both print next to the
    random baseline they have to beat.
    """
    results_dir = results_dir or config.RESULTS_DIR

    result_files = sorted(results_dir.glob("*_localization_*.json"))
    if not result_files:
        print("[compare] No localization results found.")
        return

    rows = []
    for result_file in result_files:
        stem = result_file.stem
        model_name, _, method_key = stem.partition("_localization_")
        with open(result_file, encoding="utf-8") as handle:
            summary = json.load(handle)

        macro = summary.get("macro", {})
        if not macro:
            continue
        rows.append({
            "model": model_name,
            "method": summary.get("settings", {}).get("method_label", method_key),
            "instances": summary.get("num_instances", 0),
            "pointing_game": macro.get("pointing_game"),
            "random_baseline": macro.get("random_baseline"),
            "energy_fraction": macro.get("energy_fraction"),
            "detected_fraction": macro.get("detected_fraction"),
            "degenerate_fraction": macro.get("degenerate_fraction"),
            "class_agnostic": summary.get("settings", {}).get("class_agnostic", False),
        })

    if not rows:
        print("[compare] No localization summaries could be read.")
        return

    header = (
        f"{'Model':<16}{'Method':<20}{'N':>6}{'Point':>8}{'Rand':>8}"
        f"{'Energy':>8}{'Detect':>8}{'Blank':>8}"
    )
    print(f"\n{'=' * len(header)}")
    print("  Weakly-supervised localization - macro averages across models")
    print(f"{'=' * len(header)}")
    print(header)
    print("-" * len(header))

    def _format(value):
        return f"{value:>8.3f}" if isinstance(value, (int, float)) else f"{'N/A':>8}"

    for row in sorted(rows, key=lambda r: (r["method"], -(r["pointing_game"] or 0))):
        marker = " *" if row["class_agnostic"] else ""
        print(
            f"{row['model']:<16}{row['method'] + marker:<20}{row['instances']:>6}"
            + _format(row["pointing_game"]) + _format(row["random_baseline"])
            + _format(row["energy_fraction"]) + _format(row["detected_fraction"])
            + _format(row["degenerate_fraction"])
        )

    print("-" * len(header))
    print(
        "  Point vs Rand: pointing-game accuracy against a uniform-heatmap baseline.\n"
        "  Blank: share of instances where the explainer produced no signal at all,\n"
        "  counted as misses in Point. IoU/IoBB are omitted here on purpose - they\n"
        "  favour the finer CAM grid; read them per model, in the localization JSONs."
    )
    if any(row["class_agnostic"] for row in rows):
        print(
            "  * class-agnostic: one map per image whatever the annotated class, so\n"
            "    these rows measure where the model attends overall. Not comparable\n"
            "    to the class-discriminative rows as though they measured the same thing."
        )
    print(f"{'=' * len(header)}\n")


def _check_comparable(experiments: dict) -> None:
    """
    Warn when runs are not directly comparable: macro metrics average over
    whatever classes were scored, so different class sets cannot share a table.
    """
    class_sets = {
        name: tuple(result.get("class_names") or ())
        for name, result in experiments.items()
    }
    distinct = {value for value in class_sets.values() if value}

    if len(distinct) > 1:
        print(
            "[compare] WARNING: these runs scored different class sets, so their "
            "macro/micro columns are not directly comparable:"
        )
        for name, classes in class_sets.items():
            print(f"    {name}: {len(classes)} classes")

    if any(not value for value in class_sets.values()):
        print(
            "[compare] WARNING: some results predate class-set recording; "
            "verify they used the same classes before comparing."
        )


def _read_train_sizes(output_root: Path) -> dict[str, int]:
    """Load per-experiment training-set sizes from the split summaries."""
    sizes: dict[str, int] = {}
    for summary_file in output_root.glob("*/results/dataset_summary.json"):
        try:
            with open(summary_file, encoding="utf-8") as handle:
                summary = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        size = summary.get("sizes", {}).get("train")
        if size is not None:
            sizes[summary_file.parent.parent.name] = int(size)
    return sizes


def compare_experiments(output_root: Path = config.PROJECT_ROOT / "outputs") -> None:
    """Compare macro results across every experiment in outputs/."""

    result_files = sorted(output_root.glob("*/results/*_results.json"))
    if not result_files:
        print("[compare] No experiment results found.")
        return

    experiments = {}
    train_sizes = _read_train_sizes(output_root)
    experiment_train_size = {}
    for result_file in result_files:
        exp_name = result_file.parent.parent.name
        model_name = result_file.stem.replace("_results", "")
        display_name = f"{exp_name}/{model_name}"
        with open(result_file, encoding="utf-8") as handle:
            experiments[display_name] = json.load(handle)
        experiment_train_size[display_name] = train_sizes.get(exp_name)

    _check_comparable(experiments)

    metric_keys = [
        "auroc",
        "auprc",
        "precision",
        "recall",
        "f1",
        "samples_f1",
        "subset_accuracy",
        "hamming_loss",
    ]

    col_width = max(len(name) for name in experiments) + 2

    # Create table header
    header = f"{'Metric':<22}" + "".join(f"{name:>{col_width}}" for name in experiments)

    print(f"\n{'=' * (22 + col_width * len(experiments))}")
    print("  Cross-Experiment Comparison - Macro Metrics")
    print(f"{'=' * (22 + col_width * len(experiments))}")
    print(header)
    print("-" * len(header))

    import pandas as pd
    csv_data = []

    # Training-set size gives the metric rows their x-axis.
    size_row_str = f"{'train_images':<22}"
    size_row_dict = {"Metric": "train_images"}
    for display_name in experiments:
        size = experiment_train_size.get(display_name)
        size_row_str += f"{size:>{col_width},}" if size is not None else f"{'N/A':>{col_width}}"
        size_row_dict[display_name] = size
    print(size_row_str)
    csv_data.append(size_row_dict)

    for key in metric_keys:
        row_str = f"{key:<22}"
        row_dict = {"Metric": key}
        for display_name, result in experiments.items():
            value = result.get("macro", {}).get(key)
            row_str += f"{value:>{col_width}.4f}" if value is not None else f"{'N/A':>{col_width}}"
            row_dict[display_name] = round(value, 4) if value is not None else None
        print(row_str)
        csv_data.append(row_dict)

    # Mean epoch time is fairer than total, which early stopping skews.
    timing_keys = [
        ("mean_epoch_seconds", ("training", "mean_epoch_seconds")),
        ("epochs_run", ("training", "epochs_run")),
        ("model_img_per_sec", ("inference", "model_images_per_second")),
    ]
    for label, (section, key) in timing_keys:
        row_str = f"{label:<22}"
        row_dict = {"Metric": label}
        for display_name, result in experiments.items():
            value = (result.get("timing", {}).get(section) or {}).get(key)
            if value is None:
                row_str += f"{'N/A':>{col_width}}"
            elif isinstance(value, int):
                row_str += f"{value:>{col_width},}"
            else:
                row_str += f"{value:>{col_width}.1f}"
            row_dict[display_name] = value
        print(row_str)
        csv_data.append(row_dict)

    print(f"{'=' * (22 + col_width * len(experiments))}\n")

    csv_path = output_root / "experiment_comparison.csv"
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    print(f"[compare] Saved comparison table to {csv_path}\n")
