"""
Utility helpers

Includes:
  - Reproducibility seeding
  - Training curve plotting
  - Comparison table generation
"""

import json
import random
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")

import config


def seed_everything(seed: int = config.SEED) -> None:
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_training_curves(history: dict, model_name: str) -> Path:
    """
    Plot loss, AUROC, and AUPRC curves for train/val and save to disk.
    """
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
    """
    Load all *_results.json files from results_dir and print a side-by-side
    comparison table of macro metrics.
    """
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


def compare_experiments(output_root: Path = config.PROJECT_ROOT / "outputs") -> None:
    """
    Compare results across all experiments (subset sizes).
    """
    result_files = sorted(output_root.glob("*/results/*_results.json"))
    if not result_files:
        print("[compare] No experiment results found.")
        return

    experiments = {}
    for result_file in result_files:
        exp_name = result_file.parent.parent.name
        model_name = result_file.stem.replace("_results", "")
        display_name = f"{exp_name}/{model_name}"
        with open(result_file, encoding="utf-8") as handle:
            experiments[display_name] = json.load(handle)

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
    header = f"{'Metric':<22}" + "".join(f"{name:>{col_width}}" for name in experiments)

    print(f"\n{'=' * (22 + col_width * len(experiments))}")
    print("  Cross-Experiment Comparison - Macro Metrics")
    print(f"{'=' * (22 + col_width * len(experiments))}")
    print(header)
    print("-" * len(header))

    for key in metric_keys:
        row = f"{key:<22}"
        for display_name, result in experiments.items():
            value = result.get("macro", {}).get(key)
            row += f"{value:>{col_width}.4f}" if value is not None else f"{'N/A':>{col_width}}"
        print(row)

    print(f"{'=' * (22 + col_width * len(experiments))}\n")
