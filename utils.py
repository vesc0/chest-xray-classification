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

# Use non-interactive backend for saving plots to files
matplotlib.use("Agg")

import config


def seed_everything(seed: int = config.SEED) -> None:
    """
    Seed the main-process RNGs and pin cuDNN to deterministic kernels.

    This alone does not make augmentation reproducible: DataLoader workers are
    separate processes with their own RNG state. get_dataloaders() pairs this
    with seed_worker / make_dataloader_generator to cover them.
    """

    # Python random seed
    random.seed(seed)

    # NumPy random seed
    np.random.seed(seed)

    # PyTorch CPU seed
    torch.manual_seed(seed)

    # PyTorch GPU seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Force deterministic CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """
    Seed one DataLoader worker process (used as worker_init_fn).

    PyTorch gives each worker a distinct torch seed derived from the loader's
    generator, but leaves Python's `random` and NumPy unseeded — and the
    torchvision transforms used here draw from both.
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
    """
    Plot loss, AUROC, and AUPRC curves for train/val and save to disk.
    """
    
    # Create results directory if missing
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Epoch numbers for x-axis
    epochs = range(1, len(history["train_loss"]) + 1)

    # Create 1 row × 3 columns figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ---------------- LOSS ----------------
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=4)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", marker="s", markersize=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name} - Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ---------------- AUROC ----------------
    axes[1].plot(epochs, history["train_auroc"], label="Train AUROC", marker="o", markersize=4)
    axes[1].plot(epochs, history["val_auroc"], label="Val AUROC", marker="s", markersize=4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUROC")
    axes[1].set_title(f"{model_name} - AUROC")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # ---------------- AUPRC ----------------
    axes[2].plot(epochs, history["train_auprc"], label="Train AUPRC", marker="o", markersize=4)
    axes[2].plot(epochs, history["val_auprc"], label="Val AUPRC", marker="s", markersize=4)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("AUPRC")
    axes[2].set_title(f"{model_name} - AUPRC")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Adjust subplot spacing
    fig.tight_layout()

    # Save figure
    path = config.RESULTS_DIR / f"{model_name}_training_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")

    # Free memory
    plt.close(fig)

    print(f"[utils] Training curves saved -> {path}")

    return path


def compare_models(results_dir: Path | None = None) -> None:
    """
    Load all *_results.json files from results_dir and print a side-by-side
    comparison table of macro metrics.
    """

    # Use default results directory if not provided
    results_dir = results_dir or config.RESULTS_DIR

    # Find all result JSON files
    result_files = sorted(results_dir.glob("*_results.json"))
    if not result_files:
        print("[compare] No result files found.")
        return

    # Load all model result files
    models = {}
    for result_file in result_files:
        name = result_file.stem.replace("_results", "")
        with open(result_file, encoding="utf-8") as handle:
            models[name] = json.load(handle)

    # Metrics to display
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

    # Create table header
    header = f"{'Metric':<22}" + "".join(f"{name:>16}" for name in models)

    print(f"\n{'=' * 70}")
    print("  Model Comparison - Macro Metrics")
    print(f"{'=' * 70}")
    print(header)
    print("-" * len(header))

    # Print each metric row
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

    # Find all experiment result files
    result_files = sorted(output_root.glob("*/results/*_results.json"))
    if not result_files:
        print("[compare] No experiment results found.")
        return

    # Load all experiment results
    experiments = {}
    for result_file in result_files:
        # Example: outputs/full_dataset/results/resnet_results.json
        exp_name = result_file.parent.parent.name
        model_name = result_file.stem.replace("_results", "")
        # Combined experiment/model display name
        display_name = f"{exp_name}/{model_name}"
        with open(result_file, encoding="utf-8") as handle:
            experiments[display_name] = json.load(handle)

    # Metrics to compare
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

    # Dynamically adjust column width
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

    # Print metric rows
    for key in metric_keys:
        row_str = f"{key:<22}"
        row_dict = {"Metric": key}
        for display_name, result in experiments.items():
            value = result.get("macro", {}).get(key)
            row_str += f"{value:>{col_width}.4f}" if value is not None else f"{'N/A':>{col_width}}"
            row_dict[display_name] = round(value, 4) if value is not None else None
        print(row_str)
        csv_data.append(row_dict)

    print(f"{'=' * (22 + col_width * len(experiments))}\n")

    # Save to CSV
    csv_path = output_root / "experiment_comparison.csv"
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    print(f"[compare] Saved comparison table to {csv_path}\n")
