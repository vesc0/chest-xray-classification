"""
Main entry point for the chest X-ray classification pipeline.

Usage:
  python main.py --model densenet121
  python main.py --model vit_b_16
  python main.py --model both
  python main.py --model densenet121 --eval-only
  python main.py --model both --subset 5000 --experiment my_run
  python main.py --compare-all
"""

import argparse
import sys
import time

import torch

import config
from dataset import get_dataloaders
from evaluate import calibrate_thresholds, evaluate_model
from explainability import generate_explanations
from models import build_model
from train import _get_device, train_model
from utils import compare_experiments, compare_models, plot_training_curves, seed_everything


# =============================================================================
# Single-model pipeline execution
# =============================================================================
def run_pipeline(
    model_name: str,
    train_loader,
    val_loader,
    test_loader,
    *,
    eval_only: bool = False,
) -> dict:
    """
    Execute the full pipeline for a single model.

    Returns the test-set evaluation results.
    """
    print(f"\n{'#' * 70}")
    print(f"  Pipeline: {model_name}")
    print(f"{'#' * 70}\n")

    # Model initialization
    print(f"[main] Step 1/5 - Building model: {model_name} ...")
    device = _get_device()
    model = build_model(model_name, pretrained=True).to(device)

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    ckpt_path = config.CHECKPOINT_DIR / f"{model_name}_best.pt"

    # Training or checkpoint loading
    if eval_only:
        if not ckpt_path.exists():
            print(f"[main] ERROR: Checkpoint not found at {ckpt_path}")
            print("       Run training first (without --eval-only).")
            sys.exit(1)
        print(f"[main] Step 2/5 - Loading checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    else:
        print(f"[main] Step 2/5 - Training {model_name} ...")
        t0 = time.time()
        history = train_model(model, train_loader, val_loader, model_name)
        elapsed = time.time() - t0
        print(f"[main] Training completed in {elapsed / 60:.1f} minutes")
        plot_training_curves(history, model_name)

    # Threshold calibration (important for multi-label classification)
    print(f"[main] Step 3/5 - Calibrating thresholds for {model_name} ...")
    thresholds = calibrate_thresholds(model, val_loader, model_name)

    # Final evaluation on held-out test set
    print(f"[main] Step 4/5 - Evaluating {model_name} on the test set ...")
    results = evaluate_model(model, test_loader, model_name, thresholds=thresholds)

    # Explainability (XAI visual outputs)
    print(f"[main] Step 5/5 - Generating XAI visualizations ...")
    generate_explanations(model, test_loader, model_name, thresholds=thresholds)

    return results


# =============================================================================
# Entry point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Chest X-ray classification with calibrated multi-label evaluation"
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="both",
        choices=["densenet121", "vit_b_16", "both", "ensemble"],
        help="Which model to train/evaluate (default: both)",
    )

    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; load from checkpoint and run calibration/evaluation/XAI only",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override config.NUM_WORKERS for DataLoader",
    )

    # Experiment configuration overrides
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Override config.SUBSET_SIZE (0 = full dataset)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override config.NUM_EPOCHS",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override config.BATCH_SIZE",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override config.LEARNING_RATE",
    )

    # Training strategy controls (fine-tuning behavior)
    parser.add_argument(
        "--tuning-mode",
        type=str,
        default=None,
        choices=["full", "head_only", "partial"],
        help="Fine-tuning strategy: full, head_only, partial",
    )
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=None,
        help="Warmup epochs with frozen backbone when using partial mode",
    )
    parser.add_argument(
        "--partial-fraction",
        type=float,
        default=None,
        help="Fraction of backbone params to unfreeze in partial mode",
    )

    # Loss/optimization configuration
    parser.add_argument(
        "--loss",
        type=str,
        default=None,
        choices=["asymmetric", "weighted_bce", "bce"],
        help="Override the training loss",
    )
    parser.add_argument(
        "--checkpoint-metric",
        type=str,
        default=None,
        choices=["val_loss", "val_auroc", "val_auprc"],
        help="Override the validation metric used for checkpointing",
    )

    # Threshold optimization configuration
    parser.add_argument(
        "--threshold-metric",
        type=str,
        default=None,
        choices=["f1", "fbeta", "youden"],
        help="Override the validation objective used for threshold tuning",
    )
    parser.add_argument(
        "--threshold-beta",
        type=float,
        default=None,
        help="Beta for fbeta threshold tuning",
    )

    # Experiment tracking/comparison
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name for output folder (default: auto-named from subset size)",
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Compare results across all experiments and exit",
    )

    args = parser.parse_args()

    # Experiment shortcuts
    if args.compare_all:
        compare_experiments()
        return

    # Override config dynamically from CLI
    if args.subset is not None:
        config.SUBSET_SIZE = args.subset
    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        config.LEARNING_RATE = args.lr
    if args.tuning_mode is not None:
        config.TUNING_MODE = args.tuning_mode
    if args.freeze_epochs is not None:
        config.FREEZE_EPOCHS = args.freeze_epochs
    if args.partial_fraction is not None:
        config.PARTIAL_UNFREEZE_FRACTION = args.partial_fraction
    if args.loss is not None:
        config.LOSS_NAME = args.loss
    if args.checkpoint_metric is not None:
        config.CHECKPOINT_METRIC = args.checkpoint_metric
    if args.threshold_metric is not None:
        config.THRESHOLD_METRIC = args.threshold_metric
    if args.threshold_beta is not None:
        config.THRESHOLD_BETA = args.threshold_beta

    # Experiment naming/tracking
    if args.experiment:
        exp_name = args.experiment
    elif config.SUBSET_SIZE and config.SUBSET_SIZE > 0:
        exp_name = f"subset_{config.SUBSET_SIZE}"
    else:
        exp_name = "full_dataset"
    config.set_experiment(exp_name)

    if args.num_workers is not None:
        config.NUM_WORKERS = args.num_workers

    seed_everything()

    # Print run configuration (reproducibility snapshot)
    print("[main] Configuration:")
    print(f"  Experiment:         {config.EXPERIMENT_NAME}")
    print(f"  Output dir:         {config.OUTPUT_DIR}")
    print(f"  Dataset root:       {config.DATASET_ROOT}")
    print(f"  Subset size:        {config.SUBSET_SIZE or 'FULL DATASET'}")
    print(f"  Image size:         {config.IMAGE_SIZE}")
    print(f"  Batch size:         {config.BATCH_SIZE}")
    print(f"  Epochs:             {config.NUM_EPOCHS}")
    print(f"  Learning rate:      {config.LEARNING_RATE}")
    print(f"  Tuning mode:        {config.TUNING_MODE}")
    if config.TUNING_MODE == "partial":
        print(f"  Freeze epochs:      {config.FREEZE_EPOCHS}")
        print(f"  Partial fraction:   {config.PARTIAL_UNFREEZE_FRACTION}")
    print(f"  Loss:               {config.LOSS_NAME}")
    print(f"  Checkpoint metric:  {config.CHECKPOINT_METRIC}")
    print(f"  Threshold metric:   {config.THRESHOLD_METRIC}")
    print(f"  Device:             {_get_device()}")
    print(f"  Num workers:        {config.NUM_WORKERS}")

    # Model selection logic
    if args.model == "both":
        model_names = ["densenet121", "vit_b_16"]
    else:
        model_names = [args.model]

    # Load dataset once for efficiency across models
    print("[main] Loading dataset once for all selected models ...")
    train_loader, val_loader, test_loader = get_dataloaders()

    # Standard single/multi-model execution
    all_results = {}
    for model_name in model_names:
        all_results[model_name] = run_pipeline(
            model_name,
            train_loader,
            val_loader,
            test_loader,
            eval_only=args.eval_only,
        )

    if len(model_names) > 1:
        print("\n[main] Comparing models ...")
        compare_models()

    print("\n[main] All done! Check outputs/ for results, thresholds, curves, and XAI visualizations.")


if __name__ == "__main__":
    main()
