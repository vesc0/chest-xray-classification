"""
Main entry point for the chest X-ray classification pipeline.

Usage:
  python main.py --model densenet121
  python main.py --model vit_s_16
  python main.py --model all
  python main.py --model densenet121 --eval-only
  python main.py --model all --subset 5000 --experiment my_run
  python main.py --compare-all
"""

import argparse
import sys

import torch

import config
from dataset import get_dataloaders
from device import get_device
from evaluate import THRESHOLD_METRICS, calibrate_thresholds, evaluate_model
from explainability import generate_explanations
from localization import evaluate_localization
from models import build_model
from sanity_checks import run_sanity_checks
from train import train_model
from utils import (
    compare_experiments,
    compare_localization,
    compare_models,
    plot_training_curves,
    seed_everything,
    start_run_log,
)


# --- Single-model pipeline execution ------------------------------------------
def run_pipeline(
    model_name: str,
    train_loader,
    val_loader,
    test_loader,
    *,
    eval_only: bool = False,
) -> dict:
    """Run the full pipeline for one model; returns the test results."""
    print(f"\n{'#' * 70}")
    print(f"  Pipeline: {model_name}")
    print(f"{'#' * 70}\n")

    print(f"[main] Step 1/5 - Building model: {model_name} ...")
    device = get_device()
    model = build_model(model_name, pretrained=True, weight_tag=config.WEIGHT_TAG).to(device)
    if config.WEIGHT_TAG:
        print(f"  Weight tag:           {config.WEIGHT_TAG} (override)")

    # Only meaningful once the tuning mode has been applied, which happens
    # inside train_model — so it is reported after training, not here.
    total_params = sum(param.numel() for param in model.parameters())
    print(f"  Total parameters:     {total_params:,}")

    ckpt_path = config.CHECKPOINT_DIR / f"{model_name}_best.pt"
    training_timing = None
    trainable_params = None

    if eval_only:
        if not ckpt_path.exists():
            print(f"[main] ERROR: Checkpoint not found at {ckpt_path}")
            print("       Run training first (without --eval-only).")
            sys.exit(1)
        print(f"[main] Step 2/5 - Loading checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    else:
        print(f"[main] Step 2/5 - Training {model_name} ...")
        history, training_timing = train_model(model, train_loader, val_loader, model_name)
        # After training, when requires_grad reflects the tuning mode.
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        plot_training_curves(history, model_name)

    print(f"[main] Step 3/5 - Calibrating thresholds for {model_name} ...")
    thresholds = calibrate_thresholds(model, val_loader, model_name)

    print(f"[main] Step 4/5 - Evaluating {model_name} on the test set ...")
    results = evaluate_model(
        model,
        test_loader,
        model_name,
        thresholds=thresholds,
        training_timing=training_timing,
        trainable_params=trainable_params,
    )

    # Unselected samples; off unless config.XAI_NUM_SAMPLES > 0.
    if config.XAI_NUM_SAMPLES > 0:
        print(f"[main] Generating {config.XAI_NUM_SAMPLES} qualitative XAI samples ...")
        generate_explanations(model, test_loader, model_name, thresholds=thresholds)

    print(f"[main] Step 5/5 - Scoring localization against ground-truth boxes ...")
    evaluate_localization(model, model_name, thresholds=thresholds)

    # Whether those localization numbers mean anything: an explanation that
    # survives weight randomization scores above baseline regardless.
    if config.SANITY_CHECK_ENABLED:
        print("[main] Running explanation sanity checks ...")
        run_sanity_checks(model, model_name)

    return results


# --- Entry point --------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Chest X-ray classification with calibrated multi-label evaluation"
    )

    # No default: an implicit "train everything on the full dataset" would be
    # an expensive accident. Required unless --compare-all (enforced below).
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=[*config.SUPPORTED_MODELS, "all"],
        help="Which model to train/evaluate, or 'all' for every supported architecture",
    )

    parser.add_argument(
        "--weight-tag",
        type=str,
        default=None,
        help=(
            "Override the pinned checkpoint, for the pretraining-data ablation. "
            "A timm tag for vit_s_16/convnextv2_t, e.g. "
            "deit3_small_patch16_224.fb_in22k_ft_in1k for ViT-S — rejected if it "
            "expects normalization other than config.IMAGENET_MEAN/STD. A "
            "TorchXRayVision weight name for densenet121_xrv, e.g. "
            "densenet121-res224-pc — rejected if its corpus contains ChestX-ray14"
        ),
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help=(
            "Override config.IMAGE_SIZE. The dataset must already be sourced at "
            "this resolution. maxvit_t is fixed at 224 and rejects anything else"
        ),
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

    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help=(
            "Number of TRAINING images to use (0 = full training pool). "
            "Validation and test are always held fixed"
        ),
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

    parser.add_argument(
        "--threshold-metric",
        type=str,
        default=None,
        choices=list(THRESHOLD_METRICS),
        help=(
            "Validation objective used for threshold tuning. 'sensitivity' "
            "fixes recall at --target-sensitivity and reports the most specific "
            "threshold that reaches it"
        ),
    )
    parser.add_argument(
        "--threshold-beta",
        type=float,
        default=None,
        help="Beta for fbeta threshold tuning",
    )
    parser.add_argument(
        "--target-sensitivity",
        type=float,
        default=None,
        help="Recall floor for --threshold-metric sensitivity (default 0.90)",
    )
    parser.add_argument(
        "--ece-bins",
        type=int,
        default=None,
        help="Override config.ECE_BINS",
    )
    parser.add_argument(
        "--ece-bin-strategy",
        type=str,
        default=None,
        choices=["quantile", "uniform"],
        help="Equal-count (quantile) or equal-width (uniform) calibration bins",
    )
    parser.add_argument(
        "--no-save-predictions",
        action="store_true",
        help=(
            "Skip writing raw val/test probabilities. Saving them is the default: "
            "it makes every alternative thresholding scheme a post-hoc script "
            "instead of another inference pass"
        ),
    )

    parser.add_argument(
        "--xai-samples",
        type=int,
        default=None,
        help=(
            "Render heatmaps for N unselected test images (default 0). "
            "Localization figures are usually the better option; use this to "
            "inspect the 6 classes that have no ground-truth boxes"
        ),
    )

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

    if args.compare_all:
        compare_experiments()
        return

    if args.model is None:
        parser.error("--model is required (or use --compare-all)")

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
    if args.target_sensitivity is not None:
        config.THRESHOLD_TARGET_SENSITIVITY = args.target_sensitivity
    if args.ece_bins is not None:
        config.ECE_BINS = args.ece_bins
    if args.ece_bin_strategy is not None:
        config.ECE_BIN_STRATEGY = args.ece_bin_strategy
    if args.no_save_predictions:
        config.SAVE_PREDICTIONS = False
    if args.xai_samples is not None:
        config.XAI_NUM_SAMPLES = args.xai_samples
    if args.weight_tag is not None:
        config.WEIGHT_TAG = args.weight_tag
    # Must precede get_dataloaders(): the transform reads IMAGE_SIZE.
    if args.image_size is not None:
        config.IMAGE_SIZE = args.image_size

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

    # Mirror everything below into outputs/<experiment>/logs/.
    log_path = start_run_log("eval" if args.eval_only else "train")

    print("[main] Configuration:")
    print(f"  Log file:           {log_path}")
    print(f"  Experiment:         {config.EXPERIMENT_NAME}")
    print(f"  Output dir:         {config.OUTPUT_DIR}")
    print(f"  Dataset root:       {config.DATASET_ROOT}")
    print(f"  Train subset:       {config.SUBSET_SIZE or 'FULL TRAINING POOL'}")
    print(f"  Classes:            {config.NUM_CLASSES} pathologies (No Finding = all-zero)")
    if config.WEIGHT_TAG:
        print(f"  Weight tag:         {config.WEIGHT_TAG} (overrides the pinned checkpoint)")
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
    if config.THRESHOLD_METRIC == "fbeta":
        print(f"  Threshold beta:     {config.THRESHOLD_BETA}")
    if config.THRESHOLD_METRIC == "sensitivity":
        print(f"  Target sensitivity: {config.THRESHOLD_TARGET_SENSITIVITY}")
    print(f"  Threshold support:  >= {config.THRESHOLD_MIN_SUPPORT} val positives to fit")
    print(f"  Save predictions:   {config.SAVE_PREDICTIONS}")
    print(f"  Device:             {get_device()}")
    print(f"  Num workers:        {config.NUM_WORKERS}")

    if args.model == "all":
        model_names = list(config.SWEEP_MODELS)
    else:
        model_names = [args.model]

    print("[main] Loading dataset once for all selected models ...")
    train_loader, val_loader, test_loader = get_dataloaders()

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
        compare_localization()

    print("\n[main] All done! Check outputs/ for results, thresholds, curves, and XAI visualizations.")


if __name__ == "__main__":
    main()
