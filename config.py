"""
Configuration module

Centralizes all hyperparameters and paths so experiments are easy to reproduce.
"""

import os
from pathlib import Path

# =============================================================================
# Paths
# =============================================================================
# Root of the NIH Chest X-ray dataset.
# Defaults to the external SSD used for development; override for any other
# machine with the XRAY_DATASET_ROOT environment variable:
#   export XRAY_DATASET_ROOT=/path/to/archive-chest-xrays-nih
DEFAULT_DATASET_ROOT = "/Volumes/PM961/archive-chest-xrays-nih"
DATASET_ROOT = Path(os.environ.get("XRAY_DATASET_ROOT", DEFAULT_DATASET_ROOT))
DATA_ENTRY_CSV = DATASET_ROOT / "Data_Entry_2017.csv"
TRAIN_VAL_LIST = DATASET_ROOT / "train_val_list.txt"
TEST_LIST = DATASET_ROOT / "test_list.txt"

# Image folders follow pattern: images_001/images/, images_002/images/, …
IMAGE_DIRS = [DATASET_ROOT / f"images_{i:03d}" / "images" for i in range(1, 13)]

# Output directories (inside the project)
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_DIR = OUTPUT_DIR / "results"
XAI_DIR = OUTPUT_DIR / "xai"

# Experiment name — set at runtime via set_experiment(); keeps each run isolated
EXPERIMENT_NAME = None

def set_experiment(name: str) -> None:
    """Configure output dirs under outputs/<name>/ for this experiment."""
    global OUTPUT_DIR, CHECKPOINT_DIR, RESULTS_DIR, XAI_DIR, EXPERIMENT_NAME
    EXPERIMENT_NAME = name
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / name
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    RESULTS_DIR = OUTPUT_DIR / "results"
    XAI_DIR = OUTPUT_DIR / "xai"

# =============================================================================
# Dataset
# =============================================================================
# The 14 ChestX-ray14 pathologies. "No Finding" is deliberately NOT a class:
# it is exactly the absence of all 14 (no study carries both), so training it
# would add a 54%-prevalence target to a long-tail loss without adding any
# information, and averaging it into macro/micro metrics would inflate them and
# break comparability with the Wang et al. / CheXNet benchmarks.
# Normal studies are simply an all-zero label vector; normal-vs-abnormal is
# reported as a derived metric in evaluate.py.
CLASS_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]
NUM_CLASSES = len(CLASS_NAMES)

# Number of TRAINING images for quick experiments; 0 or None uses the whole
# training pool. Validation and test are never subset (see dataset.py), so
# results stay comparable across sizes. Opt in per-run with `--subset N`.
SUBSET_SIZE = 0

# Granularity of the nested subset ordering. The training pool is split into
# this many label-matched shards, so any prefix is stratified and smaller
# subsets are strict subsets of larger ones (5k subset of 15k subset of 30k).
SUBSET_SHARDS = 100

# Fraction of CSV rows that must resolve to a file on disk before the
# dataset root is considered valid. Below this threshold, fail loudly
# instead of training on only a small subset of images.
MIN_IMAGE_MATCH_RATE = 0.5

# Fraction of train_val set used for validation
VAL_SPLIT = 0.1
PATIENT_ID_COLUMN = "Patient ID"

# Random seed for reproducibility
SEED = 42

# =============================================================================
# Preprocessing & Augmentation
# =============================================================================
# Input resolution, shared by every architecture so that resolution is a
# controlled variable rather than a per-model choice.
#
# 224 is not free to change. torchvision's MaxViT-T builds its attention
# partition sizes from a declared input size and raises on anything else, so
# 224 is a hard floor and ceiling for that model. In the other direction,
# torchvision's SwinV2 weights were trained at 256 and are therefore being used
# slightly off-resolution here; SwinV2's log-spaced continuous position bias is
# built for that transfer, and the trade is documented in models.py. Raising
# this (e.g. for a resolution study) means re-sourcing both models.
IMAGE_SIZE = 224
# ImageNet normalization. Every backbone in SUPPORTED_MODELS expects these
# statistics — including the two timm models, which is why the ViT slot uses
# DeiT weights rather than the augreg checkpoints that expect mean/std = 0.5.
# A backbone with different statistics would need its own transform, and
# main.py could no longer build the loaders once and share them.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =============================================================================
# Training
# =============================================================================
BATCH_SIZE = 32
NUM_WORKERS = 4 # DataLoader workers
NUM_EPOCHS = 60
LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-5
WARMUP_EPOCHS = 1
EARLY_STOPPING_PATIENCE = 20
CHECKPOINT_METRIC = "val_auprc" # val_loss, val_auroc, val_auprc

# Long-tail aware loss choices: "asymmetric", "weighted_bce", "bce"
LOSS_NAME = "asymmetric"
MAX_POS_WEIGHT = 25.0
ASL_GAMMA_NEG = 4.0
ASL_GAMMA_POS = 1.0
ASL_CLIP = 0.05

# Parameter counts are as built here, with the 14-class head.
#
#   densenet121   (7.0M)  CNN baseline        torchvision
#   vit_s_16     (21.7M)  pure ViT            timm (DeiT-S weights)
#   convnextv2_t (27.9M)  modern CNN          timm
#   swin_v2_t    (27.6M)  modern ViT          torchvision
#   maxvit_t     (30.4M)  hybrid CNN/ViT      torchvision
SUPPORTED_MODELS = ["densenet121", "vit_s_16", "convnextv2_t", "swin_v2_t", "maxvit_t"]

# Fine-tuning strategy:
#   - "full": train all parameters for all epochs
#   - "head_only": freeze backbone and train classifier head only
#   - "partial": train head first, then unfreeze only top backbone params
TUNING_MODE = "full"

# Used when TUNING_MODE="partial"
FREEZE_EPOCHS = 3
PARTIAL_UNFREEZE_FRACTION = 0.3

# Optional per-group learning-rate scaling
BACKBONE_LR_MULTIPLIER = 0.1
HEAD_LR_MULTIPLIER = 1.0

# =============================================================================
# Evaluation & Calibration
# =============================================================================
DEFAULT_THRESHOLD = 0.5
THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.95
THRESHOLD_STEPS = 91
THRESHOLD_METRIC = "f1" # f1, fbeta, youden
THRESHOLD_BETA = 1.0
THRESHOLD_MIN_SUPPORT = 5
ECE_BINS = 15

# =============================================================================
# Explainability
# =============================================================================
# Number of unselected test images to render heatmaps for. 0 disables the step.
# Off by default: the localization stage below produces better figures — chosen
# diagnostically (hit / miss / FN / FP) and drawn against ground-truth boxes.
# Raise this only to inspect the 6 classes that have no boxes and therefore no
# localization figures: Consolidation, Edema, Emphysema, Fibrosis, Hernia,
# Pleural_Thickening. Note the samples are the first N of the unshuffled test
# set, so they are not a representative draw.
XAI_NUM_SAMPLES = 0

# =============================================================================
# Weakly-supervised localization (Wang et al., 2017)
# =============================================================================
# 984 hand-drawn boxes over 880 images, all inside the official test split,
# covering 8 of the 14 pathologies.
BBOX_CSV = DATASET_ROOT / "BBox_List_2017.csv"

# The box CSV abbreviates one class name
BBOX_LABEL_ALIASES = {"Infiltrate": "Infiltration"}

# Heatmap is binarized at this fraction of its own maximum, then the largest
# connected component becomes the predicted box.
LOCALIZATION_CAM_THRESHOLD = 0.5

# Overlap thresholds T(IoU) / T(IoBB) reported, following Wang et al.
LOCALIZATION_THRESHOLDS = [0.1, 0.25, 0.5, 0.75, 0.9]

# Qualitative figures saved per category (TP / FN / FP / best / worst)
LOCALIZATION_NUM_FIGURES = 3

# Batch size for localization scoring, independent of the training batch size.
# Grad-CAM needs gradients, which cost several times more memory than plain
# inference, and MPS degrades sharply past a threshold rather than raising:
# measured stable at 16 and 32 but falling to ~80s/batch and worsening at 64,
# which reads as a hung run. Inheriting BATCH_SIZE would make an ordinary
# `--batch-size 64` silently stall this stage.
LOCALIZATION_BATCH_SIZE = 16
