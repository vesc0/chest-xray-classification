"""Hyperparameters, paths, and feature switches for the whole pipeline."""

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------------

# Override the dataset location with XRAY_DATASET_ROOT.
DEFAULT_DATASET_ROOT = os.path.expanduser("~/Desktop/nih-224")
DATASET_ROOT = Path(os.environ.get("XRAY_DATASET_ROOT", DEFAULT_DATASET_ROOT))
DATA_ENTRY_CSV = DATASET_ROOT / "Data_Entry_2017.csv"
TRAIN_VAL_LIST = DATASET_ROOT / "train_val_list.txt"
TEST_LIST = DATASET_ROOT / "test_list.txt"
IMAGE_DIRS = [DATASET_ROOT / f"images_{i:03d}" / "images" for i in range(1, 13)]

PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_DIR = OUTPUT_DIR / "results"
XAI_DIR = OUTPUT_DIR / "xai"

EXPERIMENT_NAME = None

def set_experiment(name: str) -> None:
    """Redirect all output dirs under outputs/<name>/ for this experiment."""
    global OUTPUT_DIR, CHECKPOINT_DIR, RESULTS_DIR, XAI_DIR, EXPERIMENT_NAME
    EXPERIMENT_NAME = name
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / name
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    RESULTS_DIR = OUTPUT_DIR / "results"
    XAI_DIR = OUTPUT_DIR / "xai"

# --- Dataset -----------------------------------------------------------------

# "No Finding" is deliberately not a class: it is exactly the absence of all 14,
# so normal studies are an all-zero vector and normal-vs-abnormal is derived.
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

# Training images to use; 0 uses the whole pool. Val/test are never subset.
SUBSET_SIZE = 0

# Label-matched shards the training pool is cut into, so any prefix is
# stratified and smaller subsets are strict subsets of larger ones.
SUBSET_SHARDS = 100

# Fraction of CSV rows that must resolve on disk before the root is accepted.
MIN_IMAGE_MATCH_RATE = 0.5

VAL_SPLIT = 0.1
PATIENT_ID_COLUMN = "Patient ID"
SEED = 42

# --- Preprocessing and augmentation ------------------------------------------

# Shared by every architecture so resolution stays a controlled variable.
# maxvit_t cannot follow --image-size: torchvision fixes its partitions at 224.
IMAGE_SIZE = 224

# Every backbone in SUPPORTED_MODELS expects these statistics; models.py
# rejects any --weight-tag that does not.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Training ----------------------------------------------------------------

BATCH_SIZE = 32
NUM_WORKERS = 10
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 1

# A runaway guard, not a budget saver; under cosine decay this rarely fires.
EARLY_STOPPING_PATIENCE = 5
CHECKPOINT_METRIC = "val_auprc" # val_loss, val_auroc, val_auprc

LOSS_NAME = "asymmetric" # asymmetric, weighted_bce, bce
MAX_POS_WEIGHT = 25.0
ASL_GAMMA_NEG = 4.0
ASL_GAMMA_POS = 1.0
ASL_CLIP = 0.05

SUPPORTED_MODELS = [
    "densenet121",
    "vit_s_16",
    "convnextv2_t",
    "swin_t",
    "swin_v2_t",
    "maxvit_t",
    "densenet121_xrv",
]

# What `--model all` expands to. densenet121_xrv is excluded because it is a
# pretraining ablation, not a distinct architecture; run it by name.
SWEEP_MODELS = [name for name in SUPPORTED_MODELS if name != "densenet121_xrv"]

# Optional timm tag overriding the checkpoint pinned in models.py. None keeps
# the roster's ImageNet-1k-only invariant.
WEIGHT_TAG = None

TUNING_MODE = "full" # full, head_only, partial
FREEZE_EPOCHS = 3 # partial only
PARTIAL_UNFREEZE_FRACTION = 0.3 # partial only
BACKBONE_LR_MULTIPLIER = 0.1
HEAD_LR_MULTIPLIER = 1.0

# --- Evaluation and calibration ----------------------------------------------

DEFAULT_THRESHOLD = 0.5

# Objective maximized when fitting one threshold per class on validation:
# f1, fbeta (beta > 1 favours recall), youden, or sensitivity (most specific
# point still reaching THRESHOLD_TARGET_SENSITIVITY). f1 is the default because
# it degrades gracefully across this prevalence range and matches the
# ChestX-ray14 literature; youden is prevalence-independent and unusable on
# Hernia at 0.16% prevalence. Candidates are the predicted scores themselves,
# not a grid, so the optimum cannot fall outside the candidate set.
THRESHOLD_METRIC = "f1"
THRESHOLD_BETA = 1.0
THRESHOLD_TARGET_SENSITIVITY = 0.90

# Validation positives a class needs before its threshold is fitted at all.
# Only Hernia (~14) falls back to DEFAULT_THRESHOLD.
THRESHOLD_MIN_SUPPORT = 50

# "quantile" gives every bin equal count, "uniform" equal width. Uniform is
# near-meaningless on rare classes, where almost every prediction lands in the
# trivially calibrated first bin.
ECE_BINS = 15
ECE_BIN_STRATEGY = "quantile"

# Persist per-class probabilities so re-thresholding needs no inference
# (see threshold_analysis.py). Roughly 1 MB per model per split.
SAVE_PREDICTIONS = True

# --- Bootstrap confidence intervals -------------------------------------------

# One seed per configuration cannot support a ranking claim, so every reported
# metric carries an interval. Costs ~70s per model on the full test split.
BOOTSTRAP_ENABLED = True
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_CI = 0.95

# Resample whole patients: images from one patient are correlated, and an
# image-level bootstrap returns an interval that is too narrow.
BOOTSTRAP_GROUP_BY_PATIENT = True

# --- Explainability -----------------------------------------------------------

# Extra unselected test images to render heatmaps for; 0 disables the step.
# The localization stage produces better figures. Raise this only to inspect
# the 6 classes with no ground-truth boxes.
XAI_NUM_SAMPLES = 0

# --- Weakly-supervised localization (Wang et al., 2017) -----------------------

# 984 hand-drawn boxes over 880 test images, covering 8 of the 14 pathologies.
BBOX_CSV = DATASET_ROOT / "BBox_List_2017.csv"
BBOX_LABEL_ALIASES = {"Infiltrate": "Infiltration"}

# Heatmaps are binarized at this fraction of their own maximum; the largest
# connected component becomes the predicted box.
LOCALIZATION_CAM_THRESHOLD = 0.5
LOCALIZATION_THRESHOLDS = [0.1, 0.25, 0.5, 0.75, 0.9]
LOCALIZATION_NUM_FIGURES = 3 # per category (TP / FN / FP / best / worst)

# --- Explanation sanity checks (Adebayo et al., 2018) -------------------------

# Re-explain the same images with the weights progressively randomized. A
# method whose maps survive that is reading the input, not the model.
SANITY_CHECK_ENABLED = True

# Annotated instances per stage per method (6 stages x up to 2 methods), so
# this dominates the cost; the falloff is unambiguous well before it matters.
SANITY_CHECK_SAMPLES = 64

# Correlation against the trained model's maps, after full randomization, above
# which the run is flagged. A heuristic for reading the table, not from the paper.
SANITY_CHECK_ALARM_CORRELATION = 0.5

# Independent of BATCH_SIZE: Grad-CAM needs gradients, and MPS degrades sharply
# past 32 rather than raising, which reads as a hung run.
LOCALIZATION_BATCH_SIZE = 16
