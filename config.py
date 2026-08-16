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
# Defaults to the Mac mini's local copy; override for any other machine
# with the XRAY_DATASET_ROOT environment variable:
#   export XRAY_DATASET_ROOT=/path/to/archive-chest-xrays-nih
DEFAULT_DATASET_ROOT = os.path.expanduser("~/Desktop/nih-224")
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
# Override per run with `--image-size`; the images must be sourced at that
# resolution too (see the resize notebook in README). densenet121, convnextv2_t
# and swin_t follow it; vit_s_16 follows it by interpolating its position
# embeddings; maxvit_t cannot and raises, because torchvision fixes its
# attention partitions at 224.
IMAGE_SIZE = 224
# ImageNet normalization. Every backbone in SUPPORTED_MODELS expects these
# statistics — including the two timm models, which is why the ViT slot uses
# DeiT III weights rather than the augreg checkpoints that expect mean/std = 0.5.
# A backbone with different statistics would need its own transform, and
# main.py could no longer build the loaders once and share them.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =============================================================================
# Training
# =============================================================================
BATCH_SIZE = 32
NUM_WORKERS = 4 # DataLoader workers
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 1
# A runaway guard, not a budget saver. With NUM_EPOCHS = 20 and cosine decay to
# MIN_LEARNING_RATE, val AUPRC usually keeps improving into the final epochs, so
# this rarely fires — plan for every run to use its full epoch budget, which is
# also what keeps runs comparable to each other.
EARLY_STOPPING_PATIENCE = 5
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
#   vit_s_16     (21.7M)  pure ViT            timm (DeiT III-S weights)
#   convnextv2_t (27.9M)  modern CNN          timm
#   swin_t       (27.5M)  modern ViT          torchvision
#   maxvit_t     (30.4M)  hybrid CNN/ViT      torchvision
#
# swin_v2_t (27.6M) is buildable but not part of the roster: its weights are
# 256-native and it fails to fit at 224. Kept so that result stays reproducible.
SUPPORTED_MODELS = [
    "densenet121",
    "vit_s_16",
    "convnextv2_t",
    "swin_t",
    "swin_v2_t",
    "maxvit_t",
]

# Optional timm tag overriding the checkpoint pinned in models.py, set per run
# with `--weight-tag`. None keeps the roster's ImageNet-1k-only invariant; the
# override exists for the pretraining-data ablation, where ViT-S is rerun with
# `deit3_small_patch16_224.fb_in22k_ft_in1k` against the IN1k run as its
# control. Only the timm-backed models accept one, and models.py rejects any
# tag whose normalization differs from IMAGENET_MEAN/STD above.
WEIGHT_TAG = None

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

# Objective maximized when fitting one decision threshold per class on the
# validation set:
#   "f1"          - harmonic mean of precision and recall
#   "fbeta"       - F-beta; beta > 1 weights recall over precision
#   "youden"      - sensitivity + specificity - 1
#   "sensitivity" - most specific point that still reaches TARGET_SENSITIVITY
#
# f1 is the default because it degrades gracefully across this dataset's
# prevalence range and is what the ChestX-ray14 literature reports, so the
# numbers stay comparable. Youden is prevalence-independent by construction,
# which sounds desirable until you apply it to Hernia at 0.16% prevalence,
# where a 50%/95% sensitivity/specificity point is a PPV of roughly 2%.
# "sensitivity" is the clinically framed alternative: fixing recall puts every
# architecture at the same operating point and makes specificity comparable.
THRESHOLD_METRIC = "f1" # f1, fbeta, youden, sensitivity
THRESHOLD_BETA = 1.0
THRESHOLD_TARGET_SENSITIVITY = 0.90

# Candidate thresholds are the distinct predicted scores themselves — the same
# candidate set behind sklearn's precision_recall_curve — rather than a fixed
# grid. A grid cannot represent an optimum outside its own bounds, and a grid
# whose lower bound sits above every score a model assigns to a rare class
# scores 0 everywhere and silently returns that bound with all-negative
# predictions, which reads as a tuned result and is an artifact.

# Positives a class needs in validation before its threshold is fitted rather
# than left at DEFAULT_THRESHOLD. With VAL_SPLIT = 0.1 the validation supports
# run from ~1,380 (Infiltration) down to ~14 (Hernia); at 50 only Hernia falls
# back, and a threshold fitted on 14 positives is noise that happens to be a
# float. An honest default beats a fitted number that cannot be defended.
THRESHOLD_MIN_SUPPORT = 50

# Expected calibration error binning. "quantile" gives every bin the same
# number of predictions, "uniform" the same width. Uniform is the textbook
# definition and is close to meaningless on the rare classes here: nearly every
# prediction for a 0.16%-prevalence label lands in the first bin, that bin is
# trivially well calibrated, and it carries almost all of the weight — so the
# class scores a near-perfect ECE for knowing only that the disease is rare.
ECE_BINS = 15
ECE_BIN_STRATEGY = "quantile" # quantile, uniform

# Persist per-class probabilities and labels for the validation and test
# splits. Every alternative thresholding scheme is then a post-hoc script over
# these arrays (see threshold_analysis.py) instead of another inference pass
# over five models. Roughly 1 MB per model per split.
SAVE_PREDICTIONS = True

# =============================================================================
# Bootstrap confidence intervals
# =============================================================================
# Resampling the test set puts an uncertainty band on every reported AUROC,
# AUPRC, precision, recall and F1 without retraining anything. With one seed per
# configuration, point estimates alone cannot support a ranking claim: "0.812 vs
# 0.818" is not a result until you know whether the interval around each number
# overlaps. The operating-point metrics need it most — Hernia rests on 86 test
# positives, so its F1 moves substantially between resamples.
#
# Set to False to skip it — the results JSON then carries point estimates only.
# The cost is roughly 70 seconds per model on the full 25,596-image test split
# at the default sample count, against hours of training.
BOOTSTRAP_ENABLED = True

# Resamples. 1000 is the usual floor for stable 95% percentile bounds; below a
# few hundred the interval endpoints themselves become noisy.
BOOTSTRAP_SAMPLES = 1000

# Two-sided interval width, e.g. 0.95 -> 2.5th and 97.5th percentiles.
BOOTSTRAP_CI = 0.95

# Resample whole patients rather than individual images. ChestX-ray14 has
# several studies per patient, and images from one patient are correlated, so
# an image-level bootstrap treats correlated rows as independent and returns an
# interval that is too narrow. Only set this False if a split is known to be
# one image per patient.
BOOTSTRAP_GROUP_BY_PATIENT = True

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

# =============================================================================
# Explanation sanity checks (Adebayo et al., 2018)
# =============================================================================
# Cascading model-parameter randomization: re-explain the same images after
# progressively destroying the network's weights, and check that the
# explanation degrades. A method whose maps survive weight randomization is
# reading the input, not the model — and would still score above the random
# baseline in localization.py, which is exactly why the check is worth running.
SANITY_CHECK_ENABLED = True

# Annotated instances the check runs on. The set is re-explained once per stage
# per method (6 stages x up to 2 methods), so this is the dominant cost;
# the shape of the falloff is unambiguous well before the standard error on any
# single row matters.
SANITY_CHECK_SAMPLES = 64

# Rank correlation against the trained model's own maps, after every parameter
# in the network has been re-drawn, above which the run is flagged. A heuristic
# for reading the table, not a threshold from the paper — Adebayo et al. report
# the falloff and leave the judgement to the reader.
SANITY_CHECK_ALARM_CORRELATION = 0.5

# Batch size for localization scoring, independent of the training batch size.
# Grad-CAM needs gradients, which cost several times more memory than plain
# inference, and MPS degrades sharply past a threshold rather than raising:
# measured stable at 16 and 32 but falling to ~80s/batch and worsening at 64,
# which reads as a hung run. Inheriting BATCH_SIZE would make an ordinary
# `--batch-size 64` silently stall this stage.
LOCALIZATION_BATCH_SIZE = 16
