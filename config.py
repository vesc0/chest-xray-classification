"""
Configuration module

Centralizes all hyperparameters and paths so experiments are easy to reproduce.
"""

import os
from pathlib import Path

# =============================================================================
# Paths
# =============================================================================
# Root of the NIH Chest X-ray dataset on external SSD
DATASET_ROOT = Path("/Volumes/pm961/archive-chest-xrays-nih")
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
# All 15 classes in the NIH dataset (14 pathologies + No Finding)
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
    "No Finding",
]
NUM_CLASSES = len(CLASS_NAMES)

# Subset size for testing (set to 0 or None for full dataset)
SUBSET_SIZE = 1000

# Fraction of train_val set used for validation
VAL_SPLIT = 0.1
PATIENT_ID_COLUMN = "Patient ID"

# Random seed for reproducibility
SEED = 42

# =============================================================================
# Preprocessing & Augmentation
# =============================================================================
# Input resolution for both CNN and ViT
IMAGE_SIZE = 224
# ImageNet normalization (pretrained models expect this)
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

# Model choices: "densenet121" (CNN), "vit_b_16" (ViT), swin_v2_b (ViT)
SUPPORTED_MODELS = ["densenet121", "vit_b_16", "swin_v2_b"]

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
# Number of sample images to generate XAI visualizations for
XAI_NUM_SAMPLES = 10
