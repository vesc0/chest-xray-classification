"""
Dataset module

Handles:
  - Parsing the metadata CSV and official train/test splits
  - Multi-label encoding (15 classes)
  - Patient-aware train / validation splitting
  - Group-preserving subset sampling for fast local experiments
  - Radiograph-appropriate augmentation and preprocessing transforms
"""

from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from sklearn.model_selection import GroupShuffleSplit, train_test_split

import config
from utils import make_dataloader_generator, seed_worker


# =============================================================================
# Data indexing utilities
# =============================================================================
def _build_image_index() -> dict[str, Path]:
    """
    Scan all image directories and return {filename: full_path}.

    Raises if no image directory is readable — usually an unmounted drive or an
    unset XRAY_DATASET_ROOT, which would otherwise surface much later as an
    empty DataFrame.
    """
    index: dict[str, Path] = {}
    found_dirs = 0

    for img_dir in config.IMAGE_DIRS:
        if img_dir.exists():
            found_dirs += 1
            for path in img_dir.iterdir():
                if path.suffix == ".png":
                    index[path.name] = path

    if not index:
        raise FileNotFoundError(
            f"No images found under {config.DATASET_ROOT} "
            f"({found_dirs}/{len(config.IMAGE_DIRS)} image directories readable). "
            "Check that the dataset drive is mounted, or point XRAY_DATASET_ROOT "
            "at the dataset root."
        )

    if found_dirs < len(config.IMAGE_DIRS):
        print(
            f"[dataset] Warning: only {found_dirs}/{len(config.IMAGE_DIRS)} "
            "image directories were found."
        )

    return index


def _stack_label_vectors(dataframe: pd.DataFrame) -> np.ndarray:
    """Stack dataframe label vectors into a dense (N, C) float32 array."""
    if dataframe.empty:
        return np.zeros((0, config.NUM_CLASSES), dtype=np.float32)
    return np.stack(dataframe["label_vector"].to_numpy()).astype(np.float32)


def _get_group_column(dataframe: pd.DataFrame) -> str | None:
    """Return the configured patient identifier column if it exists (used for leakage prevention)."""
    if config.PATIENT_ID_COLUMN in dataframe.columns:
        return config.PATIENT_ID_COLUMN
    return None


# =============================================================================
# Metadata loading + label encoding
# =============================================================================
def load_metadata() -> pd.DataFrame:
    """
    Read Data_Entry_2017.csv and add:
      - 'labels': list of finding strings per image
      - 'label_vector': multi-hot numpy array of shape (NUM_CLASSES,)
      - 'filepath': resolved absolute path to the .png file

    Returns a DataFrame ready for dataset construction.
    """
    if not config.DATA_ENTRY_CSV.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found at {config.DATA_ENTRY_CSV}. "
            "Check that the dataset drive is mounted, or point XRAY_DATASET_ROOT "
            "at the dataset root."
        )

    df = pd.read_csv(config.DATA_ENTRY_CSV)

    image_index = _build_image_index()
    df["filepath"] = df["Image Index"].map(image_index)

    # Dataset integrity check: a low match rate means the index and the CSV
    # disagree, so fail rather than silently training on whatever is left.
    missing = int(df["filepath"].isna().sum())
    if missing > 0:
        match_rate = (len(df) - missing) / max(len(df), 1)
        if match_rate < config.MIN_IMAGE_MATCH_RATE:
            raise FileNotFoundError(
                f"Only {len(df) - missing}/{len(df)} images listed in "
                f"{config.DATA_ENTRY_CSV.name} were found under {config.DATASET_ROOT} "
                f"({match_rate:.1%}, minimum {config.MIN_IMAGE_MATCH_RATE:.0%}). "
                "The dataset looks incomplete or the root path is wrong."
            )
        print(f"[dataset] Warning: {missing} images not found on disk - skipping them.")
        df = df.dropna(subset=["filepath"]).reset_index(drop=True)

    # Multi-label parsing (string → list)
    df["labels"] = df["Finding Labels"].str.split("|")
    class_to_idx = {class_name: i for i, class_name in enumerate(config.CLASS_NAMES)}

    def _encode(label_list: list[str]) -> np.ndarray:
        vector = np.zeros(config.NUM_CLASSES, dtype=np.float32)
        for label in label_list:
            label = label.strip()
            if label in class_to_idx:
                vector[class_to_idx[label]] = 1.0
        return vector

    df["label_vector"] = df["labels"].apply(_encode)

    group_col = _get_group_column(df)
    if group_col is not None:
        df[group_col] = df[group_col].astype(str)

    return df


# =============================================================================
# Subset sampling (fast experiments)
# =============================================================================
def _sample_subset(
    dataframe: pd.DataFrame,
    target_size: int,
    *,
    group_col: str | None,
    random_state: int,
) -> pd.DataFrame:
    """
    Sample a subset, preserving patient groups when possible.

    Preserving groups avoids splitting the same patient across quick-turn
    experiments and keeps the train/val leakage fix intact even in subset mode.
    """
    if target_size <= 0 or len(dataframe) <= target_size:
        return dataframe.reset_index(drop=True)

    # If no grouping available use random sampling fallback
    if group_col is None or group_col not in dataframe.columns:
        return dataframe.sample(n=target_size, random_state=random_state).reset_index(drop=True)

    rng = np.random.default_rng(random_state)
    groups = list(dataframe.groupby(group_col, sort=False))
    order = rng.permutation(len(groups))

    # Greedy group accumulation until target size reached
    selected_groups: list[pd.DataFrame] = []
    selected_rows = 0
    for idx in order:
        _, group_df = groups[idx]
        selected_groups.append(group_df)
        selected_rows += len(group_df)
        if selected_rows >= target_size:
            break

    # Shuffle final subset
    subset = pd.concat(selected_groups, ignore_index=True)
    subset = subset.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return subset


# =============================================================================
# Patient-aware train/val split
# =============================================================================
def _group_stratified_train_val_split(
    dataframe: pd.DataFrame,
    *,
    test_size: float,
    group_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a patient-aware validation split with a greedy multilabel heuristic.

    Exact grouped iterative stratification would require an extra dependency.
    This heuristic keeps every patient's studies together and greedily tries
    to match the validation label marginals to the full train_val set.
    """
    # Fallback: standard split if grouping is not usable
    if group_col not in dataframe.columns or dataframe[group_col].nunique() < 2:
        train_df, val_df = train_test_split(
            dataframe,
            test_size=test_size,
            random_state=config.SEED,
        )
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    label_matrix = _stack_label_vectors(dataframe)
    target_val_size = max(1, int(round(len(dataframe) * test_size)))
    target_label_counts = label_matrix.sum(axis=0) * test_size

    # Aggregate group-level label statistics
    group_records = []
    for group_id, group_df in dataframe.groupby(group_col, sort=False):
        group_labels = _stack_label_vectors(group_df).max(axis=0)
        group_records.append(
            {
                "group_id": group_id,
                "size": len(group_df),
                "labels": group_labels,
            }
        )

    # Prioritize rare/important groups first
    group_records.sort(
        key=lambda record: (record["labels"].sum(), record["size"]),
        reverse=True,
    )

    val_group_ids: set[str] = set()
    val_label_counts = np.zeros(config.NUM_CLASSES, dtype=np.float32)
    val_size = 0

    def _score(candidate_counts: np.ndarray, candidate_size: int) -> float:
        label_error = np.abs(candidate_counts - target_label_counts).sum()
        size_error = abs(candidate_size - target_val_size) * 0.5
        return float(label_error + size_error)

    # Greedy selection of validation groups
    for record in group_records:
        add_to_val = _score(val_label_counts + record["labels"], val_size + record["size"])
        keep_in_train = _score(val_label_counts, val_size)

        if val_size < target_val_size and add_to_val <= keep_in_train:
            val_group_ids.add(record["group_id"])
            val_label_counts += record["labels"]
            val_size += record["size"]

    # Ensure minimum coverage
    if val_size < target_val_size:
        for record in group_records:
            if record["group_id"] in val_group_ids:
                continue
            val_group_ids.add(record["group_id"])
            val_label_counts += record["labels"]
            val_size += record["size"]
            if val_size >= target_val_size:
                break

    # Fallback to sklearn if heuristic fails
    if not val_group_ids or len(val_group_ids) == len(group_records):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=config.SEED,
        )
        train_idx, val_idx = next(
            splitter.split(dataframe, groups=dataframe[group_col].to_numpy())
        )
        train_df = dataframe.iloc[train_idx].reset_index(drop=True)
        val_df = dataframe.iloc[val_idx].reset_index(drop=True)
        return train_df, val_df

    val_mask = dataframe[group_col].isin(val_group_ids)
    val_df = dataframe[val_mask].reset_index(drop=True)
    train_df = dataframe[~val_mask].reset_index(drop=True)

    return train_df, val_df


# =============================================================================
# Dataset class
# =============================================================================
class ChestXrayDataset(Dataset):
    """
    PyTorch Dataset for the NIH Chest X-ray images.

    Each sample is a dict with:
      - 'image': tensor of shape (3, H, W)
      - 'label': tensor of shape (NUM_CLASSES,)  (multi-hot)
      - 'filename': original image filename
    """

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.filepaths = self.df["filepath"].tolist()
        self.filenames = self.df["Image Index"].tolist()
        self.label_matrix = _stack_label_vectors(self.df)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        # Load image on-the-fly (memory efficient)
        with Image.open(self.filepaths[idx]) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

        label = torch.from_numpy(self.label_matrix[idx]).float()
        return {
            "image": image,
            "label": label,
            "filename": self.filenames[idx],
        }

    def get_label_matrix(self) -> np.ndarray:
        """Expose labels for class-weighting and threshold calibration."""
        return self.label_matrix


# =============================================================================
# Transforms
# =============================================================================
def get_train_transforms() -> transforms.Compose:
    """Augmentation pipeline tuned for frontal radiographs."""
    return transforms.Compose(
        [
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomAffine(
                degrees=7,
                translate=(0.02, 0.02),
                scale=(0.95, 1.05),
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
                p=0.1,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )


def get_eval_transforms() -> transforms.Compose:
    """Deterministic pipeline for validation / test."""
    return transforms.Compose(
        [
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )


# =============================================================================
# DataLoader builder
# =============================================================================
def get_dataloaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, validation, and test DataLoaders.

    Uses the official NIH train_val_list.txt / test_list.txt split.
    Validation is carved from the train_val pool with patient-aware grouping.
    If config.SUBSET_SIZE > 0, group-preserving subsets are sampled for quick
    local experiments.
    """
    print("[dataset] Loading metadata ...")
    df = load_metadata()

    # Official split files
    train_val_filenames = set(Path(config.TRAIN_VAL_LIST).read_text().strip().splitlines())
    test_filenames = set(Path(config.TEST_LIST).read_text().strip().splitlines())

    train_val_df = df[df["Image Index"].isin(train_val_filenames)].reset_index(drop=True)
    test_df = df[df["Image Index"].isin(test_filenames)].reset_index(drop=True)

    group_col = _get_group_column(train_val_df)

    # Optional subset mode for debugging/experiments
    if config.SUBSET_SIZE and config.SUBSET_SIZE > 0:
        n_train_val = min(config.SUBSET_SIZE, len(train_val_df))
        n_test = min(max(config.SUBSET_SIZE // 5, 50), len(test_df))

        train_val_df = _sample_subset(
            train_val_df,
            n_train_val,
            group_col=group_col,
            random_state=config.SEED,
        )
        test_df = _sample_subset(
            test_df,
            n_test,
            group_col=_get_group_column(test_df),
            random_state=config.SEED,
        )
        print(
            f"[dataset] Subset mode: {len(train_val_df)} train+val, "
            f"{len(test_df)} test images"
        )

    # Patient-aware train/val split
    train_df, val_df = _group_stratified_train_val_split(
        train_val_df,
        test_size=config.VAL_SPLIT,
        group_col=group_col or "",
    )

    if group_col is not None:
        patient_overlap = set(train_df[group_col]).intersection(set(val_df[group_col]))
        print(f"[dataset] Train/val patient overlap: {len(patient_overlap)}")

    print(
        f"[dataset] Split sizes - train: {len(train_df)}, "
        f"val: {len(val_df)}, test: {len(test_df)}"
    )

    # Datasets
    train_ds = ChestXrayDataset(train_df, transform=get_train_transforms())
    val_ds = ChestXrayDataset(val_df, transform=get_eval_transforms())
    test_ds = ChestXrayDataset(test_df, transform=get_eval_transforms())

    # Loader optimization
    use_pin_memory = torch.cuda.is_available()
    loader_kwargs = {
        "batch_size": config.BATCH_SIZE,
        "num_workers": config.NUM_WORKERS,
        "pin_memory": use_pin_memory,
        # Reproducible shuffling and augmentation across runs
        "worker_init_fn": seed_worker,
    }
    if config.NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    # A generator per loader, so one loader's draws cannot shift another's
    train_loader = DataLoader(
        train_ds, shuffle=True, generator=make_dataloader_generator(), **loader_kwargs
    )
    val_loader = DataLoader(
        val_ds, shuffle=False, generator=make_dataloader_generator(), **loader_kwargs
    )
    test_loader = DataLoader(
        test_ds, shuffle=False, generator=make_dataloader_generator(), **loader_kwargs
    )

    return train_loader, val_loader, test_loader
