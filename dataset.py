"""
Dataset module

Handles:
  - Parsing the metadata CSV and official train/test splits
  - Multi-label encoding (14 ChestX-ray14 pathologies)
  - Patient-aware train / validation splitting
  - Group-preserving subset sampling for fast local experiments
  - Radiograph-appropriate augmentation and preprocessing transforms
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from sklearn.model_selection import train_test_split

import config
from utils import make_dataloader_generator, seed_worker


# =============================================================================
# Data indexing utilities
# =============================================================================
# Scanning the image directories costs a full walk of every entry on the
# dataset drive, and several call sites need the index within one run.
_IMAGE_INDEX_CACHE: dict[Path, dict[str, Path]] = {}


def _build_image_index() -> dict[str, Path]:
    """
    Scan all image directories and return {filename: full_path}.

    Cached per dataset root: the walk is slow on an external drive and the
    result cannot change mid-run. Raises if no image directory is readable —
    usually an unmounted drive or an unset XRAY_DATASET_ROOT, which would
    otherwise surface much later as an empty DataFrame.
    """
    cached = _IMAGE_INDEX_CACHE.get(config.DATASET_ROOT)
    if cached is not None:
        return cached

    index: dict[str, Path] = {}
    found_dirs = 0

    for img_dir in config.IMAGE_DIRS:
        if img_dir.exists():
            found_dirs += 1
            for path in img_dir.iterdir():
                # Skip AppleDouble sidecars (._name.png) left by copying to a
                # non-HFS filesystem. They carry the .png suffix but are
                # metadata stubs, and would double the size of the index.
                if path.suffix == ".png" and not path.name.startswith("._"):
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

    _IMAGE_INDEX_CACHE[config.DATASET_ROOT] = index
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
# Grouped iterative stratification
# =============================================================================
def _aggregate_groups(
    dataframe: pd.DataFrame,
    group_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse rows into per-patient label counts and sizes."""
    labels = _stack_label_vectors(dataframe)
    codes, uniques = pd.factorize(dataframe[group_col], sort=False)
    n_groups = len(uniques)

    group_labels = np.zeros((n_groups, config.NUM_CLASSES), dtype=np.float64)
    np.add.at(group_labels, codes, labels)
    group_sizes = np.bincount(codes, minlength=n_groups).astype(np.float64)

    return codes, group_labels, group_sizes


def _iterative_stratified_group_folds(
    dataframe: pd.DataFrame,
    proportions: list[float],
    group_col: str,
    seed: int,
) -> list[np.ndarray]:
    """
    Partition patients into folds with matched multi-label marginals.

    Grouped variant of iterative stratification (Sechidis et al., 2011). Labels
    are placed rarest-first, so scarce pathologies (Hernia at 0.16%) decide the
    assignment while there is still freedom to balance them, and every patient's
    studies stay together. Studies carrying no positive label — about 58% of the
    dataset once "No Finding" stops being a class — are distributed last purely
    on size, so the normal majority cannot skew any fold's label distribution.

    Returns one array of row positions per fold, in the order given.
    """
    rng = np.random.default_rng(seed)
    codes, group_labels, group_sizes = _aggregate_groups(dataframe, group_col)
    n_groups = len(group_sizes)
    n_folds = len(proportions)

    proportions_arr = np.asarray(proportions, dtype=np.float64)
    desired_labels = np.outer(proportions_arr, group_labels.sum(axis=0))
    desired_sizes = proportions_arr * group_sizes.sum()

    fold_labels = np.zeros_like(desired_labels)
    fold_sizes = np.zeros_like(desired_sizes)
    assignment = np.full(n_groups, -1, dtype=np.int64)
    unassigned = np.ones(n_groups, dtype=bool)

    def _assign(group_idx: int, fold_idx: int) -> None:
        assignment[group_idx] = fold_idx
        fold_labels[fold_idx] += group_labels[group_idx]
        fold_sizes[fold_idx] += group_sizes[group_idx]
        unassigned[group_idx] = False

    def _ordered_candidates(mask: np.ndarray) -> np.ndarray:
        """Largest patients first — they constrain the balance most."""
        candidates = np.flatnonzero(mask)
        rng.shuffle(candidates)  # random tie-break between equal-sized patients
        return candidates[np.argsort(-group_sizes[candidates], kind="stable")]

    def _pick(tied_folds: np.ndarray) -> int:
        """
        Choose among equally-deficient folds at random.

        Always taking the lowest index systematically front-loads rare labels
        into the early folds, which biases the nested subset prefixes: a class
        with fewer patients than folds would land entirely in the first ones.
        """
        return int(tied_folds[rng.integers(len(tied_folds))])

    # Phase 1: one pass per label, rarest remaining label first
    for _ in range(config.NUM_CLASSES):
        remaining = group_labels[unassigned].sum(axis=0)
        positive = np.flatnonzero(remaining > 0)
        if positive.size == 0:
            break
        label = int(positive[np.argmin(remaining[positive])])

        for group_idx in _ordered_candidates(unassigned & (group_labels[:, label] > 0)):
            deficit = desired_labels[:, label] - fold_labels[:, label]
            best = np.flatnonzero(deficit == deficit.max())
            if best.size > 1:
                # Tie on this label: fall back to whichever fold is emptiest
                size_deficit = desired_sizes[best] - fold_sizes[best]
                best = best[np.flatnonzero(size_deficit == size_deficit.max())]
            _assign(int(group_idx), _pick(best))

    # Phase 2: patients with no positive label, balanced on size alone
    for group_idx in _ordered_candidates(unassigned):
        size_deficit = desired_sizes - fold_sizes
        _assign(int(group_idx), _pick(np.flatnonzero(size_deficit == size_deficit.max())))

    row_folds = assignment[codes]
    return [np.flatnonzero(row_folds == fold) for fold in range(n_folds)]


# =============================================================================
# Subset sampling (fast experiments)
# =============================================================================
def _order_shards_for_prefixes(
    dataframe: pd.DataFrame,
    shards: list[np.ndarray],
    seed: int,
) -> np.ndarray:
    """
    Order shards so that every prefix tracks the pool's label distribution.

    Stratifying the shards only balances them against each other; a subset takes
    a *prefix* of the sequence, so the order is what a 5k or 15k run actually
    sees. Greedily append whichever shard keeps the running prevalence closest
    to the pool's, scored in relative terms so a rare class (Hernia at 0.16%)
    weighs as much as a common one instead of being drowned out.
    """
    labels = _stack_label_vectors(dataframe)
    target = labels.mean(axis=0)
    target_safe = np.maximum(target, 1e-12)

    shard_counts = np.stack([labels[shard].sum(axis=0) for shard in shards])
    shard_sizes = np.array([len(shard) for shard in shards], dtype=np.float64)

    rng = np.random.default_rng(seed)
    remaining = list(rng.permutation(len(shards)))

    order: list[int] = []
    cumulative_counts = np.zeros(config.NUM_CLASSES, dtype=np.float64)
    cumulative_size = 0.0

    while remaining:
        candidate_counts = cumulative_counts + shard_counts[remaining]
        candidate_sizes = cumulative_size + shard_sizes[remaining]
        deviation = np.abs(
            candidate_counts / candidate_sizes[:, None] - target
        ) / target_safe
        chosen = int(np.argmin(deviation.sum(axis=1)))

        shard_idx = remaining.pop(chosen)
        order.append(shard_idx)
        cumulative_counts += shard_counts[shard_idx]
        cumulative_size += shard_sizes[shard_idx]

    return np.concatenate([shards[idx] for idx in order])


def _stratified_subset(
    dataframe: pd.DataFrame,
    target_size: int,
    *,
    group_col: str | None,
    seed: int,
) -> pd.DataFrame:
    """
    Take a nested, label-stratified prefix of the training pool.

    Patients are partitioned into equal shards with matched label marginals and
    the shards are concatenated, so any prefix approximates the full pool's
    distribution and smaller subsets are strict subsets of larger ones. That
    nesting is what makes a 5k/15k/30k scaling curve measure data volume rather
    than which patients each draw happened to catch.
    """
    if target_size <= 0 or len(dataframe) <= target_size:
        return dataframe.reset_index(drop=True)

    # If no grouping available use random sampling fallback
    if group_col is None or group_col not in dataframe.columns:
        return dataframe.sample(n=target_size, random_state=seed).reset_index(drop=True)

    n_shards = max(2, int(config.SUBSET_SHARDS))
    shards = _iterative_stratified_group_folds(
        dataframe,
        [1.0 / n_shards] * n_shards,
        group_col,
        seed,
    )
    order = _order_shards_for_prefixes(dataframe, shards, seed)
    return dataframe.iloc[order[:target_size]].reset_index(drop=True)


# =============================================================================
# Patient-aware train/val split
# =============================================================================
def _grouped_train_val_split(
    dataframe: pd.DataFrame,
    *,
    test_size: float,
    group_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carve a patient-disjoint, label-stratified validation set.

    Always applied to the full train_val pool before any subsetting, so the
    validation set — and therefore threshold calibration — is identical across
    every experiment.
    """
    # Fallback: standard split if grouping is not usable
    if group_col not in dataframe.columns or dataframe[group_col].nunique() < 2:
        train_df, val_df = train_test_split(
            dataframe,
            test_size=test_size,
            random_state=config.SEED,
        )
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    train_idx, val_idx = _iterative_stratified_group_folds(
        dataframe,
        [1.0 - test_size, test_size],
        group_col,
        config.SEED,
    )
    return (
        dataframe.iloc[train_idx].reset_index(drop=True),
        dataframe.iloc[val_idx].reset_index(drop=True),
    )


# =============================================================================
# Split reporting
# =============================================================================
def _save_split_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Path:
    """
    Record the class set and per-split support alongside the results.

    Makes cross-experiment comparisons auditable: any run whose class list or
    evaluation split differs from another's is not directly comparable.
    """

    def _support(dataframe: pd.DataFrame) -> dict[str, int]:
        matrix = _stack_label_vectors(dataframe)
        return {
            class_name: int(matrix[:, idx].sum())
            for idx, class_name in enumerate(config.CLASS_NAMES)
        }

    payload = {
        "class_names": list(config.CLASS_NAMES),
        "num_classes": config.NUM_CLASSES,
        "subset_size": config.SUBSET_SIZE or 0,
        "seed": config.SEED,
        "sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "normal_fraction": {
            split_name: round(float((_stack_label_vectors(d).sum(axis=1) == 0).mean()), 4)
            for split_name, d in (("train", train_df), ("val", val_df), ("test", test_df))
        },
        "support": {
            "train": _support(train_df),
            "val": _support(val_df),
            "test": _support(test_df),
        },
    }

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / "dataset_summary.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[dataset] Split summary saved -> {path}")
    return path


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
# Bicubic, matching the recipe four of the five backbones were pretrained under
# (SwinV2-T, MaxViT-T, ViT-S/16 and ConvNeXtV2-T; only DenseNet-121 used
# bilinear). torchvision's Resize defaults to bilinear, so leaving it implicit
# meant fine-tuning under a different resampling filter than pretraining. It is
# also the better filter for the 1024 -> 224 downsample these radiographs need.
RESIZE_INTERPOLATION = transforms.InterpolationMode.BICUBIC


def _resize() -> transforms.Resize:
    """
    Square resize to IMAGE_SIZE, shared by both pipelines.

    Deliberately a direct resize rather than resize-shorter-side plus a centre
    crop: the NIH images are already square, so nothing is distorted, and a
    centre crop would cut off the costophrenic angles and lung apices — exactly
    where effusions and pneumothoraces present. Train and eval must stay
    identical here, or the model is calibrated at one geometry and scored at
    another.
    """
    return transforms.Resize(
        (config.IMAGE_SIZE, config.IMAGE_SIZE),
        interpolation=RESIZE_INTERPOLATION,
    )


def get_train_transforms() -> transforms.Compose:
    """
    Augmentation pipeline tuned for frontal radiographs.

    Deliberately mild, and deliberately without a horizontal flip: chest
    radiographs have fixed laterality, so mirroring one would teach the model
    that dextrocardia is unremarkable and blunt a left-sided finding like
    cardiomegaly. The affine ranges approximate real positioning variation
    rather than the aggressive crops used on natural images.
    """
    return transforms.Compose(
        [
            _resize(),
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
            _resize(),
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

    The evaluation data is held fixed across every experiment: the test set is
    always the full official split, and validation is carved from the full
    train_val pool before any subsetting. config.SUBSET_SIZE scales the training
    set only, so a 5k/15k/30k curve measures training data volume rather than a
    shrinking, noisier evaluation set.
    """
    print("[dataset] Loading metadata ...")
    df = load_metadata()

    # Official split files
    train_val_filenames = set(Path(config.TRAIN_VAL_LIST).read_text().strip().splitlines())
    test_filenames = set(Path(config.TEST_LIST).read_text().strip().splitlines())

    train_val_df = df[df["Image Index"].isin(train_val_filenames)].reset_index(drop=True)

    # Test is the full official split, always — never subset
    test_df = df[df["Image Index"].isin(test_filenames)].reset_index(drop=True)

    group_col = _get_group_column(train_val_df)

    # Validation is carved from the FULL pool, before subsetting, so threshold
    # calibration sees identical data in every run
    train_pool_df, val_df = _grouped_train_val_split(
        train_val_df,
        test_size=config.VAL_SPLIT,
        group_col=group_col or "",
    )

    # Optional subset mode: training data only
    if config.SUBSET_SIZE and config.SUBSET_SIZE > 0:
        train_df = _stratified_subset(
            train_pool_df,
            min(config.SUBSET_SIZE, len(train_pool_df)),
            group_col=group_col,
            seed=config.SEED,
        )
        print(
            f"[dataset] Subset mode: {len(train_df)} training images "
            f"(pool of {len(train_pool_df)}); val and test held fixed"
        )
    else:
        train_df = train_pool_df

    if group_col is not None:
        patient_overlap = set(train_df[group_col]).intersection(set(val_df[group_col]))
        print(f"[dataset] Train/val patient overlap: {len(patient_overlap)}")

    print(
        f"[dataset] Split sizes - train: {len(train_df)}, "
        f"val: {len(val_df)}, test: {len(test_df)}"
    )
    _save_split_summary(train_df, val_df, test_df)

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
