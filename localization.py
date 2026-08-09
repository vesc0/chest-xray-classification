"""
Weakly-supervised localization evaluation

Scores XAI heatmaps against the 984 hand-drawn boxes shipped with ChestX-ray14
(BBox_List_2017.csv), following Wang et al. (2017) so the numbers are comparable
to the paper. The model is never trained on boxes — this asks whether a
classifier trained on image-level labels happens to look in the right place.

Every explainer the architecture supports is scored on identical instances with
identical metrics: Grad-CAM for all four models, plus Attention Rollout for ViT.
Rollout is class-agnostic, so it produces one map per image regardless of the
annotated class — the per-class rows remain well defined but measure where the
model attends overall rather than evidence for that class. That flag is recorded
in the output so the two methods are never read as equivalent.

Reported per class:

  pointing game   does the heatmap's peak fall inside a ground-truth box?
                  Threshold-free and the most robust single number.
  energy fraction share of total heatmap mass inside the box. Also
                  threshold-free, and less noisy than the peak alone.
  IoU / IoBB      overlap of the *predicted* box (largest connected component
                  of the binarized heatmap) with the ground-truth box, reported
                  as accuracy at each T in config.LOCALIZATION_THRESHOLDS.
                  IoBB here is intersection / predicted-box area, the
                  "intersection over the detected B-Box" of Wang et al.; the
                  literature is inconsistent about this denominator, so the
                  definition is stated rather than assumed.

Every metric is reported next to a random baseline (the box's share of the
image). Without it the numbers are uninterpretable: boxes range from 17.6% of
the image for Cardiomegaly down to 0.5% for Nodule, so an identical hit rate
means very different things per class.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import config
from dataset import get_eval_transforms, load_metadata
from explainability import _denormalize, build_explainers, release_explainers


# =============================================================================
# Ground-truth boxes
# =============================================================================
def load_boxes() -> pd.DataFrame:
    """
    Read BBox_List_2017.csv into columns: image, class_name, class_idx, x/y/w/h.

    Coordinates are in the original 1024x1024 pixel space and are rescaled to
    the model's input resolution later, per image, from the file's real size.
    """
    frame = pd.read_csv(config.BBOX_CSV)
    frame = frame.loc[:, ~frame.columns.str.startswith("Unnamed")]
    frame.columns = ["image", "class_name", "x", "y", "w", "h"]

    frame["class_name"] = frame["class_name"].replace(config.BBOX_LABEL_ALIASES)

    class_to_idx = {name: i for i, name in enumerate(config.CLASS_NAMES)}
    known = frame["class_name"].isin(class_to_idx)
    if not known.all():
        dropped = sorted(frame.loc[~known, "class_name"].unique())
        print(f"[localization] Ignoring boxes for non-modelled classes: {dropped}")
        frame = frame[known].reset_index(drop=True)

    frame["class_idx"] = frame["class_name"].map(class_to_idx)
    return frame


def _build_pairs(boxes: pd.DataFrame) -> list[dict]:
    """
    One record per (image, annotated class): its boxes plus the image's own
    full label vector, which is needed to tell a false positive from a hit.
    """
    metadata = load_metadata().set_index("Image Index")
    pairs: list[dict] = []
    missing = 0

    for (image_name, class_idx), group in boxes.groupby(["image", "class_idx"], sort=True):
        if image_name not in metadata.index:
            missing += 1
            continue
        row = metadata.loc[image_name]
        if isinstance(row, pd.DataFrame):  # duplicate index, take the first
            row = row.iloc[0]

        pairs.append(
            {
                "image": image_name,
                "filepath": row["filepath"],
                "class_idx": int(class_idx),
                "class_name": config.CLASS_NAMES[int(class_idx)],
                "boxes": group[["x", "y", "w", "h"]].to_numpy(dtype=np.float64),
                "labels": np.asarray(row["label_vector"], dtype=np.float32),
            }
        )

    if missing:
        print(f"[localization] {missing} annotated images not found on disk - skipped.")
    return pairs


class _AnnotatedPairs(Dataset):
    """Serves one annotated (image, class) pair at a time."""

    def __init__(self, pairs: list[dict], transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        record = self.pairs[idx]
        with Image.open(record["filepath"]) as image:
            width, height = image.size
            tensor = self.transform(image.convert("RGB"))

        return {
            "image": tensor,
            "class_idx": record["class_idx"],
            "index": idx,
            "width": width,
            "height": height,
        }


# =============================================================================
# Geometry
# =============================================================================
def _scale_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Rescale (x, y, w, h) from original pixels to the model's input grid.

    Uses the file's real dimensions rather than assuming 1024x1024, so this
    keeps working if IMAGE_SIZE or the resize transform ever changes.
    """
    scale_x = config.IMAGE_SIZE / float(width)
    scale_y = config.IMAGE_SIZE / float(height)
    scaled = boxes.copy().astype(np.float64)
    scaled[:, 0] *= scale_x
    scaled[:, 1] *= scale_y
    scaled[:, 2] *= scale_x
    scaled[:, 3] *= scale_y
    return scaled


def _box_mask(boxes: np.ndarray, size: int) -> np.ndarray:
    """Union of the ground-truth boxes as a boolean mask."""
    mask = np.zeros((size, size), dtype=bool)
    for x, y, w, h in boxes:
        x0 = max(int(np.floor(x)), 0)
        y0 = max(int(np.floor(y)), 0)
        x1 = min(int(np.ceil(x + w)), size)
        y1 = min(int(np.ceil(y + h)), size)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return mask


def _predicted_box(heatmap: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Binarize the heatmap and return the bounding box of its largest blob.

    This is the Wang et al. construction: threshold, take connected components,
    keep the biggest, and treat its extent as the detection.
    """
    binary = heatmap >= (config.LOCALIZATION_CAM_THRESHOLD * heatmap.max())
    if not binary.any():
        return None

    labelled, count = ndimage.label(binary)
    if count == 0:
        return None

    sizes = ndimage.sum(binary, labelled, index=range(1, count + 1))
    largest = int(np.argmax(sizes)) + 1
    rows, cols = np.where(labelled == largest)
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


def _overlap_scores(
    predicted: tuple[int, int, int, int] | None,
    truth_mask: np.ndarray,
) -> tuple[float, float]:
    """Return (IoU, IoBB) of the predicted box against the ground-truth mask."""
    if predicted is None:
        return 0.0, 0.0

    x0, y0, x1, y1 = predicted
    predicted_mask = np.zeros_like(truth_mask)
    predicted_mask[y0:y1, x0:x1] = True

    intersection = float(np.logical_and(predicted_mask, truth_mask).sum())
    union = float(np.logical_or(predicted_mask, truth_mask).sum())
    predicted_area = float(predicted_mask.sum())

    iou = intersection / union if union > 0 else 0.0
    # Wang et al.'s "intersection over the detected B-Box area"
    iobb = intersection / predicted_area if predicted_area > 0 else 0.0
    return iou, iobb


# =============================================================================
# Qualitative figures
# =============================================================================
def _save_figure(
    image_tensor: torch.Tensor,
    heatmap: np.ndarray,
    truth_boxes: np.ndarray,
    predicted: tuple[int, int, int, int] | None,
    title: str,
    save_path: Path,
) -> None:
    """Original / heatmap / overlay, with ground truth in green and the detection in red."""
    original = _denormalize(image_tensor)
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original X-ray")
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(original)
    axes[2].imshow(heatmap, cmap="jet", alpha=0.4)
    axes[2].set_title("Overlay - GT (green) vs detected (red)")

    for axis in axes:
        axis.axis("off")

    for x, y, w, h in truth_boxes:
        for axis in (axes[0], axes[2]):
            axis.add_patch(
                patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="lime", facecolor="none")
            )
    if predicted is not None:
        x0, y0, x1, y1 = predicted
        axes[2].add_patch(
            patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                linewidth=2, edgecolor="red", linestyle="--", facecolor="none",
            )
        )

    figure.suptitle(title, fontsize=12)
    figure.tight_layout()
    figure.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _select_cases(records: list[dict]) -> dict[str, list[dict]]:
    """
    Pick the handful of examples worth showing in a report.

    Chosen to be diagnostic rather than flattering: correct detections, correct
    classifications that look in the wrong place, misses, and the best/worst
    localization in the set.
    """
    limit = config.LOCALIZATION_NUM_FIGURES
    by_energy = sorted(records, key=lambda r: r["energy_fraction"], reverse=True)

    return {
        # classified positive and looked in the right place
        "true_positive_localized": [
            r for r in by_energy if r["predicted_positive"] and r["hit"]
        ][:limit],
        # classified positive but looked somewhere else
        "true_positive_mislocalized": [
            r for r in reversed(by_energy) if r["predicted_positive"] and not r["hit"]
        ][:limit],
        # finding present, model said no
        "false_negative": [
            r for r in by_energy if not r["predicted_positive"]
        ][:limit],
        # confident about a class the image does not have
        "false_positive": [
            r for r in sorted(records, key=lambda r: r["top_absent_prob"], reverse=True)
            if r["top_absent_prob"] > 0
        ][:limit],
        "best_localized": by_energy[:limit],
        "worst_localized": by_energy[-limit:][::-1],
    }


# =============================================================================
# Aggregation
# =============================================================================
def _summarize(records: list[dict]) -> dict:
    """Aggregate per-class and macro figures, each beside its random baseline."""
    summary: dict = {"per_class": {}, "macro": {}, "num_instances": len(records)}

    by_class: dict[str, list[dict]] = {}
    for record in records:
        by_class.setdefault(record["class_name"], []).append(record)

    for class_name, items in sorted(by_class.items()):
        entry = {
            "instances": len(items),
            "pointing_game": round(float(np.mean([r["hit"] for r in items])), 4),
            "energy_fraction": round(float(np.mean([r["energy_fraction"] for r in items])), 4),
            # A uniform heatmap scores exactly the box's share of the image, so
            # this is the number every metric above has to beat.
            "random_baseline": round(float(np.mean([r["box_area_fraction"] for r in items])), 4),
            "mean_iou": round(float(np.mean([r["iou"] for r in items])), 4),
            "mean_iobb": round(float(np.mean([r["iobb"] for r in items])), 4),
            "detected_fraction": round(float(np.mean([r["has_detection"] for r in items])), 4),
        }
        for threshold in config.LOCALIZATION_THRESHOLDS:
            key = f"{threshold:g}"
            entry[f"iou@{key}"] = round(float(np.mean([r["iou"] >= threshold for r in items])), 4)
            entry[f"iobb@{key}"] = round(float(np.mean([r["iobb"] >= threshold for r in items])), 4)
        summary["per_class"][class_name] = entry

    keys = set().union(*(entry.keys() for entry in summary["per_class"].values()))
    for key in sorted(keys - {"instances"}):
        summary["macro"][key] = round(
            float(np.mean([entry[key] for entry in summary["per_class"].values()])), 4
        )
    return summary


def _print_summary(summary: dict, model_name: str, method_label: str) -> None:
    thresholds = config.LOCALIZATION_THRESHOLDS
    header = (
        f"{'Class':<20}{'N':>5}{'Point':>8}{'Rand':>8}{'Energy':>8}{'IoU':>7}{'IoBB':>7}"
        + "".join(f"{'IoBB@' + f'{t:g}':>10}" for t in thresholds)
    )
    print(f"\n{'=' * len(header)}")
    print(
        f"  Weakly-supervised localization - {model_name} / {method_label}"
        f"  ({summary['num_instances']} annotated instances)"
    )
    print(f"{'=' * len(header)}")
    print(header)
    print("-" * len(header))

    for class_name, entry in summary["per_class"].items():
        row = (
            f"{class_name:<20}{entry['instances']:>5}"
            f"{entry['pointing_game']:>8.3f}{entry['random_baseline']:>8.3f}"
            f"{entry['energy_fraction']:>8.3f}{entry['mean_iou']:>7.3f}{entry['mean_iobb']:>7.3f}"
        )
        row += "".join(f"{entry[f'iobb@{t:g}']:>10.3f}" for t in thresholds)
        print(row)

    macro = summary["macro"]
    print("-" * len(header))
    row = (
        f"{'Macro average':<20}{'':>5}"
        f"{macro['pointing_game']:>8.3f}{macro['random_baseline']:>8.3f}"
        f"{macro['energy_fraction']:>8.3f}{macro['mean_iou']:>7.3f}{macro['mean_iobb']:>7.3f}"
    )
    row += "".join(f"{macro[f'iobb@{t:g}']:>10.3f}" for t in thresholds)
    print(row)
    print(
        "  'Point' vs 'Rand': pointing-game accuracy against a uniform-heatmap "
        "baseline (the box's share of the image)."
    )
    if summary.get("settings", {}).get("class_agnostic"):
        print(
            "  NOTE: this method is class-agnostic - one map per image, reused for "
            "every annotated class.\n"
            "  Per-class rows therefore measure where the model attends overall, "
            "not evidence for that class."
        )
    print(f"{'=' * len(header)}\n")


# =============================================================================
# Entry point
# =============================================================================
def _score_method(
    model: torch.nn.Module,
    model_name: str,
    method_key: str,
    method_label: str,
    explainer,
    pairs: list[dict],
    loader: DataLoader,
    thresholds: np.ndarray,
    device: torch.device,
) -> dict:
    """Score one explainer over every annotated instance."""
    print(
        f"[localization] Scoring {len(pairs)} annotated (image, class) instances "
        f"for {model_name} / {method_label} ..."
    )

    records: list[dict] = []
    figure_cache: dict[int, tuple] = {}

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        class_indices = batch["class_idx"]

        # Heatmap for the ANNOTATED class, not the top prediction: the whole
        # point is to test the explanation for the finding that is there.
        # Class-agnostic methods ignore the index and return one map per image.
        heatmaps = explainer.generate_batch(images, class_indices)

        with torch.no_grad():
            probs = torch.sigmoid(model(images)).cpu().numpy()

        for position in range(images.size(0)):
            index = int(batch["index"][position])
            record = pairs[index]
            class_idx = int(class_indices[position])

            boxes = _scale_boxes(
                record["boxes"],
                int(batch["width"][position]),
                int(batch["height"][position]),
            )
            truth_mask = _box_mask(boxes, config.IMAGE_SIZE)
            heatmap = heatmaps[position]

            peak = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
            total = float(heatmap.sum())
            predicted = _predicted_box(heatmap)
            iou, iobb = _overlap_scores(predicted, truth_mask)

            # A class the image does not have, ranked highest - used to pick
            # illustrative false positives
            absent = probs[position].copy()
            absent[record["labels"] > 0.5] = -1.0
            top_absent = int(np.argmax(absent))

            records.append(
                {
                    "image": record["image"],
                    "class_name": record["class_name"],
                    "class_idx": class_idx,
                    "hit": bool(truth_mask[peak]),
                    "energy_fraction": float(heatmap[truth_mask].sum() / total) if total > 0 else 0.0,
                    "box_area_fraction": float(truth_mask.mean()),
                    "iou": float(iou),
                    "iobb": float(iobb),
                    "has_detection": predicted is not None,
                    "probability": float(probs[position, class_idx]),
                    "predicted_positive": bool(probs[position, class_idx] >= thresholds[class_idx]),
                    "top_absent_class": config.CLASS_NAMES[top_absent],
                    "top_absent_prob": float(max(absent[top_absent], 0.0)),
                }
            )
            figure_cache[len(records) - 1] = (
                images[position].detach().cpu(), heatmap, boxes, predicted,
            )

    summary = _summarize(records)
    summary["settings"] = {
        "method": method_key,
        "method_label": method_label,
        "class_agnostic": bool(getattr(explainer, "class_agnostic", False)),
        "cam_threshold": config.LOCALIZATION_CAM_THRESHOLD,
        "overlap_thresholds": list(config.LOCALIZATION_THRESHOLDS),
        "iobb_definition": "intersection / predicted box area (Wang et al., 2017)",
        "scored_on": "all annotated instances, regardless of classification outcome",
    }

    # Same numbers restricted to instances the classifier got right - "when it
    # is right, is it right for the right reason?"
    correct = [r for r in records if r["predicted_positive"]]
    summary["predicted_positive_only"] = (
        _summarize(correct) if correct else {"num_instances": 0}
    )

    _print_summary(summary, model_name, method_label)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / f"{model_name}_localization_{method_key}.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[localization] Metrics saved -> {path}")

    # Qualitative figures: a few per category, not 880 overlays
    for index, record in enumerate(records):
        record["_index"] = index
    figure_dir = config.XAI_DIR / model_name / "localization" / method_key
    figure_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for category, chosen in _select_cases(records).items():
        for rank, record in enumerate(chosen):
            image_tensor, heatmap, boxes, predicted = figure_cache[record["_index"]]
            title = (
                f"{category.replace('_', ' ')} - {model_name} / {method_label}\n"
                f"{record['class_name']}: p={record['probability']:.2f} "
                f"(thr={thresholds[record['class_idx']]:.2f})  |  "
                f"hit={record['hit']}  energy={record['energy_fraction']:.2f}  "
                f"IoBB={record['iobb']:.2f}"
            )
            _save_figure(
                image_tensor, heatmap, boxes, predicted, title,
                figure_dir / f"{category}_{rank}_{record['image']}",
            )
            saved += 1

    print(f"[localization] Saved {saved} qualitative figures -> {figure_dir}")
    return summary


def evaluate_localization(
    model: torch.nn.Module,
    model_name: str,
    thresholds: np.ndarray | None = None,
) -> dict:
    """
    Score every available explainer against the ground-truth boxes.

    Grad-CAM is scored for all architectures; ViT is additionally scored with
    Attention Rollout, which makes the two methods comparable on identical data
    and identical metrics.

    Returns {method_key: summary}; empty if nothing could be scored.
    """
    if not Path(config.BBOX_CSV).exists():
        print(f"[localization] {config.BBOX_CSV} not found - skipping.")
        return {}

    explainers = build_explainers(model, model_name)
    if not explainers:
        print(f"[localization] No explainer available for '{model_name}' - skipping.")
        return {}

    device = next(model.parameters()).device
    model.eval()

    if thresholds is None:
        thresholds = np.full(config.NUM_CLASSES, config.DEFAULT_THRESHOLD, dtype=np.float32)
    else:
        thresholds = np.asarray(thresholds, dtype=np.float32)

    pairs = _build_pairs(load_boxes())
    if not pairs:
        print("[localization] No annotated images available - skipping.")
        return {}

    loader = DataLoader(
        _AnnotatedPairs(pairs, get_eval_transforms()),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    summaries: dict[str, dict] = {}
    try:
        for method_key, method_label, explainer in explainers:
            summaries[method_key] = _score_method(
                model, model_name, method_key, method_label,
                explainer, pairs, loader, thresholds, device,
            )
    finally:
        release_explainers(explainers)

    return summaries
