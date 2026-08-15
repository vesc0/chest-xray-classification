"""
Weakly-supervised localization evaluation

Scores XAI heatmaps against the 984 hand-drawn boxes shipped with ChestX-ray14
(BBox_List_2017.csv), following Wang et al. (2017) so the numbers are comparable
to the paper. The model is never trained on boxes — this asks whether a
classifier trained on image-level labels happens to look in the right place.

Every explainer the architecture supports is scored on identical instances with
identical metrics: Grad-CAM for all five models, plus Attention Rollout for
ViT-S/16.

One caveat when reading the overlap metrics across architectures: ViT-S/16
produces a 14x14 Grad-CAM grid where the other four produce 7x7, so its
predicted boxes are finer before either is upsampled to 224. That favours ViT on
IoU and IoBB specifically. The pointing game is far less sensitive to grid size
and is the fairer cross-architecture comparison.

Rollout is class-agnostic, so it produces one map per image regardless of the
annotated class — the per-class rows remain well defined but measure where the
model attends overall rather than evidence for that class. That flag is recorded
in the output so the two methods are never read as equivalent.

Reported per class:

  pointing game   does the heatmap's peak fall inside a ground-truth box?
                  Threshold-free and the most robust single number.
  energy fraction share of total heatmap mass inside the box. Also
                  threshold-free, and less noisy than the peak alone. Measured
                  on the un-normalized map: the per-sample min-max the
                  explainers apply for display would subtract away any uniform
                  component, and a uniform map is exactly what the random
                  baseline below is defined as.
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

Alongside them, `degenerate_fraction`: the share of instances where the
explainer returned no signal at all — Grad-CAM's ReLU can zero a map entirely.
Those count as pointing-game misses, which is honest, but a model that scores
badly by pointing in the wrong place and one that scores badly by producing
blank maps have failed in different ways, and the aggregate alone cannot tell
them apart.
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
from explainability import (
    available_explainers,
    build_explainer,
    denormalize,
    release_explainers,
)


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

    Assumes an independent linear scale per axis, which is only the right
    mapping while the eval pipeline is a bare square resize. That assumption is
    checked against the actual transform by
    check_eval_transform_geometry() rather than left as a comment.
    """
    scale_x = config.IMAGE_SIZE / float(width)
    scale_y = config.IMAGE_SIZE / float(height)
    scaled = boxes.copy().astype(np.float64)
    scaled[:, 0] *= scale_x
    scaled[:, 1] *= scale_y
    scaled[:, 2] *= scale_x
    scaled[:, 3] *= scale_y
    return scaled


# A correct pipeline scores ~0.97 against the probe below (bicubic softens the
# rectangle's edges, so exact agreement is not achievable); resize-shorter-side
# plus a centre crop scores 0.54-0.67. Anything under this is a geometry change.
TRANSFORM_PROBE_MIN_IOU = 0.90


def check_eval_transform_geometry(transform=None) -> None:
    """
    Verify that _scale_boxes still describes what the eval transform does.

    Every localization number rests on the box coordinates and the image being
    put through the *same* geometry. Today they are: get_eval_transforms() is a
    direct square resize, which is exactly the per-axis linear scale
    _scale_boxes applies. But that agreement is a coincidence of two files
    matching, and if it breaks it breaks silently — a centre crop or an
    aspect-preserving resize offsets every box, degrades every metric, and
    raises nothing. The failure would read as a model result.

    So this checks behaviour rather than structure: push a synthetic image with
    a known rectangle through the real transform and confirm the rectangle
    lands where _scale_boxes predicts. A structural check ("is there a
    CenterCrop?") only catches the geometry changes someone thought to list;
    this catches any of them.

    The probe is deliberately non-square. A square probe cannot distinguish a
    direct resize from a resize-shorter-side, because on square input they
    agree — and NIH images being square is what would make that mistake
    survive.

    Raises:
        RuntimeError: if the transform no longer matches the box scaling.
    """
    if transform is None:
        transform = get_eval_transforms()

    width, height = 300, 200
    box = np.array([[60.0, 40.0, 90.0, 60.0]])

    probe = np.zeros((height, width, 3), dtype=np.uint8)
    x, y, w, h = box[0].astype(int)
    probe[y : y + h, x : x + w] = 255

    seen = denormalize(transform(Image.fromarray(probe))).max(axis=2) > 127
    expected = _box_mask(_scale_boxes(box, width, height), config.IMAGE_SIZE)

    union = float(np.logical_or(seen, expected).sum())
    iou = float(np.logical_and(seen, expected).sum()) / union if union > 0 else 0.0

    if iou < TRANSFORM_PROBE_MIN_IOU:
        raise RuntimeError(
            "The evaluation transform no longer matches the box scaling in "
            f"_scale_boxes (probe IoU {iou:.3f} < {TRANSFORM_PROBE_MIN_IOU}). "
            "_scale_boxes maps original pixels to IMAGE_SIZE with an independent "
            "linear scale per axis, which is only correct for a direct square "
            "resize. A centre crop or an aspect-preserving resize needs the same "
            "change applied there, or every localization metric is scored "
            "against misplaced boxes."
        )


def _is_degenerate(heatmap: np.ndarray) -> bool:
    """
    True when a heatmap carries no signal at all.

    Both explainers normalize each map to [0, 1] by min-max, so a map that was
    constant before normalization — Grad-CAM's ReLU zeroing one out entirely is
    the usual cause — comes out uniformly zero rather than uniformly anything
    else. That is worth naming: such a map still has an argmax, at index 0, and
    would otherwise be scored as though the model had pointed at the top-left
    corner.
    """
    return float(heatmap.max()) <= 0.0


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

    A heatmap with no positive signal (Grad-CAM's ReLU can zero one out
    entirely) yields no detection at all. Thresholding it would otherwise pass
    every pixel and report a whole-image box, which quietly scores at the random
    baseline instead of admitting the method found nothing.
    """
    maximum = float(heatmap.max())
    if maximum <= 0.0:
        return None

    binary = heatmap >= (config.LOCALIZATION_CAM_THRESHOLD * maximum)
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
    method_label: str = "Heatmap",
) -> None:
    """Original / heatmap / overlay, with ground truth in green and the detection in red."""
    original = denormalize(image_tensor)
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original X-ray")
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title(method_label)
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
        # Confident about a class the image does not have. These are rendered
        # for the *absent* class, not the annotated one, so the figure shows the
        # evidence behind the false positive rather than an unrelated heatmap.
        "false_positive": [
            r for r in sorted(records, key=lambda r: r["top_absent_prob"], reverse=True)
            if r["top_absent_predicted_positive"]
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
            # Share of instances where the explainer returned nothing at all.
            # These are counted as misses above, which is honest, but a model
            # scoring 0.2 because it points in the wrong place and one scoring
            # 0.2 because a fifth of its maps are blank are different failures
            # and should not read the same.
            "degenerate_fraction": round(float(np.mean([r["degenerate"] for r in items])), 4),
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
        "baseline (the box's share of the image).\n"
        "  'Energy' is measured on the un-normalized map, so it shares that baseline."
    )

    blank = {
        name: entry["degenerate_fraction"]
        for name, entry in summary["per_class"].items()
        if entry["degenerate_fraction"] > 0
    }
    if blank:
        print(
            "  WARNING: this explainer returned a blank map for some instances, "
            "counted as misses:\n"
            + "".join(f"    {name}: {share:.1%}\n" for name, share in sorted(blank.items()))
            + "  Those rows are scoring an absent explanation, not a wrong one."
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
    dataset: "_AnnotatedPairs",
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

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        class_indices = batch["class_idx"]

        # Heatmap for the ANNOTATED class, not the top prediction: the whole
        # point is to test the explanation for the finding that is there.
        # Class-agnostic methods ignore the index and return one map per image.
        heatmaps = explainer.generate_batch(images, class_indices)

        # The energy fraction is measured on the un-normalized map. See
        # explainability._normalize_per_sample: min-subtraction removes any
        # uniform component, which is exactly what the random baseline printed
        # beside this metric is defined as.
        raw_maps = explainer.last_raw_maps
        if raw_maps is None:
            raise RuntimeError(
                f"{method_label} did not expose last_raw_maps. The energy "
                "fraction has to be measured before per-sample normalization; "
                "scoring the normalized map instead would make the metric "
                "incomparable to its own random baseline."
            )

        # Reuse that forward pass rather than running the model again — a second
        # pass doubles the work and the peak memory for no new information.
        logits = explainer.last_logits
        if logits is None:
            with torch.no_grad():
                logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()

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
            raw_map = raw_maps[position]

            degenerate = _is_degenerate(heatmap)
            predicted = _predicted_box(heatmap)
            iou, iobb = _overlap_scores(predicted, truth_mask)

            # A map with no signal has an argmax all the same, at index 0. Left
            # to itself the pointing game would score it as a deliberate point
            # at the top-left corner — a miss in almost every case, but a *hit*
            # for any box that happens to reach the corner. Say it is a miss.
            if degenerate:
                hit = False
            else:
                peak = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
                hit = bool(truth_mask[peak])

            total = float(raw_map.sum())

            # A class the image does not have, ranked highest - used to pick
            # illustrative false positives
            absent = probs[position].copy()
            absent[record["labels"] > 0.5] = -1.0
            top_absent = int(np.argmax(absent))

            records.append(
                {
                    "image": record["image"],
                    "pair_index": index,
                    "class_name": record["class_name"],
                    "class_idx": class_idx,
                    "hit": hit,
                    "degenerate": degenerate,
                    "energy_fraction": (
                        float(raw_map[truth_mask].sum() / total) if total > 0 else 0.0
                    ),
                    "box_area_fraction": float(truth_mask.mean()),
                    "iou": float(iou),
                    "iobb": float(iobb),
                    "has_detection": predicted is not None,
                    "probability": float(probs[position, class_idx]),
                    "predicted_positive": bool(probs[position, class_idx] >= thresholds[class_idx]),
                    "top_absent_class": config.CLASS_NAMES[top_absent],
                    "top_absent_idx": top_absent,
                    "top_absent_prob": float(max(absent[top_absent], 0.0)),
                    "top_absent_predicted_positive": bool(
                        absent[top_absent] >= thresholds[top_absent]
                    ),
                }
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
        "energy_measured_on": "un-normalized heatmap (min-max removes the uniform "
                              "component the random baseline is defined as)",
        "blank_maps_counted_as": "pointing-game misses, reported as degenerate_fraction",
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

    # Qualitative figures: a few per category, not 984 overlays
    saved = _render_cases(
        _select_cases(records), model_name, method_key, method_label,
        explainer, dataset, pairs, thresholds, device,
    )
    figure_dir = config.XAI_DIR / model_name / "localization" / method_key
    print(f"[localization] Saved {saved} qualitative figures -> {figure_dir}")
    return summary


def _render_cases(
    selected: dict[str, list[dict]],
    model_name: str,
    method_key: str,
    method_label: str,
    explainer,
    dataset: "_AnnotatedPairs",
    pairs: list[dict],
    thresholds: np.ndarray,
    device: torch.device,
) -> int:
    """
    Draw the chosen examples, re-running the explainer on each one.

    Deliberately re-explains rather than keeping the maps from the scoring pass.
    Caching them costs ~790 MB — 984 instances of a 3x224x224 image plus its
    heatmap — to serve at most a couple of dozen figures, and every scored
    method pays it again. Re-running the explainer on the handful that are
    actually drawn is a few seconds, and the explainers are deterministic in
    eval mode, so the redrawn map is the one that was scored.

    The false-positive category was already re-explaining, because its figure
    has to target the over-predicted class rather than the annotated one. Doing
    it for every category makes that the ordinary path instead of a special
    case: the only thing that varies is which class index gets explained.
    """
    figure_dir = config.XAI_DIR / model_name / "localization" / method_key
    figure_dir.mkdir(parents=True, exist_ok=True)
    agnostic = bool(getattr(explainer, "class_agnostic", False))

    saved = 0
    for category, chosen in selected.items():
        for rank, record in enumerate(chosen):
            sample = dataset[record["pair_index"]]
            image_tensor = sample["image"]
            boxes = _scale_boxes(
                pairs[record["pair_index"]]["boxes"], sample["width"], sample["height"]
            )

            if category == "false_positive":
                # The over-predicted class: a map for the annotated class would
                # not show why this fired.
                target_idx = record["top_absent_idx"]
                title = (
                    f"false positive - {model_name} / {method_label}\n"
                    f"predicted {record['top_absent_class']} "
                    f"p={record['top_absent_prob']:.2f} "
                    f"(thr={thresholds[target_idx]:.2f}) but that class is absent  |  "
                    f"green boxes = true {record['class_name']}"
                    + ("  |  map is class-agnostic" if agnostic else "")
                )
            else:
                target_idx = record["class_idx"]
                title = (
                    f"{category.replace('_', ' ')} - {model_name} / {method_label}\n"
                    f"{record['class_name']}: p={record['probability']:.2f} "
                    f"(thr={thresholds[target_idx]:.2f})  |  "
                    f"hit={record['hit']}  energy={record['energy_fraction']:.2f}  "
                    f"IoBB={record['iobb']:.2f}"
                    + ("  |  no signal in this map" if record["degenerate"] else "")
                )

            heatmap = explainer.generate_batch(
                image_tensor.unsqueeze(0).to(device), [target_idx]
            )[0]

            _save_figure(
                image_tensor, heatmap, boxes, _predicted_box(heatmap), title,
                figure_dir / f"{category}_{rank}_{record['image']}",
                method_label=method_label,
            )
            saved += 1

    return saved


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

    methods = available_explainers(model_name)
    if not methods:
        print(f"[localization] No explainer available for '{model_name}' - skipping.")
        return {}

    # Before anything is scored: the box coordinates and the images have to go
    # through the same geometry, and nothing else in the pipeline would notice
    # if they stopped doing so.
    check_eval_transform_geometry()

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

    # Deliberately not config.BATCH_SIZE — see LOCALIZATION_BATCH_SIZE
    batch_size = max(1, int(config.LOCALIZATION_BATCH_SIZE))
    dataset = _AnnotatedPairs(pairs, get_eval_transforms())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    # One explainer alive at a time. Grad-CAM's hook is inert outside its own
    # generate_batch, so holding both would be safe — but a method's pass has no
    # business keeping another method's hooks and captured activations attached
    # to the model, and the shorter lifetime is what makes that true by
    # construction rather than by a flag.
    summaries: dict[str, dict] = {}
    for method_key, method_label in methods:
        explainer = build_explainer(model, model_name, method_key)
        try:
            summaries[method_key] = _score_method(
                model, model_name, method_key, method_label,
                explainer, pairs, dataset, loader, thresholds, device,
            )
        finally:
            release_explainers([(method_key, method_label, explainer)])

    return summaries
