"""
Sanity checks for the explanations (Adebayo et al., 2018)

A heatmap that looks anatomically plausible is not evidence that the method is
explaining the model. Adebayo et al., "Sanity Checks for Saliency Maps" (2018),
showed that several widely used methods produce essentially the same map after
the network's weights have been randomized — they were reading edges out of the
input, and the model was decoration. Any such method would sail through
localization.py: it would point at the lungs, score above the random baseline,
and support a completely unearned claim about what the classifier learned.

This module runs the **cascading model-parameter randomization test**. Starting
from the trained network, it randomizes one stage at a time from the classifier
down toward the stem, re-explains the same images after each step, and measures
how far the explanation has moved:

  rank correlation   Spearman correlation between the original heatmap and the
                     heatmap from the partially randomized model, per instance.
                     Should fall off as randomization cascades. A method whose
                     maps survive intact is not a function of the weights.
  pointing game      the same metric localization.py reports, recomputed at each
                     step against the ground-truth boxes, beside the random
                     baseline it has to beat. This is the one that makes the
                     result concrete: localization should collapse to baseline
                     once the layers that produced it are noise.

Both are reported for every explainer the architecture supports, because the
question is asked of the *method*, not of the model. Attention Rollout is a
particularly interesting subject: it is class-agnostic and built from attention
matrices rather than gradients, so there is no reason to assume it behaves like
Grad-CAM here.

The test runs on a seeded subsample of the annotated instances (config
.SANITY_CHECK_SAMPLES) rather than all 984, because it re-explains the set once
per stage per method and the shape of the falloff is clear long before the
standard error matters.

Nothing here mutates the trained model: every randomization is applied to a deep
copy, and the global RNG state is saved and restored around the whole run.
"""

import copy
import json

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Subset

import config
from dataset import get_eval_transforms
from explainability import available_explainers, build_explainer, release_explainers
from localization import (
    _AnnotatedPairs,
    _box_mask,
    _build_pairs,
    _is_degenerate,
    _scale_boxes,
    load_boxes,
)


# =============================================================================
# What counts as a stage, per architecture
# =============================================================================
# Ordered top-down: the classifier first, then back toward the input. Each entry
# is (label, parameter-name prefixes), and randomization is *cascading* — stage
# k is randomized on top of stages 1..k-1, which is the version of the test in
# the paper. The independent variant (restore between steps) answers a different
# question and is not what is run here.
#
# Prefixes rather than module objects because the check is over parameters, and
# a prefix list is something a test can verify covers the model exactly. Every
# entry below is asserted to match at least one parameter, and the union is
# asserted to be all of them — a stage list that silently skips a block would
# understate how much the explanation survives.
RANDOMIZATION_STAGES: dict[str, list[tuple[str, list[str]]]] = {
    "densenet121": [
        ("classifier", ["backbone.classifier"]),
        ("denseblock4", ["backbone.features.norm5", "backbone.features.denseblock4"]),
        ("denseblock3", ["backbone.features.transition3", "backbone.features.denseblock3"]),
        ("denseblock2", ["backbone.features.transition2", "backbone.features.denseblock2"]),
        ("denseblock1", ["backbone.features.transition1", "backbone.features.denseblock1"]),
        ("stem", ["backbone.features.conv0", "backbone.features.norm0"]),
    ],
    # Identical module layout to densenet121 — XRV's DenseNet is a copy of
    # torchvision's with a 1-channel stem — so the stage list is the same.
    "densenet121_xrv": [
        ("classifier", ["backbone.classifier"]),
        ("denseblock4", ["backbone.features.norm5", "backbone.features.denseblock4"]),
        ("denseblock3", ["backbone.features.transition3", "backbone.features.denseblock3"]),
        ("denseblock2", ["backbone.features.transition2", "backbone.features.denseblock2"]),
        ("denseblock1", ["backbone.features.transition1", "backbone.features.denseblock1"]),
        ("stem", ["backbone.features.conv0", "backbone.features.norm0"]),
    ],
    "vit_s_16": [
        ("head", ["backbone.head"]),
        ("blocks 9-11", ["backbone.norm", "backbone.blocks.9", "backbone.blocks.10",
                         "backbone.blocks.11"]),
        ("blocks 6-8", ["backbone.blocks.6", "backbone.blocks.7", "backbone.blocks.8"]),
        ("blocks 3-5", ["backbone.blocks.3", "backbone.blocks.4", "backbone.blocks.5"]),
        ("blocks 0-2", ["backbone.blocks.0", "backbone.blocks.1", "backbone.blocks.2"]),
        ("patch embed", ["backbone.patch_embed", "backbone.cls_token", "backbone.pos_embed"]),
    ],
    "convnextv2_t": [
        ("head", ["backbone.head"]),
        ("stage 4", ["backbone.stages.3"]),
        ("stage 3", ["backbone.stages.2"]),
        ("stage 2", ["backbone.stages.1"]),
        ("stage 1", ["backbone.stages.0"]),
        ("stem", ["backbone.stem"]),
    ],
    # torchvision interleaves stages and patch-merging layers in `features`:
    # 0 = patch embed, 1/3/5/7 = stages, 2/4/6 = the merge before each.
    # Swin V1 and V2 share a module layout: 8 feature stages under `features`,
    # then norm/permute/avgpool/flatten/head. The two entries are identical.
    "swin_t": [
        ("head", ["backbone.head"]),
        ("stage 4", ["backbone.norm", "backbone.features.7", "backbone.features.6"]),
        ("stage 3", ["backbone.features.5", "backbone.features.4"]),
        ("stage 2", ["backbone.features.3", "backbone.features.2"]),
        ("stage 1", ["backbone.features.1"]),
        ("patch embed", ["backbone.features.0"]),
    ],
    "swin_v2_t": [
        ("head", ["backbone.head"]),
        ("stage 4", ["backbone.norm", "backbone.features.7", "backbone.features.6"]),
        ("stage 3", ["backbone.features.5", "backbone.features.4"]),
        ("stage 2", ["backbone.features.3", "backbone.features.2"]),
        ("stage 1", ["backbone.features.1"]),
        ("patch embed", ["backbone.features.0"]),
    ],
    "maxvit_t": [
        ("classifier", ["backbone.classifier"]),
        ("stage 4", ["backbone.blocks.3"]),
        ("stage 3", ["backbone.blocks.2"]),
        ("stage 2", ["backbone.blocks.1"]),
        ("stage 1", ["backbone.blocks.0"]),
        ("stem", ["backbone.stem"]),
    ],
}


def _matches(name: str, prefixes: list[str]) -> bool:
    """Whether a parameter or module name sits under any of the prefixes."""
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def randomize(model: torch.nn.Module, prefixes: list[str]) -> int:
    """
    Re-initialize every parameter under `prefixes`, in place.

    Re-initializes rather than perturbing: the test asks what the explanation
    looks like when a stage carries no learned information, and adding noise to
    trained weights leaves most of that information in place.

    Each module's own `reset_parameters()` does the work where it exists, so a
    layer is re-drawn from the distribution it was originally initialized from
    rather than from one this module picked. BatchNorm additionally has its
    running statistics reset — leaving trained statistics behind on randomized
    weights would be neither the trained model nor a random one.

    Bare parameters have no `reset_parameters` (ViT's class token and position
    embedding are the cases here), so anything still bit-identical afterwards is
    re-drawn from a normal matched to its original scale.

    Returns:
        Number of parameter tensors re-initialized.
    """
    before = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if _matches(name, prefixes)
    }
    if not before:
        raise ValueError(f"No parameters matched {prefixes}.")

    for name, module in model.named_modules():
        if not _matches(name, prefixes):
            continue
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
        # Not implied by reset_parameters; BatchNorm keeps them separately.
        if hasattr(module, "reset_running_stats"):
            module.reset_running_stats()

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in before or not torch.equal(param.detach(), before[name]):
                continue
            scale = float(before[name].std()) if before[name].numel() > 1 else 0.0
            if not np.isfinite(scale) or scale == 0.0:
                scale = 0.02  # nothing to match; the usual transformer init
            param.normal_(0.0, scale)

    return len(before)


# =============================================================================
# Running the test
# =============================================================================
def _explain_subset(explainer, loader, device) -> tuple[np.ndarray, list[int]]:
    """Heatmaps for every instance in the loader, plus their pair indices."""
    maps: list[np.ndarray] = []
    indices: list[int] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        maps.append(explainer.generate_batch(images, batch["class_idx"]))
        indices.extend(int(value) for value in batch["index"])
    return np.concatenate(maps, axis=0), indices


def _rank_correlation(reference: np.ndarray, candidate: np.ndarray) -> float | None:
    """
    Spearman correlation of two heatmaps, or None if either carries no signal.

    A constant map has no ranking to correlate, and scipy returns NaN for it.
    Averaging NaN in would poison the column; counting a blank map as
    "uncorrelated" would be worse, because a method that collapses to blank
    under randomization has in fact stopped tracking the original. Excluded and
    counted separately instead.
    """
    if _is_degenerate(reference) or _is_degenerate(candidate):
        return None
    statistic = spearmanr(reference.ravel(), candidate.ravel()).statistic
    return None if not np.isfinite(statistic) else float(statistic)


def _pointing_game(maps: np.ndarray, indices: list[int],
                   truth_masks: dict[int, np.ndarray]) -> tuple[float, float]:
    """Pointing-game accuracy over a subset, beside its random baseline."""
    hits, baselines = [], []
    for position, index in enumerate(indices):
        mask = truth_masks[index]
        heatmap = maps[position]
        if _is_degenerate(heatmap):
            hits.append(False)
        else:
            peak = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
            hits.append(bool(mask[peak]))
        baselines.append(float(mask.mean()))
    return float(np.mean(hits)), float(np.mean(baselines))


def _check_one_method(
    model: torch.nn.Module,
    model_name: str,
    method_key: str,
    method_label: str,
    loader: DataLoader,
    truth_masks: dict[int, np.ndarray],
    device: torch.device,
) -> dict:
    """Cascading randomization for one explainer."""
    stages = RANDOMIZATION_STAGES[model_name]

    explainer = build_explainer(model, model_name, method_key)
    class_agnostic = bool(getattr(explainer, "class_agnostic", False))
    try:
        reference, indices = _explain_subset(explainer, loader, device)
    finally:
        release_explainers([(method_key, method_label, explainer)])

    hit, baseline = _pointing_game(reference, indices, truth_masks)
    steps = [{
        "stage": "trained (reference)",
        "rank_correlation": 1.0,
        "comparable_instances": len(indices),
        "pointing_game": round(hit, 4),
        "random_baseline": round(baseline, 4),
        "degenerate_fraction": round(
            float(np.mean([_is_degenerate(m) for m in reference])), 4
        ),
    }]

    # The deep copy is what keeps the trained model intact: every stage below
    # randomizes cumulatively on top of the last, so this object degrades over
    # the loop and the caller's model never does.
    scratch = copy.deepcopy(model)

    for label, prefixes in stages:
        randomize(scratch, prefixes)
        explainer = build_explainer(scratch, model_name, method_key)
        try:
            maps, _ = _explain_subset(explainer, loader, device)
        finally:
            release_explainers([(method_key, method_label, explainer)])

        correlations = [
            value for value in (
                _rank_correlation(reference[i], maps[i]) for i in range(len(indices))
            ) if value is not None
        ]
        hit, baseline = _pointing_game(maps, indices, truth_masks)

        steps.append({
            "stage": f"+ {label}",
            "rank_correlation": round(float(np.mean(correlations)), 4) if correlations else None,
            "comparable_instances": len(correlations),
            "pointing_game": round(hit, 4),
            "random_baseline": round(baseline, 4),
            "degenerate_fraction": round(float(np.mean([_is_degenerate(m) for m in maps])), 4),
        })

    # How much of the network has to be destroyed before the explanation stops
    # tracking the trained one. The table shows the whole curve; this is the
    # scalar worth quoting, and it is where the interesting failures live — a
    # method can decorrelate eventually and still be almost untouched by
    # randomizing the layers it is nominally reading.
    randomized = steps[1:]
    stages_until_decorrelated = next(
        (
            position + 1
            for position, step in enumerate(randomized)
            if step["rank_correlation"] is not None
            and abs(step["rank_correlation"]) <= config.SANITY_CHECK_ALARM_CORRELATION
        ),
        None,
    )

    final = steps[-1]["rank_correlation"]
    return {
        "method": method_key,
        "method_label": method_label,
        "class_agnostic": class_agnostic,
        "instances": len(indices),
        "steps": steps,
        "stages_until_decorrelated": stages_until_decorrelated,
        "total_stages": len(stages),
        "fully_randomized_correlation": final,
        # A heuristic flag, not a standard: Adebayo et al. give no threshold.
        # It marks the case the test exists to catch — a map that still tracks
        # the trained one after every weight in the network has been re-drawn,
        # which means it is a function of the input rather than of the model.
        "alarm": bool(final is not None and final > config.SANITY_CHECK_ALARM_CORRELATION),
    }


def _print_result(result: dict, model_name: str) -> None:
    header = (
        f"{'Randomized through':<24}{'RankCorr':>10}{'N':>6}"
        f"{'Point':>8}{'Rand':>8}{'Blank':>8}"
    )
    print(f"\n{'=' * len(header)}")
    print(
        f"  Sanity check (cascading randomization) - {model_name} / "
        f"{result['method_label']}  ({result['instances']} instances)"
    )
    print(f"{'=' * len(header)}")
    print(header)
    print("-" * len(header))
    for step in result["steps"]:
        correlation = (
            f"{step['rank_correlation']:>10.3f}"
            if step["rank_correlation"] is not None else f"{'n/a':>10}"
        )
        print(
            f"{step['stage']:<24}{correlation}{step['comparable_instances']:>6}"
            f"{step['pointing_game']:>8.3f}{step['random_baseline']:>8.3f}"
            f"{step['degenerate_fraction']:>8.3f}"
        )
    print("-" * len(header))
    print(
        "  RankCorr: Spearman vs the trained model's own map; should fall toward 0.\n"
        "  Point vs Rand: localization should collapse to the random baseline.\n"
        "  N: instances where both maps had signal to rank; Blank: share with none."
    )

    if result["class_agnostic"]:
        print(
            "  NOTE: this method is class-agnostic and never reads the classifier,\n"
            "  so the first randomized row cannot move by construction. Read the "
            "falloff\n  from the rows below it."
        )

    survived = result["stages_until_decorrelated"]
    if survived is None:
        print(
            f"  Never decorrelated: still above "
            f"{config.SANITY_CHECK_ALARM_CORRELATION} after all "
            f"{result['total_stages']} stages."
        )
    elif survived > 1:
        print(
            f"  Survived {survived - 1} of {result['total_stages']} randomization "
            f"stages before dropping below {config.SANITY_CHECK_ALARM_CORRELATION} "
            "- the\n  layers randomized in those steps contributed little to the map."
        )

    if result["alarm"]:
        print(
            f"  FAILED: correlation is still "
            f"{result['fully_randomized_correlation']:.3f} with every weight "
            f"re-drawn.\n"
            "  This explanation is largely a function of the input, not the model, "
            "and its\n  localization scores do not support a claim about what the "
            "model learned."
        )
    else:
        print("  PASSED: the explanation degrades as the weights are destroyed.")
    print(f"{'=' * len(header)}\n")


def run_sanity_checks(
    model: torch.nn.Module,
    model_name: str,
    num_samples: int | None = None,
) -> dict:
    """
    Cascading parameter randomization for every explainer the model supports.

    Args:
        model: Trained model. Not modified — the test runs on deep copies.
        model_name: One of config.SUPPORTED_MODELS.
        num_samples: Annotated instances to test on. Defaults to
            config.SANITY_CHECK_SAMPLES.

    Returns:
        {method_key: result}; empty if the check could not be run.
    """
    if num_samples is None:
        num_samples = config.SANITY_CHECK_SAMPLES

    if model_name not in RANDOMIZATION_STAGES:
        print(f"[sanity] No randomization stages defined for '{model_name}' - skipping.")
        return {}

    methods = available_explainers(model_name)
    if not methods:
        print(f"[sanity] No explainer available for '{model_name}' - skipping.")
        return {}

    if not config.BBOX_CSV.exists():
        # The rank correlation alone would still work, but the pointing-game
        # column is what makes the result legible next to localization.py.
        print(f"[sanity] {config.BBOX_CSV} not found - skipping.")
        return {}

    pairs = _build_pairs(load_boxes())
    if not pairs:
        print("[sanity] No annotated images available - skipping.")
        return {}

    device = next(model.parameters()).device
    model.eval()

    # Seeded subsample, drawn across the whole annotated set: the pairs are
    # sorted by image name, so any prefix of them is skewed toward a handful of
    # classes.
    rng = np.random.default_rng(config.SEED)
    count = min(int(num_samples), len(pairs))
    chosen = sorted(rng.choice(len(pairs), size=count, replace=False).tolist())

    dataset = _AnnotatedPairs(pairs, get_eval_transforms())
    loader = DataLoader(
        Subset(dataset, chosen),
        batch_size=max(1, int(config.LOCALIZATION_BATCH_SIZE)),
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    truth_masks = {}
    for index in chosen:
        sample = dataset[index]
        truth_masks[index] = _box_mask(
            _scale_boxes(pairs[index]["boxes"], sample["width"], sample["height"]),
            config.IMAGE_SIZE,
        )

    print(
        f"[sanity] Cascading randomization over {len(RANDOMIZATION_STAGES[model_name])} "
        f"stages on {count} annotated instances for {model_name} ..."
    )

    # reset_parameters() draws from the global RNG. Restore it afterwards so a
    # sanity check cannot change what any later step in the pipeline produces.
    rng_state = torch.get_rng_state()
    torch.manual_seed(config.SEED)
    try:
        results = {}
        for method_key, method_label in methods:
            result = _check_one_method(
                model, model_name, method_key, method_label,
                loader, truth_masks, device,
            )
            _print_result(result, model_name)
            results[method_key] = result
    finally:
        torch.set_rng_state(rng_state)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / f"{model_name}_sanity_checks.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"[sanity] Results saved -> {path}")

    return results
