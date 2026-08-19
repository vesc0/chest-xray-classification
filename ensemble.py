"""
Post-hoc ensembling over saved prediction arrays.

Nothing here runs a model. Every finished run in outputs/ already wrote the
probability it assigned to each validation and test image, and those arrays are
row-aligned across runs by construction: the val/test splits are fixed by
config.SEED, never subset, and read by an unshuffled loader, so row i is the
same image in every file. Averaging them is the entire ensemble. It costs
seconds and needs neither the dataset nor a GPU.

That alignment is an assumption strong enough to be worth checking rather than
trusting. A member trained under a different SEED or VAL_SPLIT would pair each
prediction with some other image's label, and the result would not look wrong —
it would be a slightly worse number in a table of plausible numbers. So the
labels and patient IDs of every member are compared before anything is averaged,
and a mismatch raises.

Members are chosen by a rule stated in advance — the best-scoring run per
architecture family, with families below a validation floor dropped — rather
than by a search. Greedy forward selection (Caruana et al., 2004) over the same
pool was measured here and scores about 0.0015 macro AUROC higher, which is
inside the bootstrap interval on the ensemble's own margin. It buys that by
fitting the member list to the 8,652 validation images, and partly by selecting
the strongest run twice. The rule selects nearly the same set and does not ask
the reader to trust that the search was run only once.

The validation floor is what keeps a failed run out. A member that is merely
weaker still contributes — the medically-pretrained DenseNet is the least
accurate member selected here and the largest single contributor to macro
AUPRC, because its errors are the least correlated with the ImageNet backbones'.
A member that never converged is different in kind: it is near-uncorrelated with
everything because it is near-noise, and it drags the mean down.

Members are equally weighted and their probabilities averaged directly.
Averaging in logit space and averaging per-class ranks were both measured on
this pool: logit lands within 0.0001 macro AUROC of the plain mean and rank
about 0.001 below it. All three are implemented so the claim is checkable, and
the simplest is the default.

The ensemble is written as its own experiment rather than as another row beside
an architecture's results, because it is not an architecture and compare_models()
reads its table as one row per architecture.

Usage:
  python ensemble.py --auto
  python ensemble.py --auto --val-floor 0.78 --experiment s5_ensemble_strict
  python ensemble.py --members s3_swin_384:swin_t s2_maxvit_full:maxvit_t
  python ensemble.py --auto --rule logit --no-delta
"""

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

import config
from evaluate import (
    _percentile_ci,
    _rows_by_group,
    apply_thresholds,
    compute_metrics,
    load_predictions,
    load_threshold_status,
    print_results,
    save_predictions,
    save_results,
    save_thresholds,
    threshold_settings,
    tune_thresholds,
    _environment_versions,
)
from metrics import macro_average, per_class_auprc, per_class_auroc
from utils import seed_everything, start_run_log

# Combination rules, in the order they were measured. See the module docstring.
COMBINATION_RULES = ("mean", "logit", "rank", "stack")

# Validation macro AUROC a family's best run must clear to be ensembled. 0.75
# sits well below every converged run in this project (the weakest, medical
# pretraining on PadChest, reaches 0.78) and well above a run that failed to
# train (SwinV2-T reaches 0.67). It is a floor for "did this converge", not a
# quality bar — excluding merely-weak members costs accuracy, see the docstring.
DEFAULT_VAL_FLOOR = 0.75

# Probabilities are clipped this far from the open interval before the logit
# rule takes their log-odds, so a saturated 0.0 or 1.0 cannot become infinite
# and swamp every other member.
LOGIT_EPSILON = 1e-6

# How much a candidate's redundancy with the already-selected members counts
# against its accuracy under --select diverse.
#
# Only the *spread* of each term matters, not its level. Across this project's
# converged runs validation AUROC spans about 0.07 and mean error correlation
# about 0.25, so 0.3 is roughly where the two become equally decisive. Note the
# correlation spread is the wider of the two: a weight near 1 does not weigh
# diversity "a bit more", it ignores accuracy and selects whichever run is most
# unlike the others — which on any real pool is the one that trained worst.
#
# The weight only orders the candidates that get tried. Each one is kept only
# if it actually improves ensemble validation AUROC, so a badly chosen weight
# costs search order, not correctness.
DEFAULT_DIVERSITY_WEIGHT = 0.3

# Validation positives a class needs before it contributes to a diversity
# figure. Below this the correlation is computed on a handful of rows and is
# noise; Hernia has 14 positives in validation.
MIN_POSITIVES_FOR_DIVERSITY = 30


# =============================================================================
# Reading members
# =============================================================================
@contextmanager
def _reading(experiment: str):
    """
    Point config.RESULTS_DIR at another experiment for the duration.

    load_predictions() resolves its path through config, which is per-experiment
    global state — and an ensemble is the one consumer that spans experiments.
    Swapping it here reuses the loader (including its handling of files written
    before patient IDs were stored as text) instead of reimplementing the read.
    """
    saved = config.RESULTS_DIR
    config.RESULTS_DIR = config.PROJECT_ROOT / "outputs" / experiment / "results"
    try:
        yield
    finally:
        config.RESULTS_DIR = saved


def parse_member(spec: str) -> tuple[str, str]:
    """Split an `experiment:model` argument into its two halves."""
    if spec.count(":") != 1:
        raise argparse.ArgumentTypeError(
            f"Member '{spec}' must be written experiment:model, "
            f"e.g. s3_swin_384:swin_t"
        )
    experiment, model_name = spec.split(":")
    if not experiment or not model_name:
        raise argparse.ArgumentTypeError(f"Member '{spec}' has an empty half")
    return experiment, model_name


def read_member(member: tuple[str, str], split: str) -> dict[str, np.ndarray]:
    """One member's saved arrays for one split."""
    experiment, model_name = member
    with _reading(experiment):
        return load_predictions(model_name, split)


def is_ensemble_output(results_dir: Path, model_name: str) -> bool:
    """
    True when this run's results file says it was itself produced by ensembling.

    Ensembles write predictions in the same format as any other run, into the
    same outputs/ tree that --auto scans. Without this, a second --auto run
    finds the first ensemble, scores it highest (it is), selects it as a member,
    and makes it its own baseline — so the reported margin over the best member
    collapses to zero and the ensemble silently contains itself. Running the
    same command twice would produce two different results.
    """
    path = results_dir / f"{model_name}_results.json"
    if not path.exists():
        return False
    try:
        with open(path, encoding="utf-8") as handle:
            return "ensemble" in json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False


def discover_runs(
    output_root: Path | None = None,
    exclude_experiment: str | None = None,
) -> list[tuple[str, str]]:
    """
    Every (experiment, model) in outputs/ that saved both splits.

    A run missing either split cannot be ensembled — thresholds are fitted on
    validation and only then applied to test, so a member that saved only test
    predictions has no calibration half to contribute.

    Ensembles are excluded, by their own results file. `exclude_experiment`
    additionally drops the folder this run is about to write to, which covers
    the one case the marker cannot: an ensemble interrupted after saving
    predictions but before writing its results file.
    """
    output_root = output_root or config.PROJECT_ROOT / "outputs"
    suffix = "_val_predictions.npz"

    found = []
    for path in sorted(output_root.glob(f"*/results/*{suffix}")):
        model_name = path.name[: -len(suffix)]
        experiment = path.parent.parent.name
        if experiment == exclude_experiment:
            continue
        if not (path.parent / f"{model_name}_test_predictions.npz").exists():
            continue
        if is_ensemble_output(path.parent, model_name):
            continue
        found.append((experiment, model_name))
    return found


# =============================================================================
# Member selection
# =============================================================================
def _val_auroc(member: tuple[str, str]) -> float | None:
    data = read_member(member, "val")
    return macro_average(per_class_auroc(data["labels"], data["probs"]))


def select_members(
    val_floor: float = DEFAULT_VAL_FLOOR,
    output_root: Path | None = None,
    exclude_experiment: str | None = None,
) -> tuple[list[tuple[str, str]], list[dict]]:
    """
    Apply the selection rule: best run per architecture family, above the floor.

    The family is the model name, so the two pretraining corpora of
    densenet121_xrv compete with each other and the 224px and 384px DenseNet
    runs compete with each other, but neither competes with Swin. Head-only
    probes and subset runs need no special case: they lose to the full run of
    their own family on validation.

    Returns the selected members and the full scan, so the run log records what
    was considered and why each candidate was kept or dropped.
    """
    scan = []
    best_per_family: dict[str, dict] = {}

    for member in discover_runs(output_root, exclude_experiment=exclude_experiment):
        experiment, model_name = member
        score = _val_auroc(member)
        entry = {
            "experiment": experiment,
            "model": model_name,
            "family": model_name,
            "val_auroc": None if score is None else round(score, 4),
        }

        if score is None:
            entry["status"] = "unscorable"
        elif score < val_floor:
            entry["status"] = f"below floor {val_floor}"
        else:
            entry["_score"] = score
            incumbent = best_per_family.get(model_name)
            if incumbent is None or score > incumbent["_score"]:
                best_per_family[model_name] = entry

        scan.append(entry)

    # Resolved after the whole scan, so a run that was briefly the family's best
    # is reported as beaten by the family's actual winner rather than by
    # whichever run happened to overtake it first.
    for entry in scan:
        score = entry.pop("_score", None)
        if score is None:
            continue
        winner = best_per_family[entry["family"]]
        entry["status"] = (
            "selected" if winner is entry
            else f"beaten within family by {winner['experiment']}"
        )

    selected = [
        (entry["experiment"], entry["model"])
        for entry in sorted(scan, key=lambda e: -(e["val_auroc"] or 0))
        if entry["status"] == "selected"
    ]
    return selected, scan


def select_members_by_diversity(
    val_floor: float = DEFAULT_VAL_FLOOR,
    diversity_weight: float = DEFAULT_DIVERSITY_WEIGHT,
    max_members: int = 8,
    output_root: Path | None = None,
    exclude_experiment: str | None = None,
) -> tuple[list[tuple[str, str]], list[dict]]:
    """
    Grow the ensemble by adding the most accurate member that is least redundant.

    Starting from the best single run, each round scores every remaining
    candidate by `validation AUROC - diversity_weight * mean error correlation
    with the members already chosen`, tries the winner, and keeps it only if the
    ensemble's validation AUROC actually improves. The penalty decides what to
    try next; the measurement decides what to keep.

    On this project's runs this ties the family rule rather than beating it, and
    the README records why: an exhaustive sweep of every equal-weight subset of
    the pool puts the ceiling at 0.8258 test macro AUROC against the family
    rule's 0.8254, so there is roughly four ten-thousandths available to any
    selection strategy, against a bootstrap interval an order of magnitude
    wider. It is implemented because the diversity figures it reports are worth
    having, and because "we tried selecting for complementarity" is a question
    worth being able to answer with a number.
    """
    candidates = []
    for member in discover_runs(output_root, exclude_experiment=exclude_experiment):
        score = _val_auroc(member)
        candidates.append(
            {
                "experiment": member[0],
                "model": member[1],
                "family": member[1],
                "val_auroc": None if score is None else round(score, 4),
                "_member": member,
                "_score": score,
            }
        )

    eligible = [
        entry for entry in candidates
        if entry["_score"] is not None and entry["_score"] >= val_floor
    ]
    for entry in candidates:
        if entry["_score"] is None:
            entry["status"] = "unscorable"
        elif entry["_score"] < val_floor:
            entry["status"] = f"below floor {val_floor}"
        else:
            entry["status"] = "not selected"

    if not eligible:
        for entry in candidates:
            entry.pop("_member", None), entry.pop("_score", None)
        return [], candidates

    labels = read_member(eligible[0]["_member"], "val")["labels"].astype(np.int64)
    probs = {
        entry["experiment"] + ":" + entry["model"]: read_member(entry["_member"], "val")["probs"]
        for entry in eligible
    }

    def key(entry):
        return entry["experiment"] + ":" + entry["model"]

    chosen = [max(eligible, key=lambda entry: entry["_score"])]
    chosen[0]["status"] = "selected (most accurate)"
    best_val = macro_average(per_class_auroc(labels, probs[key(chosen[0])]))

    while len(chosen) < max_members:
        remaining = [entry for entry in eligible if entry not in chosen]
        if not remaining:
            break

        ranked = []
        for entry in remaining:
            redundancy = float(
                np.mean([
                    residual_correlation(labels, probs[key(entry)], probs[key(picked)])
                    for picked in chosen
                ])
            )
            ranked.append((entry["_score"] - diversity_weight * redundancy, redundancy, entry))
        ranked.sort(key=lambda item: -item[0])

        # Try candidates in penalized order and take the first that actually
        # helps, rather than stopping at the first that does not. The two differ
        # whenever the penalty ranks a weak-but-unusual run at the top: stopping
        # there ends the search at one member and reports it as an ensemble.
        accepted = None
        for _, redundancy, candidate in ranked:
            trial = combine([probs[key(entry)] for entry in chosen + [candidate]], "mean")
            trial_val = macro_average(per_class_auroc(labels, trial))
            if trial_val is not None and trial_val > best_val:
                candidate["status"] = (
                    f"selected (error correlation {redundancy:.3f} with the set so far)"
                )
                accepted = (candidate, trial_val)
                break
            candidate["status"] = (
                f"tried at size {len(chosen) + 1}, did not improve the ensemble "
                f"({trial_val:.4f} vs {best_val:.4f})"
                if trial_val is not None
                else "tried, unscorable"
            )

        if accepted is None:
            break

        candidate, best_val = accepted
        chosen.append(candidate)

    members = [entry["_member"] for entry in chosen]
    for entry in candidates:
        entry.pop("_member", None)
        entry.pop("_score", None)
    return members, candidates


# =============================================================================
# Alignment
# =============================================================================
def check_alignment(members: list[tuple[str, str]], split: str) -> dict[str, np.ndarray]:
    """
    Verify every member scored the same rows in the same order.

    Returns the shared labels/groups/class_names, read once from the first
    member. Row alignment is what makes averaging meaningful, and it is not
    self-evident from the files: two runs under different seeds produce arrays
    of identical shape whose rows are different images.
    """
    reference_member, *rest = members
    reference = read_member(reference_member, split)

    for member in rest:
        other = read_member(member, split)
        label = f"{member[0]}:{member[1]}"

        if other["probs"].shape != reference["probs"].shape:
            raise ValueError(
                f"{label} has {other['probs'].shape[0]} {split} rows, but "
                f"{reference_member[0]}:{reference_member[1]} has "
                f"{reference['probs'].shape[0]}. These are different splits and "
                f"cannot be ensembled."
            )

        if not np.array_equal(other["labels"], reference["labels"]):
            raise ValueError(
                f"{label} disagrees with "
                f"{reference_member[0]}:{reference_member[1]} on the {split} "
                f"labels. The two runs saw different rows — most likely a "
                f"different config.SEED or VAL_SPLIT — so averaging them would "
                f"pair each prediction with another image's label."
            )

        if not np.array_equal(other.get("class_names"), reference.get("class_names")):
            raise ValueError(f"{label} was scored against different class names.")

        if ("groups" in other) != ("groups" in reference) or (
            "groups" in reference and not np.array_equal(other["groups"], reference["groups"])
        ):
            raise ValueError(
                f"{label} carries different patient IDs on {split}. The rows are "
                f"not the same images."
            )

    return reference


# =============================================================================
# Diversity
# =============================================================================
def residual_correlation(labels: np.ndarray, probs_a: np.ndarray, probs_b: np.ndarray) -> float:
    """
    How similarly two members err, correlated within each true class separately.

    The obvious measure — correlate the raw residuals p - y — is nearly
    degenerate on this data. The residual is dominated by the label: for a class
    at 4% prevalence almost every row is a negative with residual -p, so two
    models correlate at 0.9+ purely because they agree the disease is rare.
    Splitting by true class removes that shared term and leaves the part that
    matters: among the patients who actually have the disease, do these two
    models miss the *same* ones?

    Returns a correlation in [-1, 1]. Lower means the two make different
    mistakes, which is what makes averaging them worth anything.
    """
    residual_a = probs_a - labels
    residual_b = probs_b - labels

    correlations = []
    for class_idx in range(labels.shape[1]):
        truth = labels[:, class_idx]
        for mask in (truth == 1, truth == 0):
            if mask.sum() < MIN_POSITIVES_FOR_DIVERSITY:
                continue
            first, second = residual_a[mask, class_idx], residual_b[mask, class_idx]
            if first.std() < 1e-12 or second.std() < 1e-12:
                continue
            correlations.append(float(np.corrcoef(first, second)[0, 1]))

    return float(np.mean(correlations)) if correlations else float("nan")


def decision_disagreement(labels: np.ndarray, probs_a: np.ndarray, probs_b: np.ndarray) -> float:
    """
    The share of positive calls two members would make differently.

    Each is allowed exactly as many positives per class as the class has, so the
    figure measures *which* patients each one picks rather than how liberally it
    predicts — two identically-ranked models score 0 however their thresholds
    are set. Reported alongside the correlation because it is the more concrete
    statement: "these two disagree on half the patients they flag".
    """
    scores = []
    for class_idx in range(labels.shape[1]):
        positives = int(labels[:, class_idx].sum())
        if positives < MIN_POSITIVES_FOR_DIVERSITY:
            continue
        flagged_a = np.zeros(labels.shape[0], dtype=bool)
        flagged_b = np.zeros(labels.shape[0], dtype=bool)
        flagged_a[np.argsort(-probs_a[:, class_idx])[:positives]] = True
        flagged_b[np.argsort(-probs_b[:, class_idx])[:positives]] = True
        # Both flag `positives` cells, so the two can differ on at most 2*positives
        # of them; dividing by that puts every class on a 0-1 scale.
        scores.append(float((flagged_a != flagged_b).sum()) / (2 * positives))

    return float(np.mean(scores)) if scores else float("nan")


def diversity_matrices(
    labels: np.ndarray, member_probs: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise error correlation and decision disagreement, both symmetric."""
    count = len(member_probs)
    correlation = np.eye(count)
    disagreement = np.zeros((count, count))

    for i in range(count):
        for j in range(i + 1, count):
            correlation[i, j] = correlation[j, i] = residual_correlation(
                labels, member_probs[i], member_probs[j]
            )
            disagreement[i, j] = disagreement[j, i] = decision_disagreement(
                labels, member_probs[i], member_probs[j]
            )
    return correlation, disagreement


def print_diversity(names: list[str], correlation: np.ndarray, disagreement: np.ndarray) -> None:
    """The complementarity evidence: are these members actually different?"""
    # Column headers are the member index, not a truncated name: at six members
    # the names are long enough that truncating them to a column width makes
    # several of them identical.
    # +6 leaves room for the '[N] ' prefix each row label carries.
    label_width = max(len(name) for name in names) + 6
    keys = [f"[{index + 1}]" for index in range(len(names))]

    def table(matrix: np.ndarray) -> None:
        print(f"  {'':{label_width}}" + "".join(f"{key:>8}" for key in keys))
        for row, name in enumerate(names):
            cells = "".join(
                "       -" if row == col else f"{matrix[row, col]:8.3f}"
                for col in range(len(names))
            )
            print(f"  {keys[row] + ' ' + name:{label_width}}{cells}")

    print("\n[ensemble] Error correlation between members (validation, within true class)")
    print("           Lower is better — it is the part of a member the others do not already have.")
    table(correlation)

    print("\n[ensemble] Share of flagged patients two members disagree on (higher = more complementary)")
    table(disagreement)

    off_diagonal = ~np.eye(len(names), dtype=bool)
    print(
        f"\n  Mean error correlation {correlation[off_diagonal].mean():.3f}, "
        f"mean disagreement {disagreement[off_diagonal].mean():.3f}. "
        f"Members near 1.0 correlation are near-duplicates and contribute little."
    )


# =============================================================================
# Combination
# =============================================================================
def combine(probability_arrays: list[np.ndarray], rule: str = "mean") -> np.ndarray:
    """Average the members into one probability array under the chosen rule."""
    stacked = np.asarray(probability_arrays, dtype=np.float64)

    if rule == "mean":
        return stacked.mean(axis=0)

    if rule == "logit":
        clipped = np.clip(stacked, LOGIT_EPSILON, 1.0 - LOGIT_EPSILON)
        log_odds = np.log(clipped / (1.0 - clipped)).mean(axis=0)
        return 1.0 / (1.0 + np.exp(-log_odds))

    if rule == "rank":
        # Ranked per class, then rescaled to [0, 1]. Scale-free, so it is the
        # rule to reach for if a member is badly calibrated — but it discards
        # the margins, and it measured worst of the three here. The output is
        # not a probability, which is why calibration figures computed from it
        # describe the rank transform rather than the ensemble.
        ranked = np.empty_like(stacked)
        rows = stacked.shape[1]
        for member_idx in range(stacked.shape[0]):
            for class_idx in range(stacked.shape[2]):
                order = stacked[member_idx, :, class_idx].argsort()
                positions = np.empty(rows, dtype=np.float64)
                positions[order] = np.arange(rows, dtype=np.float64)
                ranked[member_idx, :, class_idx] = positions / max(rows - 1, 1)
        return ranked.mean(axis=0)

    if rule == "stack":
        raise ValueError(
            "'stack' is fitted on validation labels and cannot be applied by "
            "combine(), which only sees probabilities. Route it through "
            "stacked_predictions() instead."
        )

    raise ValueError(f"Unknown combination rule '{rule}'. Pick one of {COMBINATION_RULES}.")


# =============================================================================
# Stacking
# =============================================================================
# Inverse regularization for the per-class logistic combiner. Chosen on
# patient-grouped half-splits of validation, not on test: 0.3 was the peak of a
# sweep from 0.003 to 10, and the curve is flat enough between 0.1 and 1 that
# the exact value is not load-bearing. Weaker regularization lets a member with
# 88 validation positives acquire a large coefficient it cannot support.
STACK_REGULARIZATION = 0.3

# Validation positives a class needs before a combiner is fitted for it. Below
# this it keeps the plain mean — the same reasoning as THRESHOLD_MIN_SUPPORT,
# and it is what stops Hernia (14 validation positives, 6 free parameters) from
# being fitted into noise.
STACK_MIN_POSITIVES = 50

# Folds used to produce out-of-fold validation predictions. See
# stacked_predictions() for why they are needed at all.
STACK_FOLDS = 5


def _member_logits(member_probs: list[np.ndarray], class_idx: int) -> np.ndarray:
    """One class's member probabilities as (rows, members) log-odds."""
    stacked = np.asarray([probs[:, class_idx] for probs in member_probs]).T
    clipped = np.clip(stacked, LOGIT_EPSILON, 1.0 - LOGIT_EPSILON)
    return np.log(clipped / (1.0 - clipped))


def fit_stacker(
    labels: np.ndarray,
    member_probs: list[np.ndarray],
    regularization: float = STACK_REGULARIZATION,
) -> dict[int, LogisticRegression]:
    """
    Fit one logistic combiner per class over the members' log-odds.

    Per class rather than one model over all of them: the members do not rank
    equally well on every pathology, and a single global weighting cannot say
    that Swin should dominate Emphysema while the medically-pretrained DenseNet
    carries Cardiomegaly. Fitting on log-odds rather than probabilities makes
    equal weights reproduce the logit mean exactly, so the combiner starts from
    a sensible model and has only to correct it.

    Classes below STACK_MIN_POSITIVES get no entry and fall back to the mean.
    """
    models: dict[int, LogisticRegression] = {}
    for class_idx in range(labels.shape[1]):
        if labels[:, class_idx].sum() < STACK_MIN_POSITIVES:
            continue
        models[class_idx] = LogisticRegression(
            C=regularization, max_iter=2000, solver="lbfgs"
        ).fit(_member_logits(member_probs, class_idx), labels[:, class_idx])
    return models


def apply_stacker(
    models: dict[int, LogisticRegression], member_probs: list[np.ndarray]
) -> np.ndarray:
    """Combined probabilities, falling back to the mean for unfitted classes."""
    combined = np.mean(member_probs, axis=0)
    for class_idx, model in models.items():
        combined[:, class_idx] = model.predict_proba(
            _member_logits(member_probs, class_idx)
        )[:, 1]
    return combined


def _grouped_folds(groups: np.ndarray | None, num_rows: int, num_folds: int) -> np.ndarray:
    """Fold index per row, keeping every patient's studies in one fold."""
    if groups is None:
        return np.random.default_rng(config.SEED).integers(0, num_folds, num_rows)

    patients = np.unique(groups)
    shuffled = np.random.default_rng(config.SEED).permutation(patients)
    fold_of_patient = {patient: index % num_folds for index, patient in enumerate(shuffled)}
    return np.asarray([fold_of_patient[patient] for patient in groups])


def stacked_predictions(
    val_labels: np.ndarray,
    val_members: list[np.ndarray],
    test_members: list[np.ndarray],
    groups: np.ndarray | None,
    regularization: float = STACK_REGULARIZATION,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Out-of-fold validation predictions and test predictions from the combiner.

    The out-of-fold part is not optional. Every other rule here is fixed, so its
    validation predictions are as honest as its test ones and thresholds can be
    tuned on them directly. A stacker is *fitted* on validation, so its
    validation predictions are in-sample and better than it will ever do again —
    tuning thresholds on them picks operating points calibrated to a
    performance the model does not have. Measured on this project's runs, that
    mistake costs 0.006 macro F1 on test, and costs it silently.

    So validation predictions come from folds that never saw the patient being
    predicted, and the model applied to test is refitted on all of validation.
    """
    fold_of_row = _grouped_folds(groups, val_labels.shape[0], STACK_FOLDS)

    out_of_fold = np.mean(val_members, axis=0)
    for fold in np.unique(fold_of_row):
        held_out = fold_of_row == fold
        trained_on = ~held_out
        if not trained_on.any() or not held_out.any():
            continue
        fold_models = fit_stacker(
            val_labels[trained_on],
            [probs[trained_on] for probs in val_members],
            regularization,
        )
        out_of_fold[held_out] = apply_stacker(
            fold_models, [probs[held_out] for probs in val_members]
        )

    models = fit_stacker(val_labels, val_members, regularization)
    test_probs = apply_stacker(models, test_members)

    fitted = sorted(models)
    info = {
        "regularization": regularization,
        "folds": int(STACK_FOLDS),
        "folds_grouped_by_patient": groups is not None,
        "min_positives": STACK_MIN_POSITIVES,
        "classes_fitted": [config.CLASS_NAMES[index] for index in fitted],
        "classes_left_at_mean": [
            name for index, name in enumerate(config.CLASS_NAMES) if index not in models
        ],
        "coefficients": {
            config.CLASS_NAMES[index]: [round(float(w), 4) for w in models[index].coef_[0]]
            for index in fitted
        },
    }
    return out_of_fold, test_probs, info


def print_stacker(info: dict, member_labels: list[str]) -> None:
    print(
        f"\n[ensemble] Stacked combiner: one logistic model per class over member "
        f"log-odds (C={info['regularization']})"
    )
    if info["classes_left_at_mean"]:
        print(
            f"  {len(info['classes_left_at_mean'])} class(es) below "
            f"{info['min_positives']} validation positives, left at the plain mean: "
            f"{', '.join(info['classes_left_at_mean'])}"
        )
    grouping = "patient-grouped" if info["folds_grouped_by_patient"] else "row-level"
    print(
        f"  Validation predictions are out-of-fold ({info['folds']} {grouping} folds), "
        f"so thresholds are not tuned against in-sample scores."
    )
    print(f"\n  {'class':22}" + "".join(f"{name[:9]:>10}" for name in member_labels))
    for name in config.CLASS_NAMES:
        weights = info["coefficients"].get(name)
        if weights is None:
            print(f"  {name:22}{'(mean)':>10}")
            continue
        print(f"  {name:22}" + "".join(f"{w:10.2f}" for w in weights))
    print(
        "\n  A negative weight means the class is better predicted by subtracting "
        "that member — read it as a correction, not as the member being wrong."
    )


# =============================================================================
# Ensemble versus its best member
# =============================================================================
def paired_delta(
    labels: np.ndarray,
    ensemble_probs: np.ndarray,
    baseline_probs: np.ndarray,
    groups: np.ndarray | None,
) -> dict:
    """
    Bootstrap the ensemble-minus-best-member difference on the same resamples.

    The margin an ensemble wins by is small enough that a point estimate does
    not establish it, which is the same reason config.BOOTSTRAP_ENABLED exists.
    Both models are scored on *each* resample and the difference taken there, so
    the shared variance from which images were drawn cancels; comparing two
    independently-resampled intervals for overlap would be a far weaker test.
    Patients resample together, as everywhere else in this project.
    """
    rows_by_group = _rows_by_group(groups) if groups is not None else None
    rng = np.random.default_rng(config.SEED)
    num_rows = labels.shape[0]

    auroc_draws: list[float] = []
    auprc_draws: list[float] = []

    for _ in range(config.BOOTSTRAP_SAMPLES):
        if rows_by_group is None:
            rows = rng.integers(0, num_rows, num_rows)
        else:
            picked = rng.integers(0, len(rows_by_group), len(rows_by_group))
            rows = np.concatenate([rows_by_group[group] for group in picked])

        sample_labels = labels[rows]
        for draws, per_class in ((auroc_draws, per_class_auroc), (auprc_draws, per_class_auprc)):
            ensemble_macro = macro_average(per_class(sample_labels, ensemble_probs[rows]))
            baseline_macro = macro_average(per_class(sample_labels, baseline_probs[rows]))
            if ensemble_macro is not None and baseline_macro is not None:
                draws.append(ensemble_macro - baseline_macro)

    summary = {}
    for name, draws in (("auroc", auroc_draws), ("auprc", auprc_draws)):
        if not draws:
            summary[name] = None
            continue
        values = np.asarray(draws)
        summary[name] = {
            "delta": round(float(values.mean()), 4),
            "ci": _percentile_ci(draws),
            "fraction_of_draws_favouring_ensemble": round(float((values > 0).mean()), 4),
        }
    return summary


# =============================================================================
# Reporting
# =============================================================================
def print_scan(scan: list[dict], val_floor: float, rule: str = "best per family") -> None:
    print(f"\n[ensemble] Candidate runs (rule: {rule}, floor {val_floor} val AUROC)")
    print(f"  {'experiment':34}{'model':18}{'val AUROC':>10}  status")
    print(f"  {'-' * 34}{'-' * 18}{'-' * 10}  {'-' * 34}")
    for entry in sorted(scan, key=lambda e: -(e["val_auroc"] or 0)):
        score = "n/a" if entry["val_auroc"] is None else f"{entry['val_auroc']:.4f}"
        print(f"  {entry['experiment']:34}{entry['model']:18}{score:>10}  {entry['status']}")


def print_members(member_stats: list[dict], rule: str) -> None:
    print(f"\n[ensemble] Members ({len(member_stats)}, equally weighted, '{rule}' rule)")
    print(f"  {'experiment':34}{'model':18}{'val AUROC':>10}{'test AUROC':>12}")
    print(f"  {'-' * 34}{'-' * 18}{'-' * 10}{'-' * 12}")
    for entry in member_stats:
        print(
            f"  {entry['experiment']:34}{entry['model']:18}"
            f"{entry['val_auroc']:>10.4f}{entry['test_auroc']:>12.4f}"
        )


def print_delta(delta: dict, baseline: dict) -> None:
    print(
        f"\n[ensemble] Against the best single member "
        f"({baseline['experiment']}:{baseline['model']}), on shared resamples:"
    )
    for name in ("auroc", "auprc"):
        entry = delta.get(name)
        if entry is None:
            print(f"  macro {name.upper()}: not scorable")
            continue
        interval = "n/a" if entry["ci"] is None else f"[{entry['ci'][0]:+.4f}, {entry['ci'][1]:+.4f}]"
        print(
            f"  macro {name.upper()}: delta {entry['delta']:+.4f}  "
            f"{int(config.BOOTSTRAP_CI * 100)}% CI {interval}  "
            f"favoured in {entry['fraction_of_draws_favouring_ensemble']:.1%} of draws"
        )
    print(
        "  An interval containing zero means this ensemble is not measurably "
        "better than its best member."
    )


# =============================================================================
# Pipeline
# =============================================================================
def run_ensemble(
    members: list[tuple[str, str]],
    model_name: str = "ensemble",
    rule: str = "mean",
    compute_delta: bool = True,
    selection: dict | None = None,
    stack_regularization: float = STACK_REGULARIZATION,
) -> dict:
    """Average the members, calibrate on validation, and score on test."""
    if len(members) < 2:
        raise ValueError(
            f"An ensemble needs at least two members; got {len(members)}. "
            f"Lower --val-floor, or name the members explicitly with --members."
        )

    print(f"\n[ensemble] Checking that all {len(members)} members scored the same rows ...")
    val_reference = check_alignment(members, "val")
    test_reference = check_alignment(members, "test")
    print(
        f"  val:  {val_reference['probs'].shape[0]} rows aligned\n"
        f"  test: {test_reference['probs'].shape[0]} rows aligned"
    )

    val_labels = val_reference["labels"].astype(np.int64)
    test_labels = test_reference["labels"].astype(np.int64)
    test_groups = test_reference.get("groups")
    val_groups = val_reference.get("groups")

    member_val = [read_member(member, "val")["probs"] for member in members]
    member_test = [read_member(member, "test")["probs"] for member in members]

    member_stats = [
        {
            "experiment": experiment,
            "model": model,
            "val_auroc": round(macro_average(per_class_auroc(val_labels, val_probs)), 4),
            "test_auroc": round(macro_average(per_class_auroc(test_labels, test_probs)), 4),
        }
        for (experiment, model), val_probs, test_probs in zip(members, member_val, member_test)
    ]
    print_members(member_stats, rule)

    labels_for_diversity = val_labels
    correlation, disagreement = diversity_matrices(labels_for_diversity, member_val)
    print_diversity([f"{e}:{m}" for e, m in members], correlation, disagreement)

    stacker_info = None
    if rule == "stack":
        val_probs, test_probs, stacker_info = stacked_predictions(
            val_labels, member_val, member_test, val_groups, stack_regularization
        )
        print_stacker(stacker_info, [f"{e}:{m}" for e, m in members])
    else:
        val_probs = combine(member_val, rule)
        test_probs = combine(member_test, rule)

    # Same order as a trained run: fit on validation, freeze, then score test.
    print(f"\n[ensemble] Calibrating thresholds on the validation set ...")
    save_predictions(val_labels, val_probs, model_name, "val", groups=val_groups)
    thresholds, summary = tune_thresholds(val_labels, val_probs)
    fallbacks = [name for name, entry in summary.items() if entry["status"] != "tuned"]
    if fallbacks:
        print(
            f"[ensemble] {len(fallbacks)} class(es) left at the default threshold "
            f"({config.DEFAULT_THRESHOLD}): {', '.join(fallbacks)}"
        )
    save_thresholds(thresholds, summary, model_name, num_val_samples=int(val_labels.shape[0]))

    print(f"[ensemble] Scoring the test set ...")
    save_predictions(test_labels, test_probs, model_name, "test", groups=test_groups)
    preds = apply_thresholds(test_probs, thresholds)

    groups = test_groups if config.BOOTSTRAP_GROUP_BY_PATIENT else None
    if config.BOOTSTRAP_ENABLED and config.BOOTSTRAP_GROUP_BY_PATIENT and groups is None:
        print(
            "[ensemble] WARNING: patient IDs unavailable; falling back to an "
            "image-level bootstrap. Intervals will be narrower than they should be."
        )

    results = compute_metrics(
        test_labels,
        test_probs,
        preds,
        thresholds=thresholds,
        groups=groups,
        threshold_status=load_threshold_status(model_name),
    )

    best_member = max(member_stats, key=lambda entry: entry["val_auroc"])
    delta = None
    if compute_delta and config.BOOTSTRAP_ENABLED:
        print(
            f"\n[ensemble] Bootstrapping the margin over "
            f"{best_member['experiment']}:{best_member['model']} "
            f"({config.BOOTSTRAP_SAMPLES} resamples) ..."
        )
        baseline_index = member_stats.index(best_member)
        delta = paired_delta(test_labels, test_probs, member_test[baseline_index], groups)

    off_diagonal = ~np.eye(len(members), dtype=bool)
    results["ensemble"] = {
        "members": member_stats,
        "num_members": len(members),
        "combination_rule": rule,
        "weighting": "fitted per class" if rule == "stack" else "equal",
        "stacker": stacker_info,
        "diversity": {
            "measured_on": "validation",
            "mean_error_correlation": round(float(correlation[off_diagonal].mean()), 4),
            "mean_decision_disagreement": round(float(disagreement[off_diagonal].mean()), 4),
            "error_correlation": [[round(float(v), 4) for v in row] for row in correlation],
            "decision_disagreement": [[round(float(v), 4) for v in row] for row in disagreement],
        },
        # Named on validation, so the comparison below is not a test-set choice.
        "baseline_member": f"{best_member['experiment']}:{best_member['model']}",
        "baseline_selected_by": "highest validation macro AUROC among members",
        "delta_vs_baseline": delta,
        "selection": selection or {"rule": "explicit --members"},
    }

    results["run"] = {
        "trained_this_run": False,
        "source": "post-hoc average of saved predictions; no model was run",
        **threshold_settings(),
        "ece_bins": config.ECE_BINS,
        "ece_bin_strategy": str(config.ECE_BIN_STRATEGY).lower(),
        "seed": config.SEED,
        "versions": _environment_versions(),
    }

    print_results(results, model_name)
    if delta is not None:
        print_delta(delta, best_member)
    save_results(results, model_name)
    return results


# =============================================================================
# Entry point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Average saved predictions from several runs into one ensemble"
    )
    parser.add_argument(
        "--members",
        type=parse_member,
        nargs="+",
        default=None,
        help="Members as experiment:model, e.g. s3_swin_384:swin_t s2_maxvit_full:maxvit_t",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help=(
            "Select members by rule instead: the best-scoring run per "
            "architecture family, dropping families below --val-floor"
        ),
    )
    parser.add_argument(
        "--select",
        type=str,
        default="family",
        choices=["family", "diverse"],
        help=(
            "Which rule --auto applies. 'family' takes the best run per "
            "architecture family. 'diverse' grows the set by adding the most "
            "accurate member least redundant with those already chosen, keeping "
            "it only if validation AUROC improves. They tie on this project's "
            "runs — see the README for the exhaustive ceiling that explains why"
        ),
    )
    parser.add_argument(
        "--diversity-weight",
        type=float,
        default=DEFAULT_DIVERSITY_WEIGHT,
        help=(
            f"How much redundancy counts against accuracy under --select "
            f"diverse (default {DEFAULT_DIVERSITY_WEIGHT}); 0 selects on "
            f"accuracy alone"
        ),
    )
    parser.add_argument(
        "--val-floor",
        type=float,
        default=DEFAULT_VAL_FLOOR,
        help=(
            f"Validation macro AUROC a family's best run must clear under "
            f"--auto (default {DEFAULT_VAL_FLOOR}). This is a did-it-converge "
            f"floor; a merely weak member usually still helps"
        ),
    )
    parser.add_argument(
        "--rule",
        type=str,
        default="mean",
        choices=COMBINATION_RULES,
        help=(
            "How member probabilities are combined. 'mean' (default) is the "
            "equal-weight average. 'logit' and 'rank' are the same idea in "
            "other spaces and measured within 0.001 of it. 'stack' fits one "
            "logistic combiner per class on validation — the only rule here "
            "that improves the operating point as well as the ranking, at the "
            "cost of being fitted rather than fixed"
        ),
    )
    parser.add_argument(
        "--stack-regularization",
        type=float,
        default=STACK_REGULARIZATION,
        help=(
            f"Inverse regularization strength for --rule stack (default "
            f"{STACK_REGULARIZATION}); smaller is more strongly regularized"
        ),
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="s5_ensemble",
        help="Output folder under outputs/ (default s5_ensemble)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="ensemble",
        help="Name the results files are written under (default 'ensemble')",
    )
    parser.add_argument(
        "--no-delta",
        action="store_true",
        help="Skip the paired bootstrap against the best single member",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the members the rule selects and exit without scoring",
    )

    args = parser.parse_args()

    if bool(args.members) == bool(args.auto):
        parser.error("Pass exactly one of --members or --auto")

    seed_everything()

    selection = None
    if args.auto:
        if args.select == "diverse":
            members, scan = select_members_by_diversity(
                args.val_floor,
                diversity_weight=args.diversity_weight,
                exclude_experiment=args.experiment,
            )
            rule_description = (
                f"greedy accuracy minus {args.diversity_weight} x error "
                f"correlation, kept only while validation AUROC improves"
            )
        else:
            members, scan = select_members(args.val_floor, exclude_experiment=args.experiment)
            rule_description = "best validation macro AUROC per architecture family"

        print_scan(scan, args.val_floor, rule_description)
        selection = {
            "rule": rule_description,
            "mode": args.select,
            "val_floor": args.val_floor,
            "diversity_weight": args.diversity_weight if args.select == "diverse" else None,
            "candidates_scanned": scan,
        }
        if not members:
            print("\n[ensemble] ERROR: the rule selected nothing. Lower --val-floor.")
            sys.exit(1)
    else:
        members = args.members
        missing = [
            f"{experiment}:{model}"
            for experiment, model in members
            if not (
                config.PROJECT_ROOT / "outputs" / experiment / "results"
                / f"{model}_val_predictions.npz"
            ).exists()
        ]
        if missing:
            print(f"\n[ensemble] ERROR: no saved predictions for {', '.join(missing)}")
            print("       Available: " + ", ".join(f"{e}:{m}" for e, m in discover_runs()))
            sys.exit(1)

        nested = [
            f"{experiment}:{model}"
            for experiment, model in members
            if is_ensemble_output(
                config.PROJECT_ROOT / "outputs" / experiment / "results", model
            )
        ]
        if nested:
            print(
                f"\n[ensemble] ERROR: {', '.join(nested)} is itself an ensemble. "
                f"Ensembling one would weight its members twice over and leave "
                f"the margin over 'the best member' comparing an ensemble to "
                f"itself. Name the underlying runs instead."
            )
            sys.exit(1)

    if args.dry_run:
        print("\n[ensemble] Would ensemble:")
        for experiment, model in members:
            print(f"  {experiment}:{model}")
        return

    config.set_experiment(args.experiment)
    log_path = start_run_log("ensemble")
    print(f"[ensemble] Log file:   {log_path}")
    print(f"[ensemble] Experiment: {config.EXPERIMENT_NAME}")
    print(f"[ensemble] Output dir: {config.OUTPUT_DIR}")

    run_ensemble(
        members,
        model_name=args.name,
        rule=args.rule,
        compute_delta=not args.no_delta,
        selection=selection,
        stack_regularization=args.stack_regularization,
    )

    print(
        f"\n[ensemble] Done. The ensemble is now a run like any other: "
        f"threshold_analysis.py --experiment {args.experiment} --model {args.name} "
        f"reads it, and --compare-all lists it alongside the architectures."
    )


if __name__ == "__main__":
    main()
