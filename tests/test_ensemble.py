"""
Post-hoc ensembling: member selection, alignment, and the combination rules.

Nothing here runs a model or reads the real outputs/ tree — every test builds a
small synthetic one under tmp_path and points config.PROJECT_ROOT at it.
"""

import json

import numpy as np
import pytest

import config
from ensemble import (
    _grouped_folds,
    _member_logits,
    apply_stacker,
    check_alignment,
    combine,
    decision_disagreement,
    discover_runs,
    diversity_matrices,
    is_ensemble_output,
    parse_member,
    residual_correlation,
    fit_stacker,
    select_members,
    select_members_by_diversity,
    stacked_predictions,
)


def write_run(root, experiment, model, labels, probs, groups=None, is_ensemble=False):
    """Lay down one run's saved predictions the way evaluate.py would."""
    results = root / "outputs" / experiment / "results"
    results.mkdir(parents=True, exist_ok=True)

    for split in ("val", "test"):
        payload = {
            "labels": labels.astype(np.int8),
            "probs": probs.astype(np.float32),
            "class_names": np.asarray(config.CLASS_NAMES),
        }
        if groups is not None:
            payload["groups"] = np.asarray(groups).astype(str)
        np.savez_compressed(results / f"{model}_{split}_predictions.npz", **payload)

    if is_ensemble:
        (results / f"{model}_results.json").write_text(
            json.dumps({"macro": {}, "ensemble": {"members": []}}), encoding="utf-8"
        )
    return results


@pytest.fixture
def labels(rng):
    """40 rows with every class populated, so no AUROC is undefined."""
    values = np.zeros((40, config.NUM_CLASSES), dtype=np.float32)
    for class_idx in range(config.NUM_CLASSES):
        values[rng.choice(40, 12, replace=False), class_idx] = 1.0
    return values


def strong(labels, rng):
    """Probabilities that rank the positives above the negatives."""
    return np.clip(labels * 0.6 + rng.random(labels.shape) * 0.3, 0, 1).astype(np.float32)


def chance(labels, rng):
    return rng.random(labels.shape).astype(np.float32)


def degraded(labels, rng, fraction):
    """
    Perfectly separating probabilities, with `fraction` of the rows randomized.

    Blending a noisy signal toward noise is not monotonic in AUROC, so ranking
    runs by a mixing weight makes for a fixture that fails on some seeds.
    Replacing whole rows is monotonic: more randomized rows is strictly less
    signal.
    """
    probs = (labels * 0.5 + 0.25).astype(np.float32)
    count = int(round(fraction * len(labels)))
    if count:
        rows = rng.choice(len(labels), count, replace=False)
        probs[rows] = rng.random((count, labels.shape[1])).astype(np.float32)
    return probs


@pytest.fixture
def wide_labels(rng):
    """
    200 rows with 60 positives per class.

    The diversity measures need MIN_POSITIVES_FOR_DIVERSITY rows on *both* sides
    of each class, so the 40-row fixture above scores every class as unusable
    and every correlation as NaN.
    """
    values = np.zeros((200, config.NUM_CLASSES), dtype=np.float32)
    for class_idx in range(config.NUM_CLASSES):
        values[rng.choice(200, 60, replace=False), class_idx] = 1.0
    return values


class TestResidualCorrelation:
    def test_a_member_correlates_perfectly_with_itself(self, wide_labels, rng):
        probs = strong(wide_labels, rng)
        assert residual_correlation(wide_labels, probs, probs) == pytest.approx(1.0)

    def test_independent_members_correlate_near_zero(self, wide_labels, rng):
        first = rng.random(wide_labels.shape)
        second = rng.random(wide_labels.shape)
        assert abs(residual_correlation(wide_labels, first, second)) < 0.15

    def test_it_is_symmetric(self, wide_labels, rng):
        first, second = strong(wide_labels, rng), chance(wide_labels, rng)
        assert residual_correlation(wide_labels, first, second) == pytest.approx(
            residual_correlation(wide_labels, second, first)
        )

    def test_two_accurate_members_are_not_scored_as_identical(self, wide_labels, rng):
        """
        The measure this replaced correlated everything at 0.9+.

        Correlating the raw residual p - y is dominated by the shared label
        term: at this data's prevalence nearly every row is a negative, so two
        models look near-identical for agreeing the disease is rare. Splitting
        by true class is what makes the figure informative, and this pins it —
        two independently-noised strong models must not read as duplicates.
        """
        first = np.clip(wide_labels * 0.6 + rng.random(wide_labels.shape) * 0.4, 0, 1)
        second = np.clip(wide_labels * 0.6 + rng.random(wide_labels.shape) * 0.4, 0, 1)
        assert residual_correlation(wide_labels, first, second) < 0.5


class TestDecisionDisagreement:
    def test_a_member_never_disagrees_with_itself(self, wide_labels, rng):
        probs = strong(wide_labels, rng)
        assert decision_disagreement(wide_labels, probs, probs) == pytest.approx(0.0)

    def test_opposite_rankings_disagree_completely(self, wide_labels, rng):
        probs = rng.random(wide_labels.shape)
        assert decision_disagreement(wide_labels, probs, -probs) == pytest.approx(1.0)

    def test_it_ignores_how_liberally_a_member_predicts(self, wide_labels, rng):
        """Each member is given the same number of positive calls, so two models
        that rank identically score 0 however their scores are scaled."""
        probs = rng.random(wide_labels.shape)
        assert decision_disagreement(wide_labels, probs, probs * 0.1) == pytest.approx(0.0)


class TestDiversityMatrices:
    def test_both_matrices_are_symmetric_with_a_clean_diagonal(self, wide_labels, rng):
        members = [strong(wide_labels, rng), chance(wide_labels, rng), strong(wide_labels, rng)]
        correlation, disagreement = diversity_matrices(wide_labels, members)

        assert correlation.shape == disagreement.shape == (3, 3)
        assert correlation == pytest.approx(correlation.T)
        assert disagreement == pytest.approx(disagreement.T)
        assert np.diag(correlation) == pytest.approx(np.ones(3))
        assert np.diag(disagreement) == pytest.approx(np.zeros(3))


class TestSelectMembersByDiversity:
    def test_it_prefers_a_complementary_member_over_a_near_duplicate(
        self, tmp_path, wide_labels, rng
    ):
        config.PROJECT_ROOT = tmp_path
        base = degraded(wide_labels, rng, 0.10)
        write_run(tmp_path, "best", "swin_t", wide_labels, base)
        # A near-copy of the best run, and an independently-noised run of equal
        # accuracy. Both clear the floor; only the second adds anything.
        write_run(tmp_path, "duplicate", "maxvit_t", wide_labels, base * 0.99 + 0.005)
        write_run(tmp_path, "complementary", "densenet121", wide_labels,
                  degraded(wide_labels, rng, 0.10))

        selected, _ = select_members_by_diversity(
            val_floor=0.0, output_root=tmp_path / "outputs"
        )

        assert ("best", "swin_t") in selected
        assert ("complementary", "densenet121") in selected

    def test_a_weak_outlier_does_not_end_the_search(self, tmp_path, wide_labels, rng):
        """
        Regression: the penalty ranks a weak-but-unusual run first.

        Stopping at the first candidate that fails to improve ended the search
        there and returned a one-member 'ensemble'. The search must skip it and
        carry on to the candidates that do help.
        """
        config.PROJECT_ROOT = tmp_path
        write_run(tmp_path, "best", "swin_t", wide_labels, degraded(wide_labels, rng, 0.05))
        write_run(tmp_path, "noise", "swin_v2_t", wide_labels, chance(wide_labels, rng))
        write_run(tmp_path, "useful", "densenet121", wide_labels, degraded(wide_labels, rng, 0.15))

        selected, scan = select_members_by_diversity(
            val_floor=0.0, output_root=tmp_path / "outputs", diversity_weight=1.0
        )

        assert len(selected) > 1, "search stopped at the first unhelpful candidate"
        assert ("noise", "swin_v2_t") not in selected
        tried = [e for e in scan if "did not improve" in e["status"]]
        assert tried, "the rejected candidate should be recorded as tried"

    def test_the_floor_still_applies(self, tmp_path, wide_labels, rng):
        config.PROJECT_ROOT = tmp_path
        write_run(tmp_path, "good", "swin_t", wide_labels, degraded(wide_labels, rng, 0.05))
        write_run(tmp_path, "failed", "swin_v2_t", wide_labels, chance(wide_labels, rng))

        selected, scan = select_members_by_diversity(
            val_floor=0.75, output_root=tmp_path / "outputs"
        )

        assert ("failed", "swin_v2_t") not in selected
        dropped = next(e for e in scan if e["experiment"] == "failed")
        assert "below floor" in dropped["status"]

    def test_it_never_returns_internal_bookkeeping(self, tmp_path, wide_labels, rng):
        """The scan is written into the results JSON, so it must stay serializable."""
        config.PROJECT_ROOT = tmp_path
        write_run(tmp_path, "a", "swin_t", wide_labels, degraded(wide_labels, rng, 0.05))
        write_run(tmp_path, "b", "densenet121", wide_labels, degraded(wide_labels, rng, 0.15))

        _, scan = select_members_by_diversity(val_floor=0.0, output_root=tmp_path / "outputs")

        for entry in scan:
            assert "_member" not in entry and "_score" not in entry
            json.dumps(entry)


class TestGroupedFolds:
    def test_a_patient_never_straddles_two_folds(self, rng):
        """The whole point of grouping: a fold that contains one study from a
        patient and trains on another has leaked, and the out-of-fold
        predictions it produces are in-sample in the way that matters."""
        groups = np.repeat(np.arange(60).astype(str), 4)
        folds = _grouped_folds(groups, len(groups), 5)

        for patient in np.unique(groups):
            assert len(set(folds[groups == patient])) == 1

    def test_it_uses_every_fold(self, rng):
        groups = np.repeat(np.arange(60).astype(str), 4)
        assert len(set(_grouped_folds(groups, len(groups), 5))) == 5

    def test_it_falls_back_to_rows_without_patient_ids(self):
        folds = _grouped_folds(None, 200, 5)
        assert folds.shape == (200,)
        assert set(folds) <= set(range(5))


class TestStacker:
    def _members(self, labels, rng, count=3):
        return [degraded(labels, rng, 0.1 * (i + 1)) for i in range(count)]

    def test_equal_members_produce_member_shaped_logit_features(self, wide_labels, rng):
        members = self._members(wide_labels, rng)
        features = _member_logits(members, 0)
        assert features.shape == (len(wide_labels), len(members))

    def test_it_skips_classes_with_too_few_positives(self, rng):
        """Hernia has 14 validation positives against six free parameters."""
        labels = np.zeros((300, config.NUM_CLASSES), dtype=np.float32)
        labels[rng.choice(300, 120, replace=False), 0] = 1.0   # plenty
        labels[rng.choice(300, 5, replace=False), 1] = 1.0     # far too few
        members = [rng.random(labels.shape) for _ in range(3)]

        models = fit_stacker(labels, members)

        assert 0 in models
        assert 1 not in models

    def test_unfitted_classes_keep_the_plain_mean(self, rng):
        labels = np.zeros((300, config.NUM_CLASSES), dtype=np.float32)
        labels[rng.choice(300, 120, replace=False), 0] = 1.0
        members = [rng.random(labels.shape) for _ in range(3)]

        combined = apply_stacker(fit_stacker(labels, members), members)

        mean = np.mean(members, axis=0)
        assert combined[:, 1] == pytest.approx(mean[:, 1])
        assert not np.allclose(combined[:, 0], mean[:, 0])

    def test_it_weights_an_informative_member_above_a_useless_one(self, wide_labels, rng):
        """
        The property that makes stacking worth having over the mean.

        Note what is *not* asserted: that the combined scores beat the mean's
        AUROC even in-sample. Logistic regression minimizes log-loss, not
        AUROC, so that is not guaranteed on any particular fit — the learned
        weighting is the thing to pin.
        """
        informative = degraded(wide_labels, rng, 0.05)
        useless = chance(wide_labels, rng)

        models = fit_stacker(wide_labels, [informative, useless])

        weights = models[0].coef_[0]
        assert weights[0] > weights[1]

    def test_it_produces_valid_probabilities(self, wide_labels, rng):
        members = self._members(wide_labels, rng, count=4)
        combined = apply_stacker(fit_stacker(wide_labels, members), members)
        assert combined.min() >= 0.0 and combined.max() <= 1.0


class TestStackedPredictions:
    def _setup(self, rng, rows=600):
        labels = np.zeros((rows, config.NUM_CLASSES), dtype=np.float32)
        for class_idx in range(config.NUM_CLASSES):
            labels[rng.choice(rows, rows // 4, replace=False), class_idx] = 1.0
        members = [degraded(labels, rng, 0.1 * (i + 1)) for i in range(3)]
        groups = np.repeat(np.arange(rows // 2).astype(str), 2)
        return labels, members, groups

    def test_validation_predictions_are_genuinely_out_of_fold(self, rng):
        """
        The guard this exists for.

        A stacker's in-sample validation scores are better than it will ever do
        again; tuning thresholds on them calibrates to a performance the model
        does not have. Out-of-fold predictions must therefore score *worse* than
        the in-sample fit — if they match it, the folds are not holding anything
        out.
        """
        from sklearn.metrics import roc_auc_score

        labels, members, groups = self._setup(rng)
        out_of_fold, _, _ = stacked_predictions(labels, members, members, groups)
        in_sample = apply_stacker(fit_stacker(labels, members), members)

        oof_score = np.mean([roc_auc_score(labels[:, c], out_of_fold[:, c]) for c in range(4)])
        in_sample_score = np.mean([roc_auc_score(labels[:, c], in_sample[:, c]) for c in range(4)])
        assert oof_score < in_sample_score

    def test_it_returns_one_prediction_per_row_of_each_split(self, rng):
        labels, members, groups = self._setup(rng)
        test_members = [probs[:100] for probs in members]

        out_of_fold, test_probs, info = stacked_predictions(
            labels, members, test_members, groups
        )

        assert out_of_fold.shape == labels.shape
        assert test_probs.shape == (100, config.NUM_CLASSES)
        assert info["folds_grouped_by_patient"] is True

    def test_the_report_is_serializable_and_names_the_fallbacks(self, rng):
        labels, members, groups = self._setup(rng)
        labels[:, 3] = 0.0
        labels[rng.choice(len(labels), 5, replace=False), 3] = 1.0

        _, _, info = stacked_predictions(labels, members, members, groups)

        assert config.CLASS_NAMES[3] in info["classes_left_at_mean"]
        assert config.CLASS_NAMES[3] not in info["coefficients"]
        json.dumps(info)


class TestParseMember:
    def test_splits_experiment_from_model(self):
        assert parse_member("s3_swin_384:swin_t") == ("s3_swin_384", "swin_t")

    @pytest.mark.parametrize("spec", ["s3_swin_384", "a:b:c", ":swin_t", "s3_swin_384:"])
    def test_rejects_anything_else(self, spec):
        with pytest.raises(Exception):
            parse_member(spec)


class TestCombine:
    def test_mean_averages_elementwise(self):
        a = np.full((2, 3), 0.2)
        b = np.full((2, 3), 0.4)
        assert combine([a, b], "mean") == pytest.approx(np.full((2, 3), 0.3))

    def test_every_rule_leaves_identical_members_alone(self, labels, rng):
        """Averaging a model with itself must return that model, or the rule is
        doing something other than combining."""
        probs = strong(labels, rng)
        for rule in ("mean", "logit"):
            assert combine([probs, probs], rule) == pytest.approx(probs, abs=1e-5)

    def test_logit_does_not_diverge_on_saturated_probabilities(self):
        saturated = np.array([[0.0, 1.0]])
        ordinary = np.array([[0.5, 0.5]])
        combined = combine([saturated, ordinary], "logit")
        assert np.isfinite(combined).all()
        assert (combined >= 0).all() and (combined <= 1).all()

    def test_rank_rescales_into_the_unit_interval(self, labels, rng):
        combined = combine([strong(labels, rng), chance(labels, rng)], "rank")
        assert combined.min() >= 0.0 and combined.max() <= 1.0

    def test_rejects_an_unknown_rule(self):
        with pytest.raises(ValueError, match="Unknown combination rule"):
            combine([np.zeros((2, 2))], "geometric")


class TestCheckAlignment:
    def _two_runs(self, tmp_path, labels, rng, second_labels=None, second_groups=None):
        config.PROJECT_ROOT = tmp_path
        groups = np.repeat(np.arange(20), 2)
        write_run(tmp_path, "exp_a", "densenet121", labels, strong(labels, rng), groups)
        write_run(
            tmp_path,
            "exp_b",
            "vit_s_16",
            labels if second_labels is None else second_labels,
            chance(labels, rng),
            groups if second_groups is None else second_groups,
        )
        return [("exp_a", "densenet121"), ("exp_b", "vit_s_16")]

    def test_returns_the_shared_arrays_when_rows_match(self, tmp_path, labels, rng):
        members = self._two_runs(tmp_path, labels, rng)
        reference = check_alignment(members, "val")
        assert reference["labels"].shape == labels.shape

    def test_rejects_members_that_scored_different_rows(self, tmp_path, labels, rng):
        """The failure this exists for: same shape, different images. Averaging
        would pair each prediction with another image's label and still produce
        a plausible number."""
        shuffled = labels[::-1].copy()
        members = self._two_runs(tmp_path, labels, rng, second_labels=shuffled)
        with pytest.raises(ValueError, match="different rows"):
            check_alignment(members, "val")

    def test_rejects_members_with_different_patient_ids(self, tmp_path, labels, rng):
        members = self._two_runs(
            tmp_path, labels, rng, second_groups=np.arange(len(labels))
        )
        with pytest.raises(ValueError, match="patient IDs"):
            check_alignment(members, "val")

    def test_rejects_members_from_different_splits(self, tmp_path, labels, rng):
        config.PROJECT_ROOT = tmp_path
        write_run(tmp_path, "exp_a", "densenet121", labels, strong(labels, rng))
        half = labels[:20]
        write_run(tmp_path, "exp_b", "vit_s_16", half, chance(half, rng))
        with pytest.raises(ValueError, match="different splits"):
            check_alignment([("exp_a", "densenet121"), ("exp_b", "vit_s_16")], "val")


class TestDiscoverRuns:
    def test_finds_runs_that_saved_both_splits(self, tmp_path, labels, rng):
        config.PROJECT_ROOT = tmp_path
        write_run(tmp_path, "exp_a", "densenet121", labels, strong(labels, rng))
        assert discover_runs(tmp_path / "outputs") == [("exp_a", "densenet121")]

    def test_skips_a_run_missing_the_validation_half(self, tmp_path, labels, rng):
        config.PROJECT_ROOT = tmp_path
        results = write_run(tmp_path, "exp_a", "densenet121", labels, strong(labels, rng))
        (results / "densenet121_val_predictions.npz").unlink()
        assert discover_runs(tmp_path / "outputs") == []

    def test_never_offers_an_ensemble_as_a_member(self, tmp_path, labels, rng):
        """A second --auto run would otherwise select the first ensemble, make it
        its own baseline, and report a margin of zero over 'the best member'."""
        config.PROJECT_ROOT = tmp_path
        write_run(tmp_path, "exp_a", "densenet121", labels, strong(labels, rng))
        write_run(
            tmp_path, "s5_ensemble", "ensemble", labels, strong(labels, rng), is_ensemble=True
        )
        assert discover_runs(tmp_path / "outputs") == [("exp_a", "densenet121")]

    def test_excludes_the_folder_about_to_be_written(self, tmp_path, labels, rng):
        """Covers an ensemble interrupted before it wrote the results file that
        would otherwise mark it."""
        config.PROJECT_ROOT = tmp_path
        write_run(tmp_path, "exp_a", "densenet121", labels, strong(labels, rng))
        write_run(tmp_path, "s5_ensemble", "ensemble", labels, strong(labels, rng))
        found = discover_runs(tmp_path / "outputs", exclude_experiment="s5_ensemble")
        assert found == [("exp_a", "densenet121")]


class TestIsEnsembleOutput:
    def test_reads_the_marker_off_the_results_file(self, tmp_path, labels, rng):
        results = write_run(
            tmp_path, "s5", "ensemble", labels, strong(labels, rng), is_ensemble=True
        )
        assert is_ensemble_output(results, "ensemble")

    def test_a_plain_run_is_not_one(self, tmp_path, labels, rng):
        results = write_run(tmp_path, "exp_a", "densenet121", labels, strong(labels, rng))
        assert not is_ensemble_output(results, "densenet121")

    def test_unreadable_results_do_not_raise(self, tmp_path, labels, rng):
        results = write_run(tmp_path, "exp_a", "densenet121", labels, strong(labels, rng))
        (results / "densenet121_results.json").write_text("{not json", encoding="utf-8")
        assert not is_ensemble_output(results, "densenet121")


class TestSelectMembers:
    def test_keeps_one_run_per_family_and_drops_the_rest(self, tmp_path, labels, rng):
        config.PROJECT_ROOT = tmp_path
        write_run(tmp_path, "weak_dense", "densenet121", labels, degraded(labels, rng, 0.60))
        write_run(tmp_path, "best_dense", "densenet121", labels, degraded(labels, rng, 0.0))
        write_run(tmp_path, "only_vit", "vit_s_16", labels, degraded(labels, rng, 0.20))

        selected, scan = select_members(val_floor=0.0, output_root=tmp_path / "outputs")

        families = [model for _, model in selected]
        assert sorted(families) == ["densenet121", "vit_s_16"]
        beaten = [e for e in scan if e["status"].startswith("beaten")]
        assert len(beaten) == 1

    def test_reports_the_family_winner_not_whoever_overtook_first(self, tmp_path, labels, rng):
        """Three runs in one family: the two losers must both point at the run
        that actually won, not at each other."""
        config.PROJECT_ROOT = tmp_path
        write_run(tmp_path, "a_worst", "densenet121", labels, degraded(labels, rng, 0.75))
        write_run(tmp_path, "b_middle", "densenet121", labels, degraded(labels, rng, 0.40))
        write_run(tmp_path, "c_best", "densenet121", labels, degraded(labels, rng, 0.0))

        selected, scan = select_members(val_floor=0.0, output_root=tmp_path / "outputs")

        by_experiment = {entry["experiment"]: entry["val_auroc"] for entry in scan}
        assert by_experiment["c_best"] > by_experiment["b_middle"] > by_experiment["a_worst"], (
            "fixture no longer orders the three runs as intended"
        )
        assert selected == [("c_best", "densenet121")]
        for entry in scan:
            if entry["status"].startswith("beaten"):
                assert entry["status"].endswith("c_best")

    def test_drops_a_family_below_the_floor(self, tmp_path, labels, rng):
        config.PROJECT_ROOT = tmp_path
        write_run(tmp_path, "good", "densenet121", labels, strong(labels, rng))
        write_run(tmp_path, "failed", "swin_v2_t", labels, chance(labels, rng))

        selected, scan = select_members(val_floor=0.75, output_root=tmp_path / "outputs")

        assert selected == [("good", "densenet121")]
        dropped = next(e for e in scan if e["experiment"] == "failed")
        assert "below floor" in dropped["status"]

    def test_the_scan_records_every_candidate_considered(self, tmp_path, labels, rng):
        config.PROJECT_ROOT = tmp_path
        write_run(tmp_path, "good", "densenet121", labels, strong(labels, rng))
        write_run(tmp_path, "failed", "swin_v2_t", labels, chance(labels, rng))

        _, scan = select_members(val_floor=0.75, output_root=tmp_path / "outputs")

        assert {e["experiment"] for e in scan} == {"good", "failed"}
        assert all(e["val_auroc"] is not None for e in scan)
