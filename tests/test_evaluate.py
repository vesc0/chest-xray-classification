"""
Thresholding, calibration, the derived screening metric, and the bootstrap.

Nothing here touches a model or the disk: every function under test takes
label/probability arrays. The one exception is the patient-grouping helper,
which needs a DataLoader to read its sampler — that loader wraps an in-memory
frame, not the dataset.
"""

import importlib.metadata
import json

import numpy as np
import pandas as pd
import pytest
from torch.utils.data import DataLoader

import config
from evaluate import (
    _PROVENANCE_PACKAGES,
    _environment_versions,
    _expected_calibration_error,
    _normal_vs_abnormal_metrics,
    _rows_by_group,
    _thresholded_rates,
    apply_thresholds,
    bootstrap_cis,
    compute_metrics,
    confusion_sweep,
    load_predictions,
    patient_groups,
    save_predictions,
    threshold_settings,
    tune_thresholds,
)


class TestApplyThresholds:
    def test_uses_the_default_when_none_given(self):
        probs = np.array([[0.4, 0.6]])
        assert apply_thresholds(probs, None).tolist() == [[0, 1]]

    def test_per_class_thresholds_apply_independently(self):
        probs = np.array([[0.4, 0.4]])
        thresholds = np.array([0.3, 0.5])
        assert apply_thresholds(probs, thresholds).tolist() == [[1, 0]]

    def test_boundary_is_inclusive(self):
        assert apply_thresholds(np.array([[0.5]]), 0.5).tolist() == [[1]]


class TestConfusionSweep:
    """
    The candidate set every threshold decision is made over. If the counts are
    wrong the objective is wrong, silently and for every class at once.
    """

    def test_counts_match_a_brute_force_scan(self, rng):
        y_true = (rng.random(200) < 0.3).astype(np.float32)
        y_prob = rng.random(200).round(2).astype(np.float32)  # forces ties

        thresholds, tp, fp, fn, tn = confusion_sweep(y_true, y_prob)
        for index, threshold in enumerate(thresholds):
            preds = y_prob >= threshold
            assert tp[index] == np.logical_and(preds, y_true).sum()
            assert fp[index] == np.logical_and(preds, 1 - y_true).sum()
            assert fn[index] == np.logical_and(~preds, y_true).sum()
            assert tn[index] == np.logical_and(~preds, 1 - y_true).sum()

    def test_candidates_are_the_distinct_scores_in_descending_order(self, rng):
        y_prob = rng.random(100).round(1).astype(np.float32)
        thresholds, *_ = confusion_sweep(np.zeros(100, dtype=np.float32), y_prob)
        assert thresholds.tolist() == sorted(set(y_prob.tolist()), reverse=True)

    def test_the_last_candidate_predicts_everything_positive(self, rng):
        """Why an F1 optimum can never be an all-negative rule."""
        y_true = (rng.random(150) < 0.2).astype(np.float32)
        y_prob = rng.random(150).astype(np.float32)
        _, tp, _, fn, _ = confusion_sweep(y_true, y_prob)
        assert fn[-1] == 0
        assert tp[-1] == y_true.sum()


class TestTuneThresholds:
    def _separable(self, n=200):
        """Class 0 perfectly separable at 0.9; the rest are noise."""
        labels = np.zeros((n, config.NUM_CLASSES), dtype=np.float32)
        probs = np.full((n, config.NUM_CLASSES), 0.5, dtype=np.float32)
        labels[: n // 2, 0] = 1.0
        probs[: n // 2, 0] = 0.9
        probs[n // 2 :, 0] = 0.1
        return labels, probs

    def test_low_support_classes_fall_back_to_the_default(self):
        labels = np.zeros((100, config.NUM_CLASSES), dtype=np.float32)
        labels[0, 1] = 1.0  # a single positive, below THRESHOLD_MIN_SUPPORT
        probs = np.full((100, config.NUM_CLASSES), 0.5, dtype=np.float32)

        thresholds, summary = tune_thresholds(labels, probs)
        entry = summary[config.CLASS_NAMES[1]]
        assert entry["status"] == "default"
        assert entry["support"] == 1
        assert "THRESHOLD_MIN_SUPPORT" in entry["reason"]
        assert thresholds[1] == pytest.approx(config.DEFAULT_THRESHOLD)

    def test_a_separable_class_gets_tuned(self):
        labels, probs = self._separable()
        thresholds, summary = tune_thresholds(labels, probs)
        entry = summary[config.CLASS_NAMES[0]]
        assert entry["status"] == "tuned"
        assert entry["objective"] == pytest.approx(1.0)
        assert thresholds[0] == pytest.approx(0.9)

    def test_the_tuned_threshold_actually_separates(self):
        labels, probs = self._separable()
        thresholds, _ = tune_thresholds(labels, probs)
        preds = apply_thresholds(probs, thresholds)
        assert (preds[:, 0] == labels[:, 0]).all()

    def test_every_threshold_is_a_score_the_model_actually_produced(self, rng):
        """
        Candidates come from the predictions, not a grid, so each threshold is
        reachable and reproduces the counts it was chosen for.
        """
        labels = (rng.random((400, config.NUM_CLASSES)) < 0.3).astype(np.float32)
        probs = rng.random((400, config.NUM_CLASSES)).astype(np.float32)
        thresholds, summary = tune_thresholds(labels, probs)

        for class_idx, class_name in enumerate(config.CLASS_NAMES):
            if summary[class_name]["status"] != "tuned":
                continue
            # Compared against the rounded value the summary reports, since the
            # stored threshold is one of the raw float32 scores
            assert np.isclose(probs[:, class_idx], thresholds[class_idx]).any()

    def test_a_rare_class_scored_far_below_the_old_grid_floor_still_tunes(self):
        """
        The regression this whole change exists for.

        A grid over [0.05, 0.95] scores zero at every point when a class's
        scores all sit below 0.05, and returns 0.05 with all-negative
        predictions labelled "tuned". The candidate set cannot do that: the
        threshold has to be a score the model produced.
        """
        rng = np.random.default_rng(0)
        labels = np.zeros((600, config.NUM_CLASSES), dtype=np.float32)
        probs = np.full((600, config.NUM_CLASSES), 0.5, dtype=np.float32)

        # 60 positives, every score for the class crushed below 0.05
        labels[:60, 0] = 1.0
        probs[:60, 0] = rng.uniform(0.02, 0.04, 60).astype(np.float32)
        probs[60:, 0] = rng.uniform(0.0, 0.02, 540).astype(np.float32)

        thresholds, summary = tune_thresholds(labels, probs)
        entry = summary[config.CLASS_NAMES[0]]

        assert entry["status"] == "tuned"
        assert thresholds[0] < 0.05
        assert entry["objective"] > 0.5
        # And the decisive part: it does not predict the class negative everywhere
        assert apply_thresholds(probs, thresholds)[:, 0].sum() > 0

    def test_reports_one_entry_per_class(self, rng):
        labels = (rng.random((80, config.NUM_CLASSES)) < 0.3).astype(np.float32)
        probs = rng.random((80, config.NUM_CLASSES)).astype(np.float32)
        _, summary = tune_thresholds(labels, probs)
        assert list(summary) == list(config.CLASS_NAMES)

    def test_respects_an_overridden_threshold_metric(self):
        labels, probs = self._separable()
        config.THRESHOLD_METRIC = "youden"
        _, summary = tune_thresholds(labels, probs)
        assert summary[config.CLASS_NAMES[0]]["status"] == "tuned"

    def test_youden_reports_an_unrankable_class_as_degenerate(self):
        """
        Youden can legitimately bottom out where F1 cannot. Saying so beats
        returning a threshold that separates nothing.
        """
        config.THRESHOLD_METRIC = "youden"
        labels = np.zeros((400, config.NUM_CLASSES), dtype=np.float32)
        labels[:100, 0] = 1.0
        probs = np.full((400, config.NUM_CLASSES), 0.3, dtype=np.float32)

        thresholds, summary = tune_thresholds(labels, probs)
        assert summary[config.CLASS_NAMES[0]]["status"] == "degenerate"
        assert thresholds[0] == pytest.approx(config.DEFAULT_THRESHOLD)

    def test_sensitivity_mode_reaches_its_target_recall(self, rng):
        config.THRESHOLD_METRIC = "sensitivity"
        config.THRESHOLD_TARGET_SENSITIVITY = 0.95

        labels = (rng.random((800, config.NUM_CLASSES)) < 0.25).astype(np.float32)
        probs = (labels * 0.4 + rng.random((800, config.NUM_CLASSES)) * 0.6).astype(np.float32)

        thresholds, summary = tune_thresholds(labels, probs)
        preds = apply_thresholds(probs, thresholds)

        for class_idx, class_name in enumerate(config.CLASS_NAMES):
            if summary[class_name]["status"] != "tuned":
                continue
            recall = preds[:, class_idx][labels[:, class_idx] == 1].mean()
            assert recall >= 0.95

    def test_sensitivity_mode_buys_recall_with_specificity(self, rng):
        """The trade the fixed-sensitivity operating point exists to make."""
        labels = (rng.random((800, config.NUM_CLASSES)) < 0.25).astype(np.float32)
        probs = (labels * 0.4 + rng.random((800, config.NUM_CLASSES)) * 0.6).astype(np.float32)

        config.THRESHOLD_METRIC = "f1"
        by_f1, _ = tune_thresholds(labels, probs)
        config.THRESHOLD_METRIC = "sensitivity"
        config.THRESHOLD_TARGET_SENSITIVITY = 0.95
        by_sensitivity, _ = tune_thresholds(labels, probs)

        # A higher recall floor can only be met by a looser threshold
        assert (by_sensitivity <= by_f1 + 1e-6).all()

    def test_rejects_an_unknown_threshold_metric(self):
        labels, probs = self._separable()
        config.THRESHOLD_METRIC = "nonsense"
        with pytest.raises(ValueError, match="Unsupported threshold metric"):
            tune_thresholds(labels, probs)

    def test_rejects_an_unknown_metric_even_when_no_class_is_tunable(self):
        """Fails fast on the config, not on whichever class happens to be first."""
        config.THRESHOLD_METRIC = "nonsense"
        labels = np.zeros((50, config.NUM_CLASSES), dtype=np.float32)
        probs = np.full((50, config.NUM_CLASSES), 0.5, dtype=np.float32)
        with pytest.raises(ValueError, match="Unsupported threshold metric"):
            tune_thresholds(labels, probs)


class TestThresholdSettings:
    """
    Provenance for the thresholds file. Reproducing an operating point needs the
    support cutoff and the objective's own parameters, not just its name.
    """

    def test_records_the_support_cutoff(self):
        assert threshold_settings()["min_support"] == config.THRESHOLD_MIN_SUPPORT

    def test_beta_travels_with_fbeta_only(self):
        config.THRESHOLD_METRIC = "f1"
        assert "beta" not in threshold_settings()
        config.THRESHOLD_METRIC = "fbeta"
        assert threshold_settings()["beta"] == config.THRESHOLD_BETA

    def test_target_sensitivity_travels_with_sensitivity_only(self):
        config.THRESHOLD_METRIC = "f1"
        assert "target_sensitivity" not in threshold_settings()
        config.THRESHOLD_METRIC = "sensitivity"
        assert threshold_settings()["target_sensitivity"] == (
            config.THRESHOLD_TARGET_SENSITIVITY
        )


class TestSavePredictions:
    """
    The arrays that make every later thresholding decision a script rather than
    another inference pass over five models.
    """

    def _arrays(self, rng, rows=40):
        labels = (rng.random((rows, config.NUM_CLASSES)) < 0.3).astype(np.float32)
        probs = rng.random((rows, config.NUM_CLASSES)).astype(np.float32)
        return labels, probs

    def test_round_trips_labels_probabilities_and_patients(self, rng, tmp_path):
        config.RESULTS_DIR = tmp_path
        labels, probs = self._arrays(rng)
        groups = np.repeat(np.arange(20), 2).astype(str)

        save_predictions(labels, probs, "densenet121", "val", groups=groups)
        restored = load_predictions("densenet121", "val")

        assert (restored["labels"] == labels.astype(np.int8)).all()
        assert restored["probs"] == pytest.approx(probs)
        assert list(restored["class_names"]) == list(config.CLASS_NAMES)
        assert (restored["groups"] == groups).all()

    def test_probabilities_survive_the_round_trip_exactly(self, rng, tmp_path):
        """Thresholds are compared against these values with >=, so a lossy
        round trip would shift which samples land on the positive side."""
        config.RESULTS_DIR = tmp_path
        labels, probs = self._arrays(rng)
        save_predictions(labels, probs, "densenet121", "test")
        assert (load_predictions("densenet121", "test")["probs"] == probs).all()

    def test_writes_nothing_when_disabled(self, rng, tmp_path):
        config.RESULTS_DIR = tmp_path
        config.SAVE_PREDICTIONS = False
        labels, probs = self._arrays(rng)
        assert save_predictions(labels, probs, "densenet121", "val") is None
        assert list(tmp_path.glob("*.npz")) == []

    def test_a_missing_file_says_how_to_produce_it(self, tmp_path):
        config.RESULTS_DIR = tmp_path
        with pytest.raises(FileNotFoundError, match="SAVE_PREDICTIONS"):
            load_predictions("densenet121", "val")


class TestExpectedCalibrationError:
    def test_perfectly_calibrated_scores_zero(self):
        # Half the samples positive, all predicted 0.5: confidence == accuracy
        labels = np.array([0.0, 1.0] * 50)
        probs = np.full(100, 0.5)
        assert _expected_calibration_error(labels, probs) == pytest.approx(0.0, abs=1e-9)

    def test_confidently_wrong_scores_one(self):
        assert _expected_calibration_error(np.zeros(50), np.ones(50)) == pytest.approx(1.0)

    def test_confidently_right_scores_zero(self):
        assert _expected_calibration_error(np.ones(50), np.ones(50)) == pytest.approx(0.0)

    def test_the_top_bin_is_closed_so_probability_one_is_counted(self):
        # Bin edges are half-open except the last; a p=1.0 sample must not be
        # dropped, or ECE would be computed over fewer samples than exist.
        assert _expected_calibration_error(np.zeros(10), np.ones(10)) > 0.0

    def test_every_prediction_lands_in_exactly_one_bin(self, rng):
        """A dropped sample would quietly shrink the denominator."""
        probs = rng.random(500)
        labels = (rng.random(500) < probs).astype(float)
        for strategy in ("quantile", "uniform"):
            # Perfectly miscalibrated: |confidence - accuracy| is 1 in every
            # bin, so the weighted sum equals the fraction of samples counted.
            assert _expected_calibration_error(
                np.zeros(500), np.ones(500), strategy=strategy
            ) == pytest.approx(1.0)
            assert 0.0 <= _expected_calibration_error(labels, probs, strategy=strategy) <= 1.0

    def test_uniform_bins_hide_miscalibration_in_the_rare_class_mass(self):
        """
        The reason quantile is the default.

        Every prediction is tiny, as it is for a 0.16%-prevalence label, and the
        model is badly calibrated *within* that mass — confident-ish scores are
        no likelier to be positive than near-zero ones. Uniform bins put the lot
        in bin 1 and average the error away; quantile bins split it and see it.
        """
        low = np.linspace(0.001, 0.06, 500)
        # Positives concentrated at the *bottom* of the range: ranking inverted
        labels = (np.arange(500) < 100).astype(float)

        uniform = _expected_calibration_error(labels, low, strategy="uniform")
        quantile = _expected_calibration_error(labels, low, strategy="quantile")
        assert quantile > uniform

    def test_reads_the_bin_count_at_call_time(self):
        """
        Config is read when the function runs, not when the module is imported.
        Bound as a default argument, a runtime override would never arrive.
        """
        # Four score groups; coarse bins merge them in pairs whose errors
        # partially cancel, so the granularity has to change the answer.
        probs = np.repeat([0.1, 0.3, 0.7, 0.9], 100)
        labels = np.repeat([0.0, 1.0, 0.0, 1.0], 100)

        config.ECE_BINS = 2
        coarse = _expected_calibration_error(labels, probs)
        config.ECE_BINS = 8
        fine = _expected_calibration_error(labels, probs)

        assert coarse == pytest.approx(0.3)
        assert fine == pytest.approx(0.4)

    def test_rejects_an_unknown_bin_strategy(self):
        with pytest.raises(ValueError, match="Unsupported ECE bin strategy"):
            _expected_calibration_error(np.zeros(10), np.ones(10), strategy="nonsense")


class TestNormalVsAbnormal:
    def test_prevalence_counts_rows_with_any_finding(self):
        labels = np.zeros((10, config.NUM_CLASSES), dtype=np.float32)
        labels[:3, 0] = 1.0  # 3 abnormal studies
        probs = np.zeros((10, config.NUM_CLASSES), dtype=np.float32)
        assert _normal_vs_abnormal_metrics(labels, probs)["prevalence"] == pytest.approx(0.3)

    def test_perfect_separation_scores_one_both_ways(self):
        labels = np.zeros((20, config.NUM_CLASSES), dtype=np.float32)
        probs = np.full((20, config.NUM_CLASSES), 0.01, dtype=np.float32)
        labels[:10, 3] = 1.0
        probs[:10, 3] = 0.99

        result = _normal_vs_abnormal_metrics(labels, probs)
        assert result["max_auroc"] == pytest.approx(1.0)
        assert result["noisy_or_auroc"] == pytest.approx(1.0)

    def test_reports_none_when_every_study_is_normal(self):
        labels = np.zeros((10, config.NUM_CLASSES), dtype=np.float32)
        probs = np.full((10, config.NUM_CLASSES), 0.2, dtype=np.float32)
        result = _normal_vs_abnormal_metrics(labels, probs)
        assert result["max_auroc"] is None


class TestEnvironmentVersions:
    """
    Provenance stamped into every results file.

    Two results files months apart are only comparable if you can see they were
    produced under the same libraries.
    """

    def test_reports_python_and_every_listed_package(self):
        versions = _environment_versions()
        assert "python" in versions
        assert set(_PROVENANCE_PACKAGES) <= set(versions)

    def test_records_the_versions_actually_installed(self):
        import importlib.metadata

        assert _environment_versions()["torch"] == importlib.metadata.version("torch")

    def test_torchvision_is_recorded(self):
        # The one that decides which pretrained weights `.DEFAULT` resolves to
        assert _environment_versions()["torchvision"] is not None

    def test_reports_none_for_a_missing_package_instead_of_raising(self, monkeypatch):
        """A finished training run must not be lost to a provenance lookup."""

        def explode(name):
            raise importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(importlib.metadata, "version", explode)
        versions = _environment_versions()
        assert versions["python"]  # still filled in
        assert all(versions[package] is None for package in _PROVENANCE_PACKAGES)

    def test_is_json_serializable(self):
        # It gets written straight into the results file
        json.dumps(_environment_versions())


class TestBootstrapConfidenceIntervals:
    """
    The intervals are the difference between five numbers and five claims. Their
    failure mode is not a crash — it is an interval that looks authoritative and
    is quietly too narrow.
    """

    @staticmethod
    def _scored(rng, rows=800, groups_of=1):
        """Labels and probabilities carrying real but imperfect signal."""
        labels = (
            rng.random((rows, config.NUM_CLASSES)) < np.linspace(0.35, 0.02, config.NUM_CLASSES)
        ).astype(np.float32)
        probs = (labels * 0.4 + rng.random((rows, config.NUM_CLASSES)) * 0.6).astype(np.float32)
        groups = np.repeat(np.arange(rows // groups_of), groups_of)
        return labels, probs, groups

    def test_absent_when_disabled(self, rng):
        labels, probs, _ = self._scored(rng)
        config.BOOTSTRAP_ENABLED = False
        results = compute_metrics(labels, probs, apply_thresholds(probs, 0.5))
        assert results["bootstrap"] is None
        assert "auroc_ci" not in results["macro"]

    def test_present_when_enabled(self, rng):
        labels, probs, _ = self._scored(rng)
        config.BOOTSTRAP_ENABLED = True
        config.BOOTSTRAP_SAMPLES = 40
        results = compute_metrics(labels, probs, apply_thresholds(probs, 0.5))

        assert results["bootstrap"]["samples"] == 40
        low, high = results["macro"]["auroc_ci"]
        assert low < high

    def test_the_point_estimate_falls_inside_its_own_interval(self, rng):
        labels, probs, _ = self._scored(rng)
        config.BOOTSTRAP_ENABLED = True
        config.BOOTSTRAP_SAMPLES = 60
        results = compute_metrics(labels, probs, apply_thresholds(probs, 0.5))

        for metric in ("auroc", "auprc"):
            low, high = results["macro"][f"{metric}_ci"]
            assert low <= results["macro"][metric] <= high

    def test_patient_grouping_widens_the_interval(self, rng):
        """
        The reason grouping exists. Several studies from one patient are
        correlated; resampling images pretends they are independent, and returns
        an interval narrower than the evidence supports.
        """
        config.BOOTSTRAP_SAMPLES = 120
        labels, probs, groups = self._scored(rng, rows=600, groups_of=6)

        # Make studies from one patient near-identical, which is what the real
        # correlation looks like and what image-level resampling ignores.
        for start in range(0, len(labels), 6):
            labels[start:start + 6] = labels[start]
            probs[start:start + 6] = probs[start]

        by_image = bootstrap_cis(labels, probs, groups=None)["macro"]["auroc"]
        by_patient = bootstrap_cis(labels, probs, groups=groups)["macro"]["auroc"]

        assert (by_patient[1] - by_patient[0]) > (by_image[1] - by_image[0])

    def test_thresholded_metrics_get_intervals_when_predictions_are_given(self, rng):
        """
        The threshold is frozen on validation before test is touched, so
        resampling test rows at that fixed operating point is an ordinary
        sampling interval — and F1 on the rare classes is the noisiest number
        in the whole table.
        """
        config.BOOTSTRAP_ENABLED = True
        config.BOOTSTRAP_SAMPLES = 60
        labels, probs, _ = self._scored(rng, rows=1000)
        preds = apply_thresholds(probs, 0.5)
        results = compute_metrics(labels, probs, preds)

        assert results["bootstrap"]["includes_thresholded_metrics"] is True
        for metric in ("precision", "recall", "f1"):
            low, high = results["macro"][f"{metric}_ci"]
            assert low <= results["macro"][metric] <= high

        entry = results["per_class"][config.CLASS_NAMES[0]]
        assert entry["f1_ci"][0] <= entry["f1"] <= entry["f1_ci"][1]

    def test_rare_classes_get_relatively_wider_f1_intervals(self, rng):
        """
        Compared relative to the estimate, not in absolute width: an F1 near
        zero is squeezed against the floor, so the rare class can hold a
        narrower absolute interval while being far less determined. Relative
        width is what "we do not really know Hernia's F1" actually means.
        """
        config.BOOTSTRAP_ENABLED = True
        config.BOOTSTRAP_SAMPLES = 80
        labels, probs, _ = self._scored(rng, rows=1500)
        results = compute_metrics(labels, probs, apply_thresholds(probs, 0.3))

        def relative_width(entry):
            low, high = entry["f1_ci"]
            return (high - low) / max(entry["f1"], 1e-9)

        common = results["per_class"][config.CLASS_NAMES[0]]
        rare = results["per_class"][config.CLASS_NAMES[-1]]
        assert rare["support"] < common["support"]
        assert relative_width(rare) > 3 * relative_width(common)

    def test_omitting_predictions_leaves_the_thresholded_metrics_alone(self, rng):
        config.BOOTSTRAP_SAMPLES = 20
        labels, probs, _ = self._scored(rng)
        intervals = bootstrap_cis(labels, probs)
        assert intervals["settings"]["includes_thresholded_metrics"] is False
        assert "f1" not in intervals["macro"]

    def test_rare_classes_get_wider_intervals(self, rng):
        """A class resting on a handful of positives should say so."""
        config.BOOTSTRAP_ENABLED = True
        config.BOOTSTRAP_SAMPLES = 80
        labels, probs, _ = self._scored(rng, rows=1200)
        results = compute_metrics(labels, probs, apply_thresholds(probs, 0.5))

        common = results["per_class"][config.CLASS_NAMES[0]]
        rare = results["per_class"][config.CLASS_NAMES[-1]]
        assert rare["support"] < common["support"]
        assert (rare["auroc_ci"][1] - rare["auroc_ci"][0]) > (
            common["auroc_ci"][1] - common["auroc_ci"][0]
        )

    def test_an_unscorable_class_reports_no_interval(self):
        """Better nothing than an interval built from a few lucky resamples."""
        config.BOOTSTRAP_SAMPLES = 30
        labels = np.zeros((60, config.NUM_CLASSES), dtype=np.float32)
        probs = np.full((60, config.NUM_CLASSES), 0.3, dtype=np.float32)
        intervals = bootstrap_cis(labels, probs)
        assert intervals["per_class"][config.CLASS_NAMES[0]]["auroc"] is None
        assert intervals["macro"]["auroc"] is None

    def test_is_reproducible(self, rng):
        config.BOOTSTRAP_SAMPLES = 40
        labels, probs, groups = self._scored(rng)
        first = bootstrap_cis(labels, probs, groups=groups)
        second = bootstrap_cis(labels, probs, groups=groups)
        assert first == second

    def test_results_stay_json_serializable(self, rng):
        config.BOOTSTRAP_ENABLED = True
        config.BOOTSTRAP_SAMPLES = 30
        labels, probs, groups = self._scored(rng)
        results = compute_metrics(labels, probs, apply_thresholds(probs, 0.5), groups=groups)
        json.dumps(results)


class TestPatientGroups:
    class _Frame:
        def __init__(self, frame):
            self.df = frame

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            return idx

    def _loader(self, shuffle):
        frame = pd.DataFrame({config.PATIENT_ID_COLUMN: ["a", "a", "b", "c"]})
        return DataLoader(self._Frame(frame), batch_size=2, shuffle=shuffle)

    def test_reads_patient_ids_in_loader_order(self):
        assert list(patient_groups(self._loader(shuffle=False))) == ["a", "a", "b", "c"]

    def test_refuses_a_shuffled_loader(self):
        """
        Predictions and patient IDs are matched by position, so a shuffled
        loader would pair each prediction with the wrong patient and produce a
        confident, wrong interval.
        """
        with pytest.raises(ValueError, match="unshuffled"):
            patient_groups(self._loader(shuffle=True))

    def test_reports_a_missing_patient_column(self):
        loader = DataLoader(self._Frame(pd.DataFrame({"other": [1, 2]})), batch_size=1)
        with pytest.raises(KeyError):
            patient_groups(loader)


class TestComputeMetrics:
    def test_records_the_class_set_for_cross_run_comparison(self, rng):
        labels = (rng.random((50, config.NUM_CLASSES)) < 0.3).astype(np.float32)
        probs = rng.random((50, config.NUM_CLASSES)).astype(np.float32)
        results = compute_metrics(labels, probs, apply_thresholds(probs, 0.5))
        assert results["class_names"] == list(config.CLASS_NAMES)
        assert results["num_classes"] == config.NUM_CLASSES

    def test_perfect_predictions_give_perfect_scores(self, rng):
        labels = (rng.random((60, config.NUM_CLASSES)) < 0.4).astype(np.float32)
        results = compute_metrics(labels, labels, labels.astype(np.int32))
        assert results["macro"]["f1"] == pytest.approx(1.0)
        assert results["macro"]["hamming_loss"] == pytest.approx(0.0)
        assert results["macro"]["subset_accuracy"] == pytest.approx(1.0)

    def test_a_scalar_threshold_is_broadcast_to_every_class(self, rng):
        labels = (rng.random((40, config.NUM_CLASSES)) < 0.3).astype(np.float32)
        probs = rng.random((40, config.NUM_CLASSES)).astype(np.float32)
        results = compute_metrics(labels, probs, apply_thresholds(probs, 0.4), thresholds=0.4)
        assert set(results["thresholds"].values()) == {0.4}

    def test_undefined_classes_report_none_not_zero(self):
        labels = np.zeros((30, config.NUM_CLASSES), dtype=np.float32)
        probs = np.full((30, config.NUM_CLASSES), 0.3, dtype=np.float32)
        results = compute_metrics(labels, probs, apply_thresholds(probs, 0.5))
        assert results["per_class"][config.CLASS_NAMES[0]]["auroc"] is None
        assert results["macro"]["auroc"] is None

    def test_all_negative_baseline_is_reported_next_to_sample_accuracy(self, rng):
        # Sample accuracy looks impressive on sparse labels; the baseline is
        # what makes it readable.
        labels = (rng.random((100, config.NUM_CLASSES)) < 0.05).astype(np.float32)
        probs = rng.random((100, config.NUM_CLASSES)).astype(np.float32)
        results = compute_metrics(labels, probs, apply_thresholds(probs, 0.5))
        assert 0.9 < results["macro"]["all_negative_sample_accuracy_baseline"] <= 1.0

    def test_samples_f1_credits_a_correctly_predicted_normal_study(self):
        """
        38.5% of the official test split has no findings at all. Under
        sklearn's zero_division=0 those rows score 0.0 when predicted
        correctly, which caps a *perfect* model at 0.615 and makes the metric
        unreadable next to the others.
        """
        labels = np.zeros((10, config.NUM_CLASSES), dtype=np.float32)
        labels[:4, 0] = 1.0  # 6 of 10 studies are normal

        results = compute_metrics(labels, labels, labels.astype(np.int32))
        assert results["macro"]["samples_f1"] == pytest.approx(1.0)
        assert results["macro"]["all_negative_samples_f1_baseline"] == pytest.approx(0.6)

    def test_the_free_score_from_normal_studies_is_reported_as_a_baseline(self):
        """The cost of zero_division=1: an all-negative model collects those
        rows. Stating the baseline is what keeps the metric honest."""
        labels = np.zeros((10, config.NUM_CLASSES), dtype=np.float32)
        labels[:4, 0] = 1.0
        all_negative = np.zeros((10, config.NUM_CLASSES), dtype=np.int32)

        results = compute_metrics(labels, all_negative.astype(np.float32), all_negative)
        assert results["macro"]["samples_f1"] == pytest.approx(
            results["macro"]["all_negative_samples_f1_baseline"]
        )

    def test_calibration_records_the_binning_that_produced_it(self, rng):
        labels = (rng.random((50, config.NUM_CLASSES)) < 0.3).astype(np.float32)
        probs = rng.random((50, config.NUM_CLASSES)).astype(np.float32)
        results = compute_metrics(labels, probs, apply_thresholds(probs, 0.5))
        assert results["calibration"]["ece_bins"] == config.ECE_BINS
        assert results["calibration"]["ece_bin_strategy"] == config.ECE_BIN_STRATEGY


class TestThresholdedRates:
    """The vectorized confusion counts the bootstrap runs a thousand times."""

    def test_matches_sklearn(self, rng):
        from sklearn.metrics import precision_score, recall_score, f1_score

        labels = (rng.random((200, config.NUM_CLASSES)) < 0.3).astype(np.float32)
        preds = (rng.random((200, config.NUM_CLASSES)) < 0.3).astype(np.int32)
        precision, recall, f1 = _thresholded_rates(labels, preds)

        assert precision == pytest.approx(
            precision_score(labels, preds, average=None, zero_division=0)
        )
        assert recall == pytest.approx(
            recall_score(labels, preds, average=None, zero_division=0)
        )
        assert f1 == pytest.approx(f1_score(labels, preds, average=None, zero_division=0))

    def test_a_class_predicted_never_scores_zero_not_nan(self):
        labels = np.ones((5, 1), dtype=np.float32)
        preds = np.zeros((5, 1), dtype=np.int32)
        assert all(np.isfinite(rate).all() for rate in _thresholded_rates(labels, preds))


class TestRowsByGroup:
    def test_partitions_every_row_exactly_once(self, rng):
        groups = rng.integers(0, 40, 500).astype(str)
        partition = _rows_by_group(groups)
        assert sorted(np.concatenate(partition).tolist()) == list(range(500))

    def test_each_block_holds_one_group(self, rng):
        groups = rng.integers(0, 40, 500).astype(str)
        for rows in _rows_by_group(groups):
            assert len(set(groups[rows])) == 1
