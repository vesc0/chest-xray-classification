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
    apply_thresholds,
    bootstrap_cis,
    compute_metrics,
    patient_groups,
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


class TestTuneThresholds:
    def _separable(self, n=200):
        """Class 0 perfectly separable at 0.5; the rest are noise."""
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
        assert thresholds[1] == pytest.approx(config.DEFAULT_THRESHOLD)

    def test_a_separable_class_gets_tuned(self):
        labels, probs = self._separable()
        thresholds, summary = tune_thresholds(labels, probs)
        entry = summary[config.CLASS_NAMES[0]]
        assert entry["status"] == "tuned"
        assert entry["objective"] == pytest.approx(1.0)
        assert 0.1 < thresholds[0] <= 0.9

    def test_the_tuned_threshold_actually_separates(self):
        labels, probs = self._separable()
        thresholds, _ = tune_thresholds(labels, probs)
        preds = apply_thresholds(probs, thresholds)
        assert (preds[:, 0] == labels[:, 0]).all()

    def test_every_threshold_stays_inside_the_configured_grid(self, rng):
        labels = (rng.random((200, config.NUM_CLASSES)) < 0.3).astype(np.float32)
        probs = rng.random((200, config.NUM_CLASSES)).astype(np.float32)
        thresholds, _ = tune_thresholds(labels, probs)
        assert (thresholds >= config.THRESHOLD_MIN).all()
        assert (thresholds <= config.THRESHOLD_MAX).all()

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

    def test_rejects_an_unknown_threshold_metric(self):
        labels, probs = self._separable()
        config.THRESHOLD_METRIC = "nonsense"
        with pytest.raises(ValueError, match="Unsupported threshold metric"):
            tune_thresholds(labels, probs)


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
