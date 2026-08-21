"""
Shared ranking metrics.

The NaN convention is the load-bearing part: a class nobody could score has to
stay distinguishable from one scored at chance, or it silently drags every
macro average toward 0.5.
"""

import numpy as np
import pytest

import config
from metrics import (
    class_auprc,
    class_auroc,
    epoch_metrics,
    has_both_labels,
    macro_average,
    per_class_auroc,
)


class TestHasBothLabels:
    def test_true_when_mixed(self):
        assert has_both_labels(np.array([0.0, 1.0, 0.0]))

    def test_false_when_all_negative(self):
        assert not has_both_labels(np.zeros(5))

    def test_false_when_all_positive(self):
        assert not has_both_labels(np.ones(5))

    def test_false_when_empty(self):
        assert not has_both_labels(np.array([]))


class TestClassAuroc:
    def test_perfect_ranking_scores_one(self):
        targets = np.array([0.0, 0.0, 1.0, 1.0])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        assert class_auroc(targets, scores) == pytest.approx(1.0)

    def test_inverted_ranking_scores_zero(self):
        targets = np.array([0.0, 0.0, 1.0, 1.0])
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        assert class_auroc(targets, scores) == pytest.approx(0.0)

    @pytest.mark.parametrize("targets", [np.zeros(4), np.ones(4)])
    def test_nan_when_only_one_outcome_present(self, targets):
        # Not 0.0 and not 0.5 — undefined, so macro averaging must skip it.
        assert np.isnan(class_auroc(targets, np.array([0.1, 0.4, 0.6, 0.9])))


class TestClassAuprc:
    def test_nan_when_no_positives(self):
        assert np.isnan(class_auprc(np.zeros(4), np.array([0.1, 0.4, 0.6, 0.9])))

    def test_defined_when_no_negatives(self):
        # AUROC is undefined here but average precision is not.
        assert class_auprc(np.ones(4), np.array([0.1, 0.4, 0.6, 0.9])) == pytest.approx(1.0)

    def test_perfect_ranking_scores_one(self):
        targets = np.array([0.0, 0.0, 1.0, 1.0])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        assert class_auprc(targets, scores) == pytest.approx(1.0)


class TestMacroAverage:
    def test_skips_undefined_classes(self):
        # The mean of the two real values, not of four with NaNs as 0.
        assert macro_average(np.array([0.8, np.nan, 0.6, np.nan])) == pytest.approx(0.7)

    def test_none_when_nothing_scorable(self):
        assert macro_average(np.array([np.nan, np.nan])) is None

    def test_none_when_empty(self):
        assert macro_average(np.array([])) is None


class TestEpochMetrics:
    def test_falls_back_to_zero_not_none(self, rng):
        # Feeds the history arrays behind early stopping, which reject None.
        labels = np.zeros((10, config.NUM_CLASSES), dtype=np.float32)
        result = epoch_metrics(labels, rng.random((10, config.NUM_CLASSES)))
        assert result == {"auroc": 0.0, "auprc": 0.0}

    def test_perfect_predictions_score_one(self, rng):
        labels = (rng.random((60, config.NUM_CLASSES)) < 0.5).astype(np.float32)
        assert epoch_metrics(labels, labels)["auroc"] == pytest.approx(1.0)


class TestTrainEvaluationAgreement:
    """
    Guards the duplication these helpers replaced: train.py and evaluate.py
    each carried their own AUROC loop, so a fix to one left the other behind.
    """

    def test_epoch_auroc_matches_evaluate_macro_auroc(self, rng):
        import evaluate

        labels = (rng.random((300, config.NUM_CLASSES)) < 0.2).astype(np.float32)
        probs = rng.random((300, config.NUM_CLASSES)).astype(np.float32)
        labels[:, 2] = 0.0  # undefined class, skipped by both paths

        from_training = epoch_metrics(labels, probs)
        preds = evaluate.apply_thresholds(probs, 0.5)
        from_evaluation = evaluate.compute_metrics(labels, probs, preds)["macro"]

        assert round(from_training["auroc"], 4) == from_evaluation["auroc"]
        assert round(from_training["auprc"], 4) == from_evaluation["auprc"]

    def test_undefined_class_excluded_from_both(self, rng):
        labels = (rng.random((80, config.NUM_CLASSES)) < 0.3).astype(np.float32)
        probs = rng.random((80, config.NUM_CLASSES)).astype(np.float32)
        labels[:, 0] = 1.0  # no negatives -> AUROC undefined for this class

        assert np.isnan(per_class_auroc(labels, probs)[0])
        assert not np.isnan(epoch_metrics(labels, probs)["auroc"])
