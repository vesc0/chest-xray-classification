"""
Splitting and subset sampling.

These lock down the three properties the README claims and the experimental
protocol depends on: train/val are patient-disjoint, subsets are nested, and
every split tracks the pool's label distribution. All three are silent when
broken — the run still trains, it just no longer measures what it claims to.
"""

import numpy as np
import pytest

import config
from conftest import make_label_frame
from dataset import (
    _grouped_train_val_split,
    _iterative_stratified_group_folds,
    _stack_label_vectors,
    _stratified_subset,
)


def prevalence(frame):
    return _stack_label_vectors(frame).mean(axis=0)


class TestStackLabelVectors:
    def test_empty_frame_keeps_the_class_axis(self):
        # Downstream code indexes [:, class_idx]; a bare (0,) would break it
        empty = make_label_frame(n_patients=1).iloc[0:0]
        assert _stack_label_vectors(empty).shape == (0, config.NUM_CLASSES)

    def test_shape_and_dtype(self, label_frame):
        matrix = _stack_label_vectors(label_frame)
        assert matrix.shape == (len(label_frame), config.NUM_CLASSES)
        assert matrix.dtype == np.float32


class TestGroupedTrainValSplit:
    def test_no_patient_appears_in_both_splits(self, label_frame):
        """The leakage check. A patient in both splits inflates validation."""
        train, val = _grouped_train_val_split(
            label_frame, test_size=0.2, group_col=config.PATIENT_ID_COLUMN
        )
        overlap = set(train[config.PATIENT_ID_COLUMN]) & set(val[config.PATIENT_ID_COLUMN])
        assert overlap == set()

    def test_every_row_lands_somewhere_exactly_once(self, label_frame):
        train, val = _grouped_train_val_split(
            label_frame, test_size=0.2, group_col=config.PATIENT_ID_COLUMN
        )
        assert len(train) + len(val) == len(label_frame)
        assert set(train["Image Index"]).isdisjoint(set(val["Image Index"]))
        assert set(train["Image Index"]) | set(val["Image Index"]) == set(
            label_frame["Image Index"]
        )

    def test_validation_is_roughly_the_requested_fraction(self, label_frame):
        train, val = _grouped_train_val_split(
            label_frame, test_size=0.2, group_col=config.PATIENT_ID_COLUMN
        )
        # Whole patients move together, so the fraction cannot be exact
        assert 0.15 <= len(val) / len(label_frame) <= 0.25

    def test_label_marginals_are_preserved(self):
        """
        Sized deliberately larger than the default fixture.

        Balancing needs enough patients to have any freedom left once the rare
        classes have been placed: measured across seeds, 300 patients leaves
        residual deviations up to ~0.13 on the common classes, while 800 holds
        within 0.04. A tight bound on a small frame would be testing sampling
        noise rather than the stratification.
        """
        frame = make_label_frame(n_patients=800, seed=3)
        train, val = _grouped_train_val_split(
            frame, test_size=0.2, group_col=config.PATIENT_ID_COLUMN
        )
        assert prevalence(val) == pytest.approx(prevalence(train), abs=0.05)

    def test_is_deterministic_for_a_fixed_seed(self, label_frame):
        first, _ = _grouped_train_val_split(
            label_frame, test_size=0.2, group_col=config.PATIENT_ID_COLUMN
        )
        second, _ = _grouped_train_val_split(
            label_frame, test_size=0.2, group_col=config.PATIENT_ID_COLUMN
        )
        assert list(first["Image Index"]) == list(second["Image Index"])

    def test_falls_back_when_grouping_is_unusable(self, label_frame):
        frame = label_frame.drop(columns=[config.PATIENT_ID_COLUMN])
        train, val = _grouped_train_val_split(frame, test_size=0.2, group_col="missing")
        assert len(train) + len(val) == len(frame)


class TestStratifiedSubset:
    def test_subsets_are_nested(self, label_frame):
        """
        The property that makes a scaling curve mean anything.

        If a 100-image subset is not contained in a 200-image one, the curve
        measures which patients each draw happened to catch rather than the
        effect of training-data volume.
        """
        small = _stratified_subset(
            label_frame, 100, group_col=config.PATIENT_ID_COLUMN, seed=config.SEED
        )
        large = _stratified_subset(
            label_frame, 200, group_col=config.PATIENT_ID_COLUMN, seed=config.SEED
        )
        assert set(small["Image Index"]) <= set(large["Image Index"])

    def test_nesting_holds_across_three_sizes(self, label_frame):
        sizes = [80, 160, 320]
        subsets = [
            set(
                _stratified_subset(
                    label_frame, n, group_col=config.PATIENT_ID_COLUMN, seed=config.SEED
                )["Image Index"]
            )
            for n in sizes
        ]
        assert subsets[0] <= subsets[1] <= subsets[2]

    def test_returns_the_requested_size(self, label_frame):
        subset = _stratified_subset(
            label_frame, 150, group_col=config.PATIENT_ID_COLUMN, seed=config.SEED
        )
        assert len(subset) == 150

    def test_a_prefix_tracks_the_pools_label_distribution(self, label_frame):
        subset = _stratified_subset(
            label_frame, 300, group_col=config.PATIENT_ID_COLUMN, seed=config.SEED
        )
        assert prevalence(subset) == pytest.approx(prevalence(label_frame), abs=0.06)

    def test_a_prefix_beats_a_random_draw_of_the_same_size(self):
        """
        The test that distinguishes stratified sampling from plain sampling.

        An absolute-prevalence check is not enough on its own: a random draw of
        a few hundred rows also lands within a few points of the pool on the
        common classes, so it passes the assertion above. Scoring deviation in
        *relative* terms is what exposes the difference — it weights a 0.5%
        class as heavily as a 40% one, which is precisely what the shard
        ordering optimizes and what a random draw gets wrong.

        Measured across fixture seeds, the stratified prefix beats the best of
        20 random draws every time, so this compares against that best rather
        than the median.
        """
        frame = make_label_frame(n_patients=800, seed=5)
        pool = prevalence(frame)
        safe = np.maximum(pool, 1e-12)

        def relative_deviation(subset):
            return float((np.abs(prevalence(subset) - pool) / safe).sum())

        stratified = relative_deviation(
            _stratified_subset(frame, 200, group_col=config.PATIENT_ID_COLUMN, seed=config.SEED)
        )
        best_random = min(
            relative_deviation(frame.sample(n=200, random_state=seed)) for seed in range(20)
        )
        assert stratified < best_random

    def test_returns_everything_when_the_target_exceeds_the_pool(self, label_frame):
        subset = _stratified_subset(
            label_frame, len(label_frame) + 500,
            group_col=config.PATIENT_ID_COLUMN, seed=config.SEED,
        )
        assert len(subset) == len(label_frame)

    def test_zero_or_negative_target_means_no_subsetting(self, label_frame):
        subset = _stratified_subset(
            label_frame, 0, group_col=config.PATIENT_ID_COLUMN, seed=config.SEED
        )
        assert len(subset) == len(label_frame)

    def test_falls_back_to_random_sampling_without_a_group_column(self, label_frame):
        subset = _stratified_subset(label_frame, 100, group_col=None, seed=config.SEED)
        assert len(subset) == 100


class TestIterativeStratifiedGroupFolds:
    def test_folds_partition_every_row(self, label_frame):
        folds = _iterative_stratified_group_folds(
            label_frame, [0.5, 0.3, 0.2], config.PATIENT_ID_COLUMN, seed=0
        )
        combined = np.concatenate(folds)
        assert sorted(combined.tolist()) == list(range(len(label_frame)))

    def test_patients_are_never_split_across_folds(self, label_frame):
        folds = _iterative_stratified_group_folds(
            label_frame, [0.5, 0.5], config.PATIENT_ID_COLUMN, seed=0
        )
        seen: dict[str, int] = {}
        for fold_idx, rows in enumerate(folds):
            for patient in label_frame.iloc[rows][config.PATIENT_ID_COLUMN]:
                assert seen.setdefault(patient, fold_idx) == fold_idx

    def test_fold_sizes_approximate_the_requested_proportions(self, label_frame):
        folds = _iterative_stratified_group_folds(
            label_frame, [0.7, 0.3], config.PATIENT_ID_COLUMN, seed=0
        )
        assert len(folds[0]) / len(label_frame) == pytest.approx(0.7, abs=0.06)

    def test_rare_classes_reach_every_fold(self, label_frame):
        """
        The reason for iterative stratification over a plain grouped split.

        The rarest class here sits near 0.5%; a naive split can drop it from a
        fold entirely, which makes its AUROC undefined and quietly removes it
        from the macro average.
        """
        folds = _iterative_stratified_group_folds(
            label_frame, [0.5, 0.5], config.PATIENT_ID_COLUMN, seed=0
        )
        pool_counts = _stack_label_vectors(label_frame).sum(axis=0)
        for rows in folds:
            counts = _stack_label_vectors(label_frame.iloc[rows]).sum(axis=0)
            # Any class with a real presence in the pool must survive the split
            assert (counts[pool_counts >= 4] > 0).all()
