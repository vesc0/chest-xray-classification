"""
Subgroup stratification over saved predictions.

The load-bearing part is alignment: the join is positional, so the tests that
matter prove a misaligned one is refused rather than silently producing
plausible disparities. The metadata frames are synthetic.
"""

import numpy as np
import pandas as pd
import pytest

import config
from bias_analysis import (
    AGE_BANDS,
    METRIC_KEYS,
    align_metadata,
    bootstrap_gaps,
    shared_classes,
    subgroup_masks,
    subgroup_metrics,
)

ALL = np.arange(2)


def make_meta(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """A test-split metadata frame: two studies per patient, mixed demographics."""
    generator = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Image Index": [f"{i:05d}.png" for i in range(n)],
            config.PATIENT_ID_COLUMN: [str(i // 2) for i in range(n)],
            "Patient Age": generator.integers(5, 95, n),
            "Patient Gender": generator.choice(["M", "F"], n),
        }
    )


class TestAlignMetadata:
    def test_accepts_matching_patient_sequence(self):
        meta = make_meta()
        groups = meta[config.PATIENT_ID_COLUMN].to_numpy()
        assert align_metadata(meta, groups) is meta

    def test_rejects_reordered_patients(self):
        meta = make_meta()
        groups = meta[config.PATIENT_ID_COLUMN].to_numpy()[::-1]
        with pytest.raises(ValueError, match="do not line up"):
            align_metadata(meta, groups)

    def test_rejects_missing_rows(self):
        meta = make_meta()
        groups = meta[config.PATIENT_ID_COLUMN].to_numpy()[:-1]
        with pytest.raises(ValueError, match="do not line up"):
            align_metadata(meta, groups)

    def test_rejects_predictions_without_patient_ids(self):
        with pytest.raises(ValueError, match="no patient IDs"):
            align_metadata(make_meta(), None)


class TestSubgroupMasks:
    def test_sex_levels_partition_the_split(self):
        meta = make_meta()
        axes = subgroup_masks(meta)
        assert (axes["sex"]["M"] | axes["sex"]["F"]).all()
        assert not (axes["sex"]["M"] & axes["sex"]["F"]).any()

    def test_age_bands_do_not_overlap(self):
        axes = subgroup_masks(make_meta())
        stacked = np.stack(list(axes["age"].values()))
        assert stacked.sum(axis=0).max() <= 1

    def test_impossible_ages_fall_outside_every_band(self):
        meta = make_meta(n=4)
        meta.loc[0, "Patient Age"] = 414
        axes = subgroup_masks(meta)
        assert not np.stack(list(axes["age"].values()))[:, 0].any()

    def test_band_upper_bound_is_inclusive(self):
        meta = make_meta(n=2)
        meta.loc[0, "Patient Age"] = AGE_BANDS[0][1]
        axes = subgroup_masks(meta)
        assert axes["age"][f"{AGE_BANDS[0][0]}-{AGE_BANDS[0][1]}"][0]


class TestSharedClasses:
    def test_keeps_only_classes_attested_in_both_groups(self):
        # Class 0 is common in both; class 1 lives entirely in the reference.
        labels = np.array([[1, 0]] * 20 + [[1, 1]] * 20)
        level = np.array([True] * 20 + [False] * 20)
        assert shared_classes(labels, level, ~level, minimum=10).tolist() == [0]

    def test_rare_class_is_dropped_from_both_sides(self):
        """The Hernia case: one positive in a band is not an AUROC, and left in
        it swings that band's macro by points that read as a disparity."""
        labels = np.zeros((40, 2), dtype=np.int8)
        labels[:20, 0] = 1
        labels[20:, 0] = 1
        labels[0, 1] = 1
        level = np.array([True] * 20 + [False] * 20)
        assert shared_classes(labels, level, ~level, minimum=10).tolist() == [0]

    def test_returns_empty_when_nothing_is_attested(self):
        labels = np.zeros((10, 2), dtype=np.int8)
        level = np.array([True] * 5 + [False] * 5)
        assert shared_classes(labels, level, ~level, minimum=10).size == 0


class TestSubgroupMetrics:
    def test_perfect_model_has_no_misses(self):
        labels = np.array([[1, 0], [0, 1], [1, 1]])
        probs = labels.astype(np.float32)
        result = subgroup_metrics(labels, probs, probs > 0.5, ALL)
        assert result["fnr"] == pytest.approx(0.0)
        assert result["underdiagnosis"] == pytest.approx(0.0)

    def test_silent_model_misses_every_abnormal_study(self):
        labels = np.array([[1, 0], [0, 1], [0, 0]])
        probs = np.zeros_like(labels, dtype=np.float32)
        result = subgroup_metrics(labels, probs, probs > 0.5, ALL)
        assert result["fnr"] == pytest.approx(1.0)
        # Two of the three studies carry a finding, and neither fires.
        assert result["underdiagnosis"] == pytest.approx(1.0)

    def test_underdiagnosis_ignores_healthy_studies(self):
        labels = np.array([[0, 0], [0, 0], [1, 0]])
        preds = np.array([[True, True], [False, False], [True, False]])
        assert subgroup_metrics(
            labels, np.zeros_like(labels, dtype=np.float32), preds, ALL
        )["underdiagnosis"] == pytest.approx(0.0)

    def test_class_absent_from_subgroup_is_skipped_not_zeroed(self):
        # Class 1 has no positives; a NaN-blind macro would halve the FNR.
        labels = np.array([[1, 0], [1, 0]])
        preds = np.zeros_like(labels, dtype=bool)
        assert subgroup_metrics(labels, labels.astype(np.float32), preds, ALL)[
            "fnr"
        ] == pytest.approx(1.0)

    def test_empty_subgroup_is_all_nan(self):
        empty = np.zeros((0, 2), dtype=np.int8)
        result = subgroup_metrics(
            empty, empty.astype(np.float32), empty.astype(bool), ALL
        )
        assert all(np.isnan(result[key]) for key in METRIC_KEYS)

    def test_macro_covers_only_the_given_classes(self):
        """Restricting the class set must change the macro, not just the label."""
        labels = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        probs = np.array([[0.9, 0.9], [0.1, 0.1], [0.8, 0.8], [0.2, 0.2]], dtype=np.float32)
        both = subgroup_metrics(labels, probs, probs > 0.5, ALL)["auroc"]
        first = subgroup_metrics(labels, probs, probs > 0.5, np.array([0]))["auroc"]
        assert first == pytest.approx(1.0)
        assert both < first

    def test_empty_class_set_is_nan_but_keeps_underdiagnosis(self):
        labels = np.array([[1, 0], [0, 0]])
        preds = np.zeros_like(labels, dtype=bool)
        result = subgroup_metrics(
            labels, labels.astype(np.float32), preds, np.array([], dtype=int)
        )
        assert np.isnan(result["auroc"]) and np.isnan(result["fnr"])
        assert result["underdiagnosis"] == pytest.approx(1.0)


class TestBootstrapGaps:
    def test_reference_gap_is_identically_zero(self):
        """The reference minus itself is zero in every draw, interval included."""
        meta = make_meta(n=40)
        generator = np.random.default_rng(3)
        labels = generator.integers(0, 2, (40, config.NUM_CLASSES)).astype(np.int8)
        probs = generator.random((40, config.NUM_CLASSES)).astype(np.float32)
        axes = subgroup_masks(meta)
        classes = np.arange(config.NUM_CLASSES)
        class_sets = {
            axis: {level: classes for level in levels} for axis, levels in axes.items()
        }

        gaps = bootstrap_gaps(
            labels,
            probs,
            probs > 0.5,
            axes,
            class_sets,
            meta[config.PATIENT_ID_COLUMN].to_numpy(),
            num_samples=20,
        )
        assert gaps["sex"]["M"]["auroc"] == [0.0, 0.0]

    def test_resamples_whole_patients(self, monkeypatch):
        """
        Rows arrive as complete patients, never as loose studies — row-level
        draws would shrink every interval. Patient 0 owns three rows here, so
        its subgroup can only ever be empty or a multiple of three.
        """
        import bias_analysis

        patients = ["0", "0", "0"] + [str(i // 2) for i in range(2, 40)]
        rows = len(patients)
        mask = np.array([patient == "0" for patient in patients])

        seen = []
        original = bias_analysis.per_class_rates

        def spy(labels, probs, preds):
            seen.append(labels.shape[0])
            return original(labels, probs, preds)

        monkeypatch.setattr(bias_analysis, "per_class_rates", spy)
        generator = np.random.default_rng(4)
        labels = generator.integers(0, 2, (rows, config.NUM_CLASSES)).astype(np.int8)
        probs = generator.random((rows, config.NUM_CLASSES)).astype(np.float32)

        classes = np.arange(config.NUM_CLASSES)
        bootstrap_gaps(
            labels,
            probs,
            probs > 0.5,
            {"sex": {"M": mask, "F": ~mask}},
            {"sex": {"M": classes, "F": classes}},
            np.array(patients),
            num_samples=40,
        )
        drawn = seen[::2]  # the masked level is scored first in each draw
        assert all(size % 3 == 0 for size in drawn)
        assert len(set(drawn)) > 1  # the draw actually varies
