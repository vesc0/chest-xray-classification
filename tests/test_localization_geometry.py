"""
Localization geometry.

These are the functions behind the numbers compared against Wang et al., and
they are pure — no model, no dataset. The IoBB denominator in particular is a
definition the literature disagrees about, so it is pinned here rather than
left to whoever reads the code next.
"""

import numpy as np
import pytest

import config
from localization import _box_mask, _overlap_scores, _predicted_box, _scale_boxes


class TestScaleBoxes:
    def test_full_frame_box_maps_to_full_grid(self):
        boxes = np.array([[0.0, 0.0, 1024.0, 1024.0]])
        scaled = _scale_boxes(boxes, width=1024, height=1024)
        assert scaled[0] == pytest.approx([0.0, 0.0, config.IMAGE_SIZE, config.IMAGE_SIZE])

    def test_scales_from_the_files_real_size_not_a_hardcoded_1024(self):
        # A 512-wide source must scale by 224/512, not 224/1024
        boxes = np.array([[256.0, 0.0, 128.0, 512.0]])
        scaled = _scale_boxes(boxes, width=512, height=512)
        factor = config.IMAGE_SIZE / 512
        assert scaled[0] == pytest.approx([256 * factor, 0.0, 128 * factor, 512 * factor])

    def test_handles_non_square_sources_per_axis(self):
        boxes = np.array([[100.0, 200.0, 50.0, 80.0]])
        scaled = _scale_boxes(boxes, width=1000, height=500)
        assert scaled[0][0] == pytest.approx(100 * config.IMAGE_SIZE / 1000)
        assert scaled[0][1] == pytest.approx(200 * config.IMAGE_SIZE / 500)

    def test_does_not_mutate_its_input(self):
        boxes = np.array([[10.0, 20.0, 30.0, 40.0]])
        original = boxes.copy()
        _scale_boxes(boxes, width=1024, height=1024)
        assert np.array_equal(boxes, original)


class TestBoxMask:
    def test_covers_exactly_the_box_area(self):
        mask = _box_mask(np.array([[10.0, 20.0, 30.0, 40.0]]), size=224)
        assert mask.sum() == 30 * 40
        assert mask[20:60, 10:40].all()
        assert not mask[0:20, :].any()

    def test_unions_overlapping_boxes_without_double_counting(self):
        boxes = np.array([[0.0, 0.0, 20.0, 20.0], [10.0, 10.0, 20.0, 20.0]])
        mask = _box_mask(boxes, size=224)
        assert mask.sum() == 400 + 400 - 100

    def test_clips_boxes_running_past_the_edge(self):
        mask = _box_mask(np.array([[200.0, 200.0, 100.0, 100.0]]), size=224)
        assert mask.sum() == 24 * 24
        assert mask[200:224, 200:224].all()

    def test_ignores_degenerate_boxes(self):
        assert not _box_mask(np.array([[10.0, 10.0, 0.0, 0.0]]), size=224).any()


class TestPredictedBox:
    def test_returns_none_for_an_all_zero_heatmap(self):
        # Grad-CAM's ReLU can zero a map entirely. Thresholding it would pass
        # every pixel and report a whole-image detection, which scores at the
        # random baseline instead of admitting nothing was found.
        assert _predicted_box(np.zeros((224, 224))) is None

    def test_bounds_a_single_blob(self):
        heatmap = np.zeros((224, 224))
        heatmap[50:70, 30:60] = 1.0
        assert _predicted_box(heatmap) == (30, 50, 60, 70)

    def test_keeps_only_the_largest_connected_component(self):
        heatmap = np.zeros((224, 224))
        heatmap[10:15, 10:15] = 1.0   # small blob
        heatmap[100:140, 100:150] = 1.0  # larger blob
        assert _predicted_box(heatmap) == (100, 100, 150, 140)

    def test_threshold_is_relative_to_the_maps_own_maximum(self):
        # Peak 0.4 with everything else at 0.1: at half the max (0.2) only the
        # peak survives, even though no value reaches an absolute 0.5.
        heatmap = np.full((224, 224), 0.1)
        heatmap[100:110, 100:110] = 0.4
        assert _predicted_box(heatmap) == (100, 100, 110, 110)


class TestOverlapScores:
    def test_no_detection_scores_zero(self):
        truth = _box_mask(np.array([[10.0, 10.0, 50.0, 50.0]]), size=224)
        assert _overlap_scores(None, truth) == (0.0, 0.0)

    def test_exact_match_scores_one_on_both(self):
        truth = _box_mask(np.array([[10.0, 10.0, 50.0, 50.0]]), size=224)
        iou, iobb = _overlap_scores((10, 10, 60, 60), truth)
        assert iou == pytest.approx(1.0)
        assert iobb == pytest.approx(1.0)

    def test_iobb_denominator_is_the_predicted_box_not_the_union(self):
        """
        Wang et al.'s "intersection over the detected B-Box".

        A detection wholly inside a larger ground-truth box scores IoBB 1.0
        while IoU stays low. If this ever reports 0.25 for both, someone has
        switched the denominator to the union and the localization numbers are
        no longer comparable to the paper.
        """
        truth = _box_mask(np.array([[0.0, 0.0, 20.0, 20.0]]), size=224)
        iou, iobb = _overlap_scores((0, 0, 10, 10), truth)
        assert iou == pytest.approx(100 / 400)
        assert iobb == pytest.approx(1.0)

    def test_disjoint_boxes_score_zero(self):
        truth = _box_mask(np.array([[0.0, 0.0, 20.0, 20.0]]), size=224)
        iou, iobb = _overlap_scores((100, 100, 120, 120), truth)
        assert iou == 0.0
        assert iobb == 0.0

    def test_partial_overlap(self):
        truth = _box_mask(np.array([[0.0, 0.0, 20.0, 20.0]]), size=224)
        iou, iobb = _overlap_scores((10, 10, 30, 30), truth)
        intersection = 10 * 10
        assert iobb == pytest.approx(intersection / (20 * 20))
        assert iou == pytest.approx(intersection / (400 + 400 - intersection))
