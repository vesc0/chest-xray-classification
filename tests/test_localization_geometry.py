"""
Localization geometry: the pure functions behind the numbers compared against
Wang et al. The IoBB denominator is a definition the literature disagrees
about, so it is pinned here rather than left to the next reader.
"""

import numpy as np
import pytest
import torch
from torchvision import transforms

import config
from dataset import RESIZE_INTERPOLATION, get_eval_transforms
from localization import (
    TRANSFORM_PROBE_MIN_IOU,
    _box_mask,
    _is_degenerate,
    _overlap_scores,
    _predicted_box,
    _scale_boxes,
    check_eval_transform_geometry,
)


class TestScaleBoxes:
    def test_full_frame_box_maps_to_full_grid(self):
        boxes = np.array([[0.0, 0.0, 1024.0, 1024.0]])
        scaled = _scale_boxes(boxes, width=1024, height=1024)
        assert scaled[0] == pytest.approx([0.0, 0.0, config.IMAGE_SIZE, config.IMAGE_SIZE])

    def test_scales_from_the_files_real_size_not_a_hardcoded_1024(self):
        # A 512-wide source must scale by 224/512, not 224/1024.
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


class TestEvalTransformGeometry:
    """
    Ties _scale_boxes to the transform the images actually go through. Every
    localization number depends on the two agreeing, and nothing else notices.
    """

    def test_the_real_eval_pipeline_passes(self):
        check_eval_transform_geometry()

    def test_a_centre_crop_is_rejected(self):
        """The most likely thing to reach for, and it offsets every box while
        raising nothing."""
        cropping = transforms.Compose([
            transforms.Resize(256, interpolation=RESIZE_INTERPOLATION),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ])
        with pytest.raises(RuntimeError, match="no longer matches the box scaling"):
            check_eval_transform_geometry(cropping)

    def test_an_aspect_preserving_resize_is_rejected(self):
        """Resize(224) is square-in, square-out, which is why the probe is not
        square: on non-square input it letterboxes."""
        aspect = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE, interpolation=RESIZE_INTERPOLATION),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ])
        with pytest.raises(RuntimeError, match="no longer matches the box scaling"):
            check_eval_transform_geometry(aspect)

    def test_the_threshold_leaves_room_for_resampling_softness(self):
        """
        Bicubic softens the probe's edges, so a correct pipeline cannot score
        1.0. The gap to what a wrong pipeline scores is the margin this
        threshold sits in.
        """
        assert 0.5 < TRANSFORM_PROBE_MIN_IOU < 0.97


class TestIsDegenerate:
    def test_an_all_zero_map_is_degenerate(self):
        assert _is_degenerate(np.zeros((224, 224)))

    def test_a_map_with_a_peak_is_not(self):
        heatmap = np.zeros((224, 224))
        heatmap[100, 100] = 1.0
        assert not _is_degenerate(heatmap)

    def test_a_constant_map_normalizes_to_zero_and_is_caught(self):
        """min-max on a constant map yields zeros, which is the form a
        signal-free map actually reaches localization in."""
        from explainability import _normalize_per_sample

        constant = _normalize_per_sample(torch.full((1, 8, 8), 0.7)).numpy()[0]
        assert _is_degenerate(constant)


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
        # Thresholding a zeroed map passes every pixel and reports a
        # whole-image detection, scoring at the random baseline.
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
        # At half the max (0.2) only the peak survives, though nothing reaches
        # an absolute 0.5.
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
        Wang et al.'s "intersection over the detected B-Box": a detection
        wholly inside a larger box scores IoBB 1.0 while IoU stays low. Both
        reporting 0.25 means someone switched the denominator to the union.
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
