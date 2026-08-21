"""
The image pipeline, whose failures are all silent: a resampling filter, a
train/eval geometry mismatch and an anatomically wrong augmentation each train
a model to completion and simply move the numbers.

Synthetic PIL images throughout; nothing reads the dataset.
"""

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

import config
from dataset import RESIZE_INTERPOLATION, get_eval_transforms, get_train_transforms


@pytest.fixture
def radiograph():
    """A 1024x1024 8-bit image, the shape and depth of a ChestX-ray14 PNG."""
    generator = np.random.default_rng(0)
    return Image.fromarray(
        generator.integers(0, 256, (1024, 1024), dtype=np.uint8), mode="L"
    ).convert("RGB")


def _resizes(pipeline):
    return [t for t in pipeline.transforms if isinstance(t, transforms.Resize)]


class TestInterpolation:
    def test_both_pipelines_resize_bicubic(self):
        """Four of five backbones were pretrained bicubic, and Resize defaults
        to bilinear, so leaving it implicit changes the filter."""
        assert RESIZE_INTERPOLATION is transforms.InterpolationMode.BICUBIC
        for pipeline in (get_train_transforms(), get_eval_transforms()):
            resizes = _resizes(pipeline)
            assert resizes, "no Resize in the pipeline"
            for resize in resizes:
                assert resize.interpolation is transforms.InterpolationMode.BICUBIC


class TestGeometry:
    def test_train_and_eval_agree_on_output_geometry(self, radiograph):
        """Thresholds are calibrated on one pipeline and applied to the other,
        so any geometry difference decalibrates every reported number."""
        train = get_train_transforms()(radiograph)
        evaluation = get_eval_transforms()(radiograph)
        assert train.shape == evaluation.shape
        assert train.shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE)

    def test_the_full_frame_survives_resizing(self):
        """A crop would remove the costophrenic angles and lung apices, where
        effusions and pneumothoraces present."""
        for pipeline in (get_train_transforms(), get_eval_transforms()):
            assert not any(
                isinstance(t, (transforms.CenterCrop, transforms.RandomResizedCrop))
                for t in pipeline.transforms
            )


class TestAugmentation:
    def test_no_horizontal_flip(self):
        """
        Radiographs have fixed laterality, so mirroring teaches the model that
        dextrocardia is unremarkable — and a flip is the single most likely
        augmentation for someone to add here by reflex.
        """
        assert not any(
            isinstance(t, transforms.RandomHorizontalFlip)
            for t in get_train_transforms().transforms
        )

    def test_evaluation_is_deterministic(self, radiograph):
        """Val and test must not move between two passes over the same image."""
        pipeline = get_eval_transforms()
        assert torch.equal(pipeline(radiograph), pipeline(radiograph))

    def test_training_augmentation_actually_perturbs(self, radiograph):
        pipeline = get_train_transforms()
        torch.manual_seed(0)
        first = pipeline(radiograph)
        torch.manual_seed(1)
        assert not torch.equal(first, pipeline(radiograph))


class TestNormalization:
    def test_grayscale_is_replicated_across_three_channels(self, radiograph):
        """
        convert("RGB") replicates the single channel and the per-channel
        normalization then shifts the copies apart, so identical channels here
        would mean the conversion was skipped.
        """
        tensor = get_eval_transforms()(radiograph)
        assert tensor.shape[0] == 3

        mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)
        raw = tensor * std + mean
        assert torch.allclose(raw[0], raw[1], atol=1e-5)
        assert torch.allclose(raw[1], raw[2], atol=1e-5)
