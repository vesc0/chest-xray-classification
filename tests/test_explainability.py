"""
Explainer plumbing.

The reshape transforms and target-layer lookups are the part of the XAI code
that fails quietly: a wrong layer or a transposed axis still yields a
plausible-looking heatmap, it just no longer means anything.
"""

import numpy as np
import pytest
import torch

import config
from explainability import (
    GRADCAM_TARGETS,
    GradCAM,
    _nhwc_to_nchw,
    _tokens_to_grid,
    build_explainers,
    denormalize,
    release_explainers,
)

ARCHITECTURES = ["densenet121", "efficientnet_b4", "vit_b_16", "swin_v2_b"]


class TestReshapeTransforms:
    def test_tokens_become_a_square_grid(self):
        # ViT-B/16 at 224: 196 patch tokens plus [CLS]
        tokens = torch.randn(2, 197, 768)
        assert _tokens_to_grid(tokens).shape == (2, 768, 14, 14)

    def test_the_cls_token_is_dropped(self):
        tokens = torch.zeros(1, 5, 3)
        tokens[:, 0, :] = 99.0  # [CLS] carries no location
        assert (_tokens_to_grid(tokens) != 99.0).all()

    def test_rejects_a_non_square_token_count(self):
        with pytest.raises(ValueError, match="square patch grid"):
            _tokens_to_grid(torch.randn(1, 8, 16))

    def test_nhwc_becomes_nchw(self):
        assert _nhwc_to_nchw(torch.randn(2, 7, 7, 512)).shape == (2, 512, 7, 7)

    def test_nhwc_conversion_preserves_values(self):
        tensor = torch.randn(1, 3, 4, 5)
        assert torch.equal(_nhwc_to_nchw(tensor)[0, :, 2, 3], tensor[0, 2, 3, :])


class TestDenormalize:
    def test_inverts_the_eval_transforms_normalization(self, rng):
        original = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        tensor = torch.from_numpy(original).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)

        recovered = denormalize((tensor - mean) / std)
        assert np.abs(recovered.astype(int) - original.astype(int)).max() <= 1

    def test_returns_a_displayable_uint8_image(self):
        result = denormalize(torch.randn(3, 16, 16))
        assert result.shape == (16, 16, 3)
        assert result.dtype == np.uint8

    def test_clamps_out_of_range_values(self):
        assert denormalize(torch.full((3, 8, 8), 50.0)).max() == 255
        assert denormalize(torch.full((3, 8, 8), -50.0)).min() == 0


class TestGradcamTargets:
    @pytest.mark.parametrize("name", ARCHITECTURES)
    def test_the_target_layer_resolves_for_every_architecture(self, name, model_cache):
        """
        Catches a torchvision attribute rename, which would otherwise surface as
        an AttributeError only after a full training run.
        """
        resolve, _ = GRADCAM_TARGETS[name]
        assert isinstance(resolve(model_cache(name)), torch.nn.Module)

    def test_every_supported_model_has_a_target(self):
        assert set(config.SUPPORTED_MODELS) <= set(GRADCAM_TARGETS)

    def test_the_vit_target_is_not_the_final_encoder_norm(self, model_cache):
        """
        The head reads only token 0, so gradients w.r.t. every patch token at
        encoder.ln are exactly zero and the CAM comes out blank. ln_1 of the
        last block still routes through attention to all tokens.
        """
        model = model_cache("vit_b_16")
        resolve, _ = GRADCAM_TARGETS["vit_b_16"]
        assert resolve(model) is model.backbone.encoder.layers[-1].ln_1
        assert resolve(model) is not model.backbone.encoder.ln


class TestGradCAM:
    @pytest.mark.parametrize("name", ["densenet121", "vit_b_16"])
    def test_produces_a_normalized_map_at_input_resolution(self, name, model_cache):
        images = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        with GradCAM(model_cache(name), model_name=name) as cam:
            heatmaps = cam.generate_batch(images, [0, 1])

        assert heatmaps.shape == (2, config.IMAGE_SIZE, config.IMAGE_SIZE)
        assert heatmaps.min() >= 0.0 and heatmaps.max() <= 1.0
        assert np.isfinite(heatmaps).all()

    def test_exposes_the_logits_from_its_own_forward_pass(self, model_cache):
        """Callers reuse these instead of running the model a second time."""
        images = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        with GradCAM(model_cache("densenet121"), model_name="densenet121") as cam:
            cam.generate_batch(images, [0, 1])
            assert cam.last_logits.shape == (2, config.NUM_CLASSES)

    def test_works_with_a_fully_frozen_backbone(self, model_cache):
        """
        With head_only tuning nothing upstream requires grad, so without rooting
        the graph at the target layer no gradient would be available at all.
        """
        model = model_cache("densenet121")
        for param in model.parameters():
            param.requires_grad = False
        try:
            images = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
            with GradCAM(model, model_name="densenet121") as cam:
                assert np.isfinite(cam.generate_batch(images, [0])).all()
        finally:
            for param in model.parameters():
                param.requires_grad = True

    def test_leaves_no_parameter_gradients_behind(self, model_cache):
        """autograd.grad walks back only to the target layer, so .grad stays clean."""
        model = model_cache("densenet121")
        model.zero_grad(set_to_none=True)
        images = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        with GradCAM(model, model_name="densenet121") as cam:
            cam.generate_batch(images, [0])
        assert all(p.grad is None for p in model.parameters())

    def test_rejects_an_unregistered_architecture(self, model_cache):
        with pytest.raises(ValueError, match="No Grad-CAM target layer"):
            GradCAM(model_cache("densenet121"), model_name="resnet50")

    def test_hooks_come_off_on_exit(self, model_cache):
        model = model_cache("densenet121")
        cam = GradCAM(model, model_name="densenet121")
        assert cam.handles
        cam.remove_hooks()
        assert not cam.handles
        cam.remove_hooks()  # idempotent

    def test_refuses_to_run_after_its_hooks_are_removed(self, model_cache):
        cam = GradCAM(model_cache("densenet121"), model_name="densenet121")
        cam.remove_hooks()
        images = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        with pytest.raises(RuntimeError, match="hooks have been removed"):
            cam.generate_batch(images, [0])


class TestBuildExplainers:
    @pytest.mark.parametrize("name", ARCHITECTURES)
    def test_every_architecture_gets_gradcam(self, name, model_cache):
        explainers = build_explainers(model_cache(name), name)
        try:
            assert "gradcam" in {key for key, _, _ in explainers}
        finally:
            release_explainers(explainers)

    def test_only_vit_additionally_gets_rollout(self, model_cache):
        vit = build_explainers(model_cache("vit_b_16"), "vit_b_16")
        cnn = build_explainers(model_cache("densenet121"), "densenet121")
        try:
            assert {key for key, _, _ in vit} == {"gradcam", "rollout"}
            assert {key for key, _, _ in cnn} == {"gradcam"}
        finally:
            release_explainers(vit)
            release_explainers(cnn)

    def test_the_class_agnostic_flag_distinguishes_the_two_methods(self, model_cache):
        """
        Rollout produces one map per image whatever the class. Losing this flag
        would let the localization report present the two as equivalent.
        """
        explainers = build_explainers(model_cache("vit_b_16"), "vit_b_16")
        try:
            flags = {key: explainer.class_agnostic for key, _, explainer in explainers}
            assert flags == {"gradcam": False, "rollout": True}
        finally:
            release_explainers(explainers)

    def test_release_detaches_every_hook(self, model_cache):
        explainers = build_explainers(model_cache("vit_b_16"), "vit_b_16")
        release_explainers(explainers)
        for _, _, explainer in explainers:
            if isinstance(explainer, GradCAM):
                assert not explainer.handles
