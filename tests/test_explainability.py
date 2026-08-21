"""
Explainer plumbing: the reshape transforms and target-layer lookups, which fail
quietly — a wrong layer or a transposed axis still yields a plausible-looking
heatmap that no longer means anything.
"""

import numpy as np
import pytest
import torch

import config
from explainability import (
    GRADCAM_TARGETS,
    AttentionRollout,
    GradCAM,
    _nhwc_to_nchw,
    _normalize_per_sample,
    _tokens_to_grid,
    available_explainers,
    build_explainer,
    build_explainers,
    denormalize,
    release_explainers,
)

ARCHITECTURES = list(config.SUPPORTED_MODELS)


class TestReshapeTransforms:
    def test_tokens_become_a_square_grid(self):
        # ViT-S/16 at 224: 196 patch tokens plus [CLS], width 384.
        tokens = torch.randn(2, 197, 384)
        assert _tokens_to_grid(tokens).shape == (2, 384, 14, 14)

    def test_the_cls_token_is_dropped(self):
        tokens = torch.zeros(1, 5, 3)
        tokens[:, 0, :] = 99.0  # [CLS] carries no location
        assert (_tokens_to_grid(tokens) != 99.0).all()

    def test_more_prefix_tokens_can_be_dropped(self):
        """Dropping the wrong count still reshapes cleanly whenever the
        remainder is square, so it has to be passed in rather than assumed."""
        tokens = torch.zeros(1, 6, 3)
        tokens[:, :2, :] = 99.0
        assert (_tokens_to_grid(tokens, num_prefix_tokens=2) != 99.0).all()

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
        """A torchvision attribute rename would otherwise surface only after a
        full training run."""
        resolve, _ = GRADCAM_TARGETS[name]
        assert isinstance(resolve(model_cache(name)), torch.nn.Module)

    def test_the_target_table_matches_the_roster_exactly(self):
        """Equality, not containment: a stale entry is how a target table
        starts describing models that no longer exist."""
        assert set(GRADCAM_TARGETS) == set(config.SUPPORTED_MODELS)

    def test_the_vit_target_is_not_the_final_norm(self, model_cache):
        """The head reads only [CLS], so gradients w.r.t. every patch token at
        the final norm are zero and the CAM comes out blank."""
        model = model_cache("vit_s_16")
        resolve, _ = GRADCAM_TARGETS["vit_s_16"]
        assert resolve(model) is model.backbone.blocks[-1].norm1
        assert resolve(model) is not model.backbone.norm

    def test_the_vit_reshape_is_bound_to_the_backbone(self, model_cache):
        """The prefix-token count is read off the backbone, so the registry
        entry has to consult the model it is given."""
        model = model_cache("vit_s_16")
        _, build_transform = GRADCAM_TARGETS["vit_s_16"]

        tokens = torch.zeros(1, 197, 384)
        tokens[:, 0, :] = 99.0
        grid = build_transform(model)(tokens)

        assert grid.shape == (1, 384, 14, 14)
        assert (grid != 99.0).all()

    def test_the_vit_cam_grid_is_finer_than_the_hierarchical_models(self, model_cache):
        """
        A patch-16 transformer keeps one token per patch where the others have
        downsampled to 7x7. The asymmetry favours ViT on IoU/IoBB, so it is
        pinned as a measured fact rather than a claim in a docstring.
        """
        images = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        grids = {}
        for name in ("vit_s_16", "densenet121", "convnextv2_t", "swin_v2_t", "maxvit_t"):
            model = model_cache(name)
            resolve, build_transform = GRADCAM_TARGETS[name]
            captured = {}
            handle = resolve(model).register_forward_hook(
                lambda mod, inp, out: captured.update(out=out)
            )
            try:
                with torch.no_grad():
                    model(images)
            finally:
                handle.remove()
            grids[name] = build_transform(model)(captured["out"]).shape[-1]

        assert grids["vit_s_16"] == 14
        assert all(size == 7 for name, size in grids.items() if name != "vit_s_16")


class TestGradCAM:
    @pytest.mark.parametrize("name", ["densenet121", "vit_s_16"])
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
        """With head_only tuning nothing upstream requires grad, so without
        rooting the graph at the target layer there is no gradient at all."""
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


class TestAttentionRollout:
    def test_captures_attention_from_every_block(self, model_cache):
        """timm's default fused SDPA never materializes an attention matrix, so
        without the unfused path the rollout comes out empty, not erroring."""
        model = model_cache("vit_s_16")
        rollout = AttentionRollout(model)
        rollout.generate_batch(torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE))
        assert len(rollout.attention_maps) == len(model.backbone.blocks)

    def test_leaves_the_model_as_it_found_it(self, model_cache):
        """Capture mutates fused_attn on every block; left flipped, the model
        runs the slower path for the rest of the process."""
        model = model_cache("vit_s_16")
        before = [block.attn.fused_attn for block in model.backbone.blocks]

        rollout = AttentionRollout(model)
        rollout.generate_batch(torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE))

        assert [block.attn.fused_attn for block in model.backbone.blocks] == before
        assert not rollout._handles

    def test_produces_a_normalized_map_at_input_resolution(self, model_cache):
        rollout = AttentionRollout(model_cache("vit_s_16"))
        maps = rollout.generate_batch(
            torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        )
        assert maps.shape == (2, config.IMAGE_SIZE, config.IMAGE_SIZE)
        assert maps.min() >= 0.0 and maps.max() <= 1.0
        assert np.isfinite(maps).all()

    def test_raises_rather_than_returning_a_blank_map(self, model_cache):
        """A constant heatmap would pass through localization scoring as a real
        result, so failing loudly is the only way it surfaces."""
        rollout = AttentionRollout(model_cache("vit_s_16"))
        rollout._start_capture = lambda: None  # capture never installed

        with pytest.raises(RuntimeError, match="captured no attention"):
            rollout.generate_batch(torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE))


class TestNormalization:
    def test_each_sample_is_normalized_independently(self):
        maps = torch.stack([torch.tensor([[0.0, 1.0]]), torch.tensor([[5.0, 9.0]])])
        result = _normalize_per_sample(maps)
        assert result[0].flatten().tolist() == pytest.approx([0.0, 1.0])
        assert result[1].flatten().tolist() == pytest.approx([0.0, 1.0])

    def test_a_constant_map_collapses_to_zero(self):
        """The signature localization._is_degenerate looks for: a constant map
        has no peak, and this is the form it arrives in."""
        assert _normalize_per_sample(torch.full((1, 4, 4), 3.0)).max() == 0.0


class TestRawMaps:
    """
    The energy fraction is measured before normalization, since min-max
    subtracts exactly the uniform component the random baseline is defined as.
    Both explainers have to expose the pre-normalization map.
    """

    @pytest.mark.parametrize("name", ["densenet121", "vit_s_16"])
    def test_gradcam_exposes_maps_before_normalization(self, name, model_cache):
        images = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        with GradCAM(model_cache(name), model_name=name) as cam:
            cam.generate_batch(images, [0, 1])
            raw = cam.last_raw_maps

        assert raw.shape == (2, config.IMAGE_SIZE, config.IMAGE_SIZE)
        assert (raw >= 0).all()  # Grad-CAM's ReLU floors these

    def test_rollout_exposes_maps_before_normalization(self, model_cache):
        rollout = AttentionRollout(model_cache("vit_s_16"))
        rollout.generate_batch(torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE))

        assert rollout.last_raw_maps.shape == (2, config.IMAGE_SIZE, config.IMAGE_SIZE)
        assert (rollout.last_raw_maps >= 0).all()

    def test_rollouts_raw_map_has_the_floor_that_normalization_removes(self, model_cache):
        """
        Rolled-up attention is strictly positive with a substantial floor, so
        min-subtraction is not a rescale for it: measured on the normalized
        map, the metric no longer shares a baseline with the number beside it.
        """
        rollout = AttentionRollout(model_cache("vit_s_16"))
        normalized = rollout.generate_batch(
            torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        )
        raw = rollout.last_raw_maps

        assert raw.min() > 0.0
        assert normalized.min() == pytest.approx(0.0, abs=1e-6)
        # A real share of the map, not a rounding artifact.
        assert raw.min() / raw.max() > 0.01


class TestHookGating:
    def test_the_hook_does_not_capture_outside_generate_batch(self, model_cache):
        """
        Both explainers are live during ViT's XAI stage, so an always-on hook
        fires on the other's forward passes and calls requires_grad_() inside
        no_grad — an exception the moment a caller uses inference_mode().
        """
        model = model_cache("densenet121")
        with GradCAM(model, model_name="densenet121") as cam:
            with torch.no_grad():
                model(torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE))
            assert cam._captured is None

    def test_a_live_explainer_survives_an_inference_mode_forward(self, model_cache):
        model = model_cache("densenet121")
        with GradCAM(model, model_name="densenet121"):
            with torch.inference_mode():
                model(torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE))

    def test_capturing_is_off_again_after_a_failed_generate(self, model_cache):
        """The flag is cleared in a finally, so one bad call cannot arm it forever."""
        cam = GradCAM(model_cache("densenet121"), model_name="densenet121")
        try:
            with pytest.raises(Exception):
                cam.generate_batch(torch.randn(1, 3, 8, 8), [0])  # wrong input size
            assert not cam._capturing
        finally:
            cam.remove_hooks()


class TestExplainerConstruction:
    @pytest.mark.parametrize("name", ARCHITECTURES)
    def test_available_explainers_needs_no_model(self, name):
        """Localization plans its run over the methods before instantiating
        any, which is what lets it keep one attached at a time."""
        assert [key for key, _ in available_explainers(name)][0] == "gradcam"

    def test_available_explainers_agrees_with_build_explainers(self, model_cache):
        for name in ARCHITECTURES:
            explainers = build_explainers(model_cache(name), name)
            try:
                assert [(k, l) for k, l, _ in explainers] == available_explainers(name)
            finally:
                release_explainers(explainers)

    def test_build_explainer_rejects_a_method_the_model_does_not_have(self, model_cache):
        with pytest.raises(ValueError, match="Unknown explainer"):
            build_explainer(model_cache("densenet121"), "densenet121", "rollout")


class TestBuildExplainers:
    @pytest.mark.parametrize("name", ARCHITECTURES)
    def test_every_architecture_gets_gradcam(self, name, model_cache):
        explainers = build_explainers(model_cache(name), name)
        try:
            assert "gradcam" in {key for key, _, _ in explainers}
        finally:
            release_explainers(explainers)

    def test_only_vit_additionally_gets_rollout(self, model_cache):
        vit = build_explainers(model_cache("vit_s_16"), "vit_s_16")
        cnn = build_explainers(model_cache("densenet121"), "densenet121")
        try:
            assert {key for key, _, _ in vit} == {"gradcam", "rollout"}
            assert {key for key, _, _ in cnn} == {"gradcam"}
        finally:
            release_explainers(vit)
            release_explainers(cnn)

    def test_the_class_agnostic_flag_distinguishes_the_two_methods(self, model_cache):
        """Rollout produces one map per image whatever the class; losing this
        flag would let the report present the two methods as equivalent."""
        explainers = build_explainers(model_cache("vit_s_16"), "vit_s_16")
        try:
            flags = {key: explainer.class_agnostic for key, _, explainer in explainers}
            assert flags == {"gradcam": False, "rollout": True}
        finally:
            release_explainers(explainers)

    def test_release_detaches_every_hook(self, model_cache):
        explainers = build_explainers(model_cache("vit_s_16"), "vit_s_16")
        release_explainers(explainers)
        for _, _, explainer in explainers:
            if isinstance(explainer, GradCAM):
                assert not explainer.handles
