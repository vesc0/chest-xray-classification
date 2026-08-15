"""
The model factory and the architecture assumptions built on top of it.

Two things are pinned here. The roster the factory can build has to stay in
step with config.SUPPORTED_MODELS, because a name present in one but not the
other fails only when a sweep reaches that model — hours into a run. And the
structural facts the XAI code relies on (prefix-token count, fixed input
resolution) are properties of third-party checkpoints, so they need a test
rather than a comment.

Backbones are built with pretrained=False; nothing here downloads weights.
"""

import pytest
import torch

import config
from models import (
    CONVNEXTV2_T_WEIGHT_TAG,
    MODEL_BUILDERS,
    VIT_S_WEIGHT_TAG,
    build_model,
)

ARCHITECTURES = list(config.SUPPORTED_MODELS)


class TestFactory:
    def test_the_factory_and_the_roster_agree(self):
        assert set(MODEL_BUILDERS) == set(config.SUPPORTED_MODELS)

    def test_rejects_an_unknown_model(self):
        with pytest.raises(ValueError, match="Unknown model"):
            build_model("resnet50", pretrained=False)

    @pytest.mark.parametrize("name", ARCHITECTURES)
    def test_every_architecture_emits_one_logit_per_pathology(self, name, model_cache):
        images = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        with torch.no_grad():
            logits = model_cache(name)(images)
        assert logits.shape == (2, config.NUM_CLASSES)


class TestWeightTags:
    def test_no_timm_tag_pulls_in_21k_pretraining(self):
        """
        Pretraining data is the variable the roster is controlled on. DenseNet,
        SwinV2 and MaxViT have no IN21k option here, so a 21k tag on either timm
        model would hand it training data no other backbone got — and would
        flatter the pure ViT in particular.
        """
        for tag in (VIT_S_WEIGHT_TAG, CONVNEXTV2_T_WEIGHT_TAG):
            assert "in21k" not in tag and "in22k" not in tag, tag


class TestWeightTagOverride:
    """
    `--weight-tag` is the one sanctioned way past the IN1k-only invariant, for
    the pretraining-data ablation. Every guard below exists because its failure
    mode is silent: the run trains, converges, and reports a number that is
    simply wrong about what it measured.
    """

    def test_override_builds_a_working_model(self):
        model = build_model(
            "vit_s_16",
            pretrained=False,
            weight_tag="deit3_small_patch16_224.fb_in22k_ft_in1k",
        )
        images = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        with torch.no_grad():
            assert model(images).shape == (2, config.NUM_CLASSES)

    def test_the_in22k_tag_keeps_the_structure_the_xai_code_assumes(self):
        """
        The ablation swaps checkpoints, not architectures — so the Grad-CAM
        reshape and rollout must still find one prefix token and 12 blocks.
        """
        model = build_model(
            "vit_s_16",
            pretrained=False,
            weight_tag="deit3_small_patch16_224.fb_in22k_ft_in1k",
        )
        assert model.backbone.num_prefix_tokens == 1
        assert len(model.backbone.blocks) == 12

    def test_the_ablation_pair_differs_only_in_pretraining_data(self):
        """
        What makes the default ViT-S run a valid control for the 22k run: same
        architecture, same recipe family, same input statistics. If a timm
        release ever repoints either tag, this failing is the signal.
        """
        import timm

        in1k = timm.get_pretrained_cfg(VIT_S_WEIGHT_TAG)
        in22k = timm.get_pretrained_cfg("deit3_small_patch16_224.fb_in22k_ft_in1k")
        assert in1k.mean == in22k.mean == tuple(config.IMAGENET_MEAN)
        assert in1k.std == in22k.std == tuple(config.IMAGENET_STD)
        assert in1k.input_size == in22k.input_size

    def test_rejects_a_tag_whose_normalization_differs(self):
        """
        The augreg checkpoints are JAX ports expecting mean/std = 0.5. Feeding
        them ImageNet-normalized input de-normalizes every image, and nothing
        downstream would notice.
        """
        with pytest.raises(ValueError, match="expects mean/std"):
            build_model(
                "vit_s_16",
                pretrained=False,
                weight_tag="vit_small_patch16_224.augreg_in21k_ft_in1k",
            )

    @pytest.mark.parametrize("name", ["densenet121", "swin_v2_t", "maxvit_t"])
    def test_rejects_an_override_on_the_torchvision_models(self, name):
        """
        Those three resolve weights through `.DEFAULT` enums. Accepting a tag
        and ignoring it would report an ablation that never happened.
        """
        with pytest.raises(ValueError, match="only supported for"):
            build_model(
                name, pretrained=False, weight_tag="deit3_small_patch16_224.fb_in1k"
            )

    def test_rejects_an_unknown_tag(self):
        with pytest.raises(ValueError, match="Unknown timm weight tag"):
            build_model("vit_s_16", pretrained=False, weight_tag="not_a_real_tag")


class TestArchitectureAssumptions:
    def test_vit_has_exactly_one_prefix_token(self, model_cache):
        """
        Both the Grad-CAM reshape and Attention Rollout drop prefix tokens to
        recover a square patch grid. A distilled or register-token checkpoint
        carries more, which would shift every patch position.
        """
        assert model_cache("vit_s_16").backbone.num_prefix_tokens == 1

    def test_maxvit_is_what_fixes_the_input_resolution(self, model_cache):
        """
        MaxViT builds its attention partitions from a declared input size and
        reshapes against them, so it — not the other four — is why
        config.IMAGE_SIZE cannot be raised on its own. If a torchvision release
        ever makes this flexible, this test failing is the signal that the
        constraint lifted.
        """
        model = model_cache("maxvit_t")
        with torch.no_grad():
            model(torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE))

            with pytest.raises(RuntimeError):
                model(torch.randn(1, 3, config.IMAGE_SIZE + 32, config.IMAGE_SIZE + 32))
