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
