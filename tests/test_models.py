"""
The model factory and the architecture assumptions built on top of it.

Pins two things: the factory's roster against config.SUPPORTED_MODELS, since a
mismatch only fails once a sweep reaches that model, and the structural facts
the XAI code relies on, which are properties of third-party checkpoints.

Backbones are built with pretrained=False; nothing here downloads weights.
"""

import pytest
import torch

import config
from models import (
    CONVNEXTV2_T_WEIGHT_TAG,
    MODEL_BUILDERS,
    VIT_S_WEIGHT_TAG,
    XRV_DEFAULT_WEIGHT_TAG,
    XRV_LEAKY_WEIGHTS,
    _import_torchxrayvision,
    _xrv_densenet_tags,
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
        """A 21k tag would hand one model training data no other backbone got."""
        for tag in (VIT_S_WEIGHT_TAG, CONVNEXTV2_T_WEIGHT_TAG):
            assert "in21k" not in tag and "in22k" not in tag, tag


class TestWeightTagOverride:
    """
    The one sanctioned way past the IN1k-only invariant. Every guard below
    exists because its failure mode is a converged run reporting a number that
    is simply wrong about what it measured.
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
        """The ablation swaps checkpoints, not architectures, so the XAI code
        must still find one prefix token and 12 blocks."""
        model = build_model(
            "vit_s_16",
            pretrained=False,
            weight_tag="deit3_small_patch16_224.fb_in22k_ft_in1k",
        )
        assert model.backbone.num_prefix_tokens == 1
        assert len(model.backbone.blocks) == 12

    def test_the_ablation_pair_differs_only_in_pretraining_data(self):
        """What makes the IN1k run a valid control for the 22k one. If a timm
        release repoints either tag, this failing is the signal."""
        import timm

        in1k = timm.get_pretrained_cfg(VIT_S_WEIGHT_TAG)
        in22k = timm.get_pretrained_cfg("deit3_small_patch16_224.fb_in22k_ft_in1k")
        assert in1k.mean == in22k.mean == tuple(config.IMAGENET_MEAN)
        assert in1k.std == in22k.std == tuple(config.IMAGENET_STD)
        assert in1k.input_size == in22k.input_size

    def test_rejects_a_tag_whose_normalization_differs(self):
        """The augreg JAX ports want mean/std = 0.5; feeding them
        ImageNet-normalized input de-normalizes every image, silently."""
        with pytest.raises(ValueError, match="expects mean/std"):
            build_model(
                "vit_s_16",
                pretrained=False,
                weight_tag="vit_small_patch16_224.augreg_in21k_ft_in1k",
            )

    @pytest.mark.parametrize("name", ["densenet121", "swin_v2_t", "maxvit_t"])
    def test_rejects_an_override_on_the_torchvision_models(self, name):
        """They resolve weights through `.DEFAULT`; accepting a tag and
        ignoring it would report an ablation that never happened."""
        with pytest.raises(ValueError, match="only supported for"):
            build_model(
                name, pretrained=False, weight_tag="deit3_small_patch16_224.fb_in1k"
            )

    def test_rejects_an_unknown_tag(self):
        with pytest.raises(ValueError, match="Unknown timm weight tag"):
            build_model("vit_s_16", pretrained=False, weight_tag="not_a_real_tag")


class TestXRVWeights:
    """
    The medical-pretraining arm. Pins that the input adapter reproduces XRV's
    normalization exactly, and that no ChestX-ray14-trained checkpoint can be
    selected — a converged run on de-normalized images, and a test score that
    is really a training score, respectively.
    """

    def test_the_adapter_reproduces_xrv_normalization(self):
        """
        The load-bearing claim: the shared transform can feed an XRV checkpoint
        without a second DataLoader, because the conversion is exact. Compared
        against xrv's own function, not a re-derivation of the same arithmetic.
        """
        import numpy as np
        from PIL import Image
        from torchvision import transforms

        # The guarded importer, not a bare `import`, which would leave XRV's
        # folders on sys.path and fail the shadowing test below.
        xrv = _import_torchxrayvision()

        pixels = np.random.default_rng(0).integers(0, 256, (32, 32), dtype=np.uint8)
        # What dataset.py does to these grayscale radiographs.
        image = Image.fromarray(pixels, mode="L").convert("RGB")
        shared_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
            ]
        )

        model = build_model("densenet121_xrv", pretrained=False)
        ours = model._to_xrv_scale(shared_transform(image).unsqueeze(0))
        theirs = torch.from_numpy(xrv.utils.normalize(pixels.astype("float32"), 255))

        assert ours.shape == (1, 1, 32, 32)
        # Tolerance is float32 rounding on a [-1024, 1024] scale, not slack.
        assert torch.allclose(ours[0, 0], theirs, atol=1e-3)

    @pytest.mark.parametrize("tag", sorted(XRV_LEAKY_WEIGHTS))
    def test_rejects_weights_that_saw_the_test_split(self, tag):
        """Every blocked tag has already seen images this pipeline reports as
        held out, and nothing downstream could detect it."""
        with pytest.raises(ValueError, match="leaks the evaluation set"):
            build_model("densenet121_xrv", pretrained=False, weight_tag=tag)

    def test_importing_it_does_not_shadow_the_project_modules(self):
        """
        XRV's vendored models insert their own folders at sys.path[0], and one
        holds a package named `config`. Every spawned DataLoader worker would
        then import theirs and die on a missing SEED, surfacing in the parent
        as a BrokenPipeError nowhere near the cause.
        """
        import sys

        build_model("densenet121_xrv", pretrained=False)

        intruders = [entry for entry in sys.path if "torchxrayvision" in entry]
        assert not intruders, f"torchxrayvision left {intruders} on sys.path"
        assert sys.modules["config"].__file__ == config.__file__

    def test_the_default_corpus_is_leakage_free(self):
        assert XRV_DEFAULT_WEIGHT_TAG not in XRV_LEAKY_WEIGHTS

    def test_every_available_tag_is_classified(self):
        """A new XRV checkpoint must not default to "allowed": each tag is
        either blocked or asserted NIH-disjoint here."""
        reviewed_clean = {
            "densenet121-res224-chex",  # CheXpert, Stanford
            "densenet121-res224-pc",  # PadChest, Spain
            "densenet121-res224-mimic_ch",  # MIMIC-CXR, BIDMC
            "densenet121-res224-mimic_nb",  # MIMIC-CXR, BIDMC
        }
        assert set(_xrv_densenet_tags()) == reviewed_clean | set(XRV_LEAKY_WEIGHTS)

    def test_rejects_an_unknown_tag(self):
        with pytest.raises(ValueError, match="Unknown TorchXRayVision weight tag"):
            build_model("densenet121_xrv", pretrained=False, weight_tag="not_a_real_tag")

    def test_it_stays_out_of_the_architecture_sweep(self):
        """Sweeping it would enter DenseNet-121 into the architecture
        comparison table twice, once per pretraining corpus."""
        assert "densenet121_xrv" in config.SUPPORTED_MODELS
        assert "densenet121_xrv" not in config.SWEEP_MODELS
        assert set(config.SWEEP_MODELS) < set(config.SUPPORTED_MODELS)


class TestArchitectureAssumptions:
    def test_vit_has_exactly_one_prefix_token(self, model_cache):
        """Both explainers drop prefix tokens to recover a square patch grid;
        a distilled or register-token checkpoint would shift every patch."""
        assert model_cache("vit_s_16").backbone.num_prefix_tokens == 1

    def test_maxvit_is_what_fixes_the_input_resolution(self, model_cache):
        """
        MaxViT alone is why config.IMAGE_SIZE cannot be raised across the
        board. This failing is the signal that torchvision lifted it.
        """
        model = model_cache("maxvit_t")
        with torch.no_grad():
            model(torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE))

            with pytest.raises(RuntimeError):
                model(torch.randn(1, 3, config.IMAGE_SIZE + 32, config.IMAGE_SIZE + 32))
