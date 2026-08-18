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


class TestXRVWeights:
    """
    The medical-pretraining arm. Two things need pinning: that the input
    adapter reproduces XRV's own normalization exactly, and that no checkpoint
    trained on ChestX-ray14 can be selected. Both fail silently otherwise — the
    first as a converged run trained on de-normalized images, the second as a
    test score that is really a training score.
    """

    def test_the_adapter_reproduces_xrv_normalization(self):
        """
        The load-bearing claim of DenseNet121XRVClassifier: the shared
        ImageNet-normalized transform can feed an XRV checkpoint without a
        second DataLoader, because undoing one affine map and applying another
        is exact. Compared against xrv's own function rather than against a
        re-derivation of the same arithmetic.
        """
        import numpy as np
        from PIL import Image
        from torchvision import transforms

        # Through the project's guarded importer, not a bare `import`: a bare
        # one here would leave torchxrayvision's folders on sys.path and fail
        # test_importing_it_does_not_shadow_the_project_modules below, for a
        # reason that has nothing to do with the code under test.
        xrv = _import_torchxrayvision()

        pixels = np.random.default_rng(0).integers(0, 256, (32, 32), dtype=np.uint8)
        # convert("RGB") is what dataset.py does to these grayscale radiographs
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
        """
        Every blocked tag was trained on ChestX-ray14 or on a dataset derived
        from it, so it has already seen images this pipeline reports as held
        out. Nothing downstream can detect that, which is why it is blocked at
        construction.
        """
        with pytest.raises(ValueError, match="leaks the evaluation set"):
            build_model("densenet121_xrv", pretrained=False, weight_tag=tag)

    def test_importing_it_does_not_shadow_the_project_modules(self):
        """
        torchxrayvision's vendored baseline models `sys.path.insert(0, ...)`
        their own folders, one of which holds a package named `config`. Ahead
        of the project root, that makes `import config` resolve to theirs in
        any interpreter that has not already imported ours — which is every
        spawned DataLoader worker. The symptom lands nowhere near the cause:
        workers die on `module 'config' has no attribute 'SEED'` and the parent
        reports BrokenPipeError.
        """
        import sys

        build_model("densenet121_xrv", pretrained=False)

        intruders = [entry for entry in sys.path if "torchxrayvision" in entry]
        assert not intruders, f"torchxrayvision left {intruders} on sys.path"
        assert sys.modules["config"].__file__ == config.__file__

    def test_the_default_corpus_is_leakage_free(self):
        assert XRV_DEFAULT_WEIGHT_TAG not in XRV_LEAKY_WEIGHTS

    def test_every_available_tag_is_classified(self):
        """
        A new XRV release adding a checkpoint should not default to "allowed".
        Each tag is either on the blocklist or asserted disjoint from NIH here,
        so an unreviewed one fails this test instead of quietly training.
        """
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
        """
        `--model all` compares architectures. This is DenseNet-121 again with
        different weights, so sweeping it would put the same architecture in
        the comparison table twice.
        """
        assert "densenet121_xrv" in config.SUPPORTED_MODELS
        assert "densenet121_xrv" not in config.SWEEP_MODELS
        assert set(config.SWEEP_MODELS) < set(config.SUPPORTED_MODELS)


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
