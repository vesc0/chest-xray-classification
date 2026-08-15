"""
The explanation sanity check itself.

This is the check that decides whether the localization numbers mean anything,
so its own failure modes matter. Two in particular are silent: a stage list that
skips part of the network understates how much of the explanation survives
randomization, and a randomization that does not actually re-initialize anything
makes every method look like it passed.
"""

import numpy as np
import pytest
import torch

import config
from explainability import available_explainers
from sanity_checks import (
    RANDOMIZATION_STAGES,
    _matches,
    _pointing_game,
    _rank_correlation,
    randomize,
)

ARCHITECTURES = list(config.SUPPORTED_MODELS)


@pytest.fixture
def fresh_model():
    """
    A model nobody else shares.

    randomize() mutates in place, so these tests cannot use the session-scoped
    model_cache without corrupting every test that runs after them.
    """
    from models import build_model

    return build_model("densenet121", pretrained=False).eval()


class TestStageRegistry:
    def test_every_architecture_has_stages(self):
        assert set(RANDOMIZATION_STAGES) == set(config.SUPPORTED_MODELS)

    @pytest.mark.parametrize("name", ARCHITECTURES)
    def test_every_prefix_matches_something(self, name, model_cache):
        """
        A prefix that matches nothing is a stage that randomizes nothing, and
        the run would report the explanation surviving a step that never
        happened.
        """
        parameter_names = [n for n, _ in model_cache(name).named_parameters()]
        for label, prefixes in RANDOMIZATION_STAGES[name]:
            for prefix in prefixes:
                assert any(_matches(n, [prefix]) for n in parameter_names), (
                    f"{name}/{label}: prefix '{prefix}' matches no parameter"
                )

    @pytest.mark.parametrize("name", ARCHITECTURES)
    def test_the_stages_cover_the_whole_model(self, name, model_cache):
        """
        Cascading randomization ends with a fully random network. A parameter no
        stage touches stays trained through every step, so the final row would
        not be the fully-randomized control it is read as.
        """
        parameter_names = [n for n, _ in model_cache(name).named_parameters()]
        covered = {
            n for n in parameter_names
            for _, prefixes in RANDOMIZATION_STAGES[name]
            if _matches(n, prefixes)
        }
        assert covered == set(parameter_names)

    @pytest.mark.parametrize("name", ARCHITECTURES)
    def test_the_stages_are_disjoint(self, name, model_cache):
        """
        Overlapping stages would randomize a block twice and report the second
        pass as new information.
        """
        parameter_names = [n for n, _ in model_cache(name).named_parameters()]
        for n in parameter_names:
            hits = [
                label for label, prefixes in RANDOMIZATION_STAGES[name]
                if _matches(n, prefixes)
            ]
            assert len(hits) == 1, f"{name}: '{n}' claimed by {hits}"

    @pytest.mark.parametrize("name", ARCHITECTURES)
    def test_the_classifier_is_randomized_first(self, name):
        """
        Cascading means top-down. Starting anywhere else measures a different
        thing from the paper's test.
        """
        first_label, _ = RANDOMIZATION_STAGES[name][0]
        assert first_label in {"head", "classifier"}


class TestMatches:
    def test_matches_an_exact_name(self):
        assert _matches("backbone.cls_token", ["backbone.cls_token"])

    def test_matches_a_descendant(self):
        assert _matches("backbone.blocks.1.attn.qkv.weight", ["backbone.blocks.1"])

    def test_does_not_match_a_sibling_sharing_a_prefix_string(self):
        """`blocks.1` must not swallow `blocks.11` — the dot is load-bearing."""
        assert not _matches("backbone.blocks.11.attn.qkv.weight", ["backbone.blocks.1"])

    def test_the_root_module_matches_nothing(self):
        assert not _matches("", ["backbone"])


class TestRandomize:
    def test_it_actually_changes_the_parameters(self, fresh_model):
        before = fresh_model.backbone.classifier[1].weight.detach().clone()
        randomize(fresh_model, ["backbone.classifier"])
        assert not torch.equal(before, fresh_model.backbone.classifier[1].weight)

    def test_it_leaves_everything_else_alone(self, fresh_model):
        untouched = fresh_model.backbone.features.conv0.weight.detach().clone()
        randomize(fresh_model, ["backbone.classifier"])
        assert torch.equal(untouched, fresh_model.backbone.features.conv0.weight)

    def test_it_resets_batchnorm_running_statistics(self, fresh_model):
        """
        reset_parameters() does not touch these. Trained statistics on random
        weights is neither the trained model nor a random one.
        """
        norm = fresh_model.backbone.features.norm0
        with torch.no_grad():
            norm.running_mean.fill_(5.0)
        randomize(fresh_model, ["backbone.features.norm0"])
        assert torch.allclose(norm.running_mean, torch.zeros_like(norm.running_mean))

    def test_bare_parameters_are_randomized_too(self, model_cache):
        """
        ViT's class token and position embedding have no reset_parameters, so
        the module-level pass cannot reach them and the fallback has to.
        """
        from models import build_model

        model = build_model("vit_s_16", pretrained=False).eval()
        before = model.backbone.pos_embed.detach().clone()
        randomize(model, ["backbone.pos_embed"])
        assert not torch.equal(before, model.backbone.pos_embed)

    def test_it_reports_how_many_tensors_it_touched(self, fresh_model):
        count = randomize(fresh_model, ["backbone.classifier"])
        assert count == 2  # weight and bias

    def test_an_empty_prefix_list_is_an_error(self, fresh_model):
        """Silently randomizing nothing would read as the explanation surviving."""
        with pytest.raises(ValueError, match="No parameters matched"):
            randomize(fresh_model, ["backbone.does_not_exist"])

    def test_randomizing_a_stage_changes_the_output(self, fresh_model):
        images = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        with torch.no_grad():
            before = fresh_model(images)
            randomize(fresh_model, ["backbone.features.denseblock4"])
            after = fresh_model(images)
        assert not torch.allclose(before, after)


class TestRankCorrelation:
    def test_a_map_correlates_perfectly_with_itself(self, rng):
        heatmap = rng.random((16, 16))
        assert _rank_correlation(heatmap, heatmap) == pytest.approx(1.0)

    def test_an_inverted_map_correlates_negatively(self, rng):
        heatmap = rng.random((16, 16))
        assert _rank_correlation(heatmap, 1.0 - heatmap) == pytest.approx(-1.0)

    def test_a_blank_map_is_excluded_rather_than_scored(self):
        """
        scipy returns NaN for a constant input. Averaging that in poisons the
        column; calling it zero would credit a method that collapsed to blank
        with having decorrelated.
        """
        assert _rank_correlation(np.zeros((8, 8)), np.random.random((8, 8))) is None
        assert _rank_correlation(np.random.random((8, 8)), np.zeros((8, 8))) is None


class TestPointingGame:
    def test_a_peak_inside_the_box_is_a_hit(self):
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:5, 2:5] = True
        heatmap = np.zeros((8, 8))
        heatmap[3, 3] = 1.0

        hit, baseline = _pointing_game(np.array([heatmap]), [0], {0: mask})
        assert hit == 1.0
        assert baseline == pytest.approx(9 / 64)

    def test_a_blank_map_is_a_miss_not_a_point_at_the_corner(self):
        """
        A signal-free map still has an argmax, at index 0. If a box reaches the
        top-left corner, scoring that argmax would turn an absent explanation
        into a hit.
        """
        mask = np.zeros((8, 8), dtype=bool)
        mask[0:3, 0:3] = True  # includes (0, 0)

        hit, _ = _pointing_game(np.zeros((1, 8, 8)), [0], {0: mask})
        assert hit == 0.0


class TestCoverage:
    def test_every_explainer_gets_checked(self, model_cache):
        """
        The question is asked of the method, not the model — rollout is built
        from attention rather than gradients and has no reason to behave like
        Grad-CAM here, which is exactly why it must not be skipped.
        """
        for name in ARCHITECTURES:
            assert name in RANDOMIZATION_STAGES
            assert available_explainers(name)
