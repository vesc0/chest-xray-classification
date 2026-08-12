"""
Loss, freezing, and checkpoint-metric direction.

The tuning-mode tests build real backbones (randomly initialized, no download)
because the head/backbone split is resolved by attribute name per architecture
— exactly the thing a torchvision upgrade renames without warning.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

import config
from train import (
    AsymmetricLoss,
    _apply_tuning_mode,
    _build_loss,
    _count_trainable_params,
    _disable_frozen_norm_updates,
    _get_monitored_metric,
    _split_backbone_head_params,
)

ARCHITECTURES = ["densenet121", "efficientnet_b4", "vit_b_16", "swin_v2_b"]


class TestAsymmetricLoss:
    def test_penalizes_wrong_predictions_more_than_right_ones(self):
        loss = AsymmetricLoss()
        targets = torch.tensor([[1.0, 0.0]])
        right = loss(torch.tensor([[5.0, -5.0]]), targets)
        wrong = loss(torch.tensor([[-5.0, 5.0]]), targets)
        assert wrong > right

    def test_stays_finite_at_saturating_logits(self):
        loss = AsymmetricLoss()
        targets = torch.tensor([[1.0, 0.0]])
        for logits in ([[100.0, -100.0]], [[-100.0, 100.0]]):
            assert torch.isfinite(loss(torch.tensor(logits), targets))

    def test_survives_half_precision_input(self):
        """
        Under CUDA autocast the logits arrive as fp16, where eps underflows to
        zero and the log terms become -inf. The loss casts to fp32 first; this
        is the guard on that cast.
        """
        loss = AsymmetricLoss()
        logits = torch.tensor([[8.0, -8.0]], dtype=torch.float16)
        targets = torch.tensor([[1.0, 0.0]], dtype=torch.float16)
        value = loss(logits, targets)
        assert torch.isfinite(value)
        assert value.dtype == torch.float32

    def test_gradients_flow(self):
        loss = AsymmetricLoss()
        logits = torch.tensor([[0.5, -0.5]], requires_grad=True)
        loss(logits, torch.tensor([[1.0, 0.0]])).backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    def test_down_weights_negatives_more_than_positives(self):
        # gamma_neg > gamma_pos is the whole point of the asymmetry
        symmetric = AsymmetricLoss(gamma_neg=1.0, gamma_pos=1.0)
        asymmetric = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0)
        logits = torch.tensor([[0.0, 0.0]])
        targets = torch.tensor([[1.0, 0.0]])
        assert asymmetric(logits, targets) < symmetric(logits, targets)


class TestBuildLoss:
    def test_rejects_an_unknown_loss(self):
        config.LOSS_NAME = "nonsense"
        with pytest.raises(ValueError, match="Unsupported loss"):
            _build_loss(None, torch.device("cpu"))

    def test_builds_each_supported_loss(self):
        config.LOSS_NAME = "asymmetric"
        assert isinstance(_build_loss(None, torch.device("cpu")), AsymmetricLoss)
        config.LOSS_NAME = "bce"
        assert isinstance(_build_loss(None, torch.device("cpu")), nn.BCEWithLogitsLoss)


class TestParameterSplit:
    @pytest.mark.parametrize("name", ARCHITECTURES)
    def test_head_is_found_for_every_architecture(self, name, model_cache):
        backbone, head = _split_backbone_head_params(model_cache(name), )
        assert head, f"no classifier head located for {name}"
        assert backbone

    @pytest.mark.parametrize("name", ARCHITECTURES)
    def test_split_is_a_partition(self, name, model_cache):
        model = model_cache(name)
        backbone, head = _split_backbone_head_params(model)
        ids = {id(p) for p in backbone} | {id(p) for p in head}
        assert len(ids) == len(backbone) + len(head)  # disjoint
        assert ids == {id(p) for p in model.parameters()}  # exhaustive


class TestTuningModes:
    def test_full_trains_everything(self, model_cache):
        model = model_cache("densenet121")
        backbone, head = _split_backbone_head_params(model)
        config.TUNING_MODE = "full"
        assert _apply_tuning_mode(model, 1, backbone, head) == "full"
        assert all(p.requires_grad for p in backbone + head)

    def test_head_only_freezes_the_backbone(self, model_cache):
        model = model_cache("densenet121")
        backbone, head = _split_backbone_head_params(model)
        config.TUNING_MODE = "head_only"
        assert _apply_tuning_mode(model, 1, backbone, head) == "head_only"
        assert not any(p.requires_grad for p in backbone)
        assert all(p.requires_grad for p in head)

    def test_partial_warms_up_on_the_head_then_unfreezes(self, model_cache):
        model = model_cache("densenet121")
        backbone, head = _split_backbone_head_params(model)
        config.TUNING_MODE = "partial"
        config.FREEZE_EPOCHS = 3
        config.PARTIAL_UNFREEZE_FRACTION = 0.3

        assert _apply_tuning_mode(model, 1, backbone, head) == "partial_head_warmup"
        assert not any(p.requires_grad for p in backbone)

        assert _apply_tuning_mode(model, 4, backbone, head) == "partial_unfrozen"
        unfrozen = sum(p.requires_grad for p in backbone)
        assert 0 < unfrozen < len(backbone)

    def test_partial_unfreezes_the_top_of_the_backbone(self, model_cache):
        model = model_cache("densenet121")
        backbone, head = _split_backbone_head_params(model)
        config.TUNING_MODE = "partial"
        config.FREEZE_EPOCHS = 0
        config.PARTIAL_UNFREEZE_FRACTION = 0.25
        _apply_tuning_mode(model, 1, backbone, head)
        # Later layers, not earlier ones
        assert backbone[-1].requires_grad
        assert not backbone[0].requires_grad

    def test_head_only_reduces_the_trainable_count(self, model_cache):
        model = model_cache("densenet121")
        backbone, head = _split_backbone_head_params(model)

        config.TUNING_MODE = "full"
        _apply_tuning_mode(model, 1, backbone, head)
        full = _count_trainable_params(model)

        config.TUNING_MODE = "head_only"
        _apply_tuning_mode(model, 1, backbone, head)
        assert _count_trainable_params(model) < full

    def test_rejects_an_unknown_tuning_mode(self, model_cache):
        model = model_cache("densenet121")
        backbone, head = _split_backbone_head_params(model)
        config.TUNING_MODE = "nonsense"
        with pytest.raises(ValueError, match="Unsupported tuning mode"):
            _apply_tuning_mode(model, 1, backbone, head)


class TestFrozenBatchNorm:
    def test_frozen_batchnorm_stops_updating_its_statistics(self, model_cache):
        """
        requires_grad=False stops weight updates but not running statistics:
        model.train() switches the whole tree, so a "frozen" backbone would
        still drift onto radiograph statistics.
        """
        model = model_cache("densenet121")
        backbone, head = _split_backbone_head_params(model)
        config.TUNING_MODE = "head_only"
        _apply_tuning_mode(model, 1, backbone, head)

        model.train()
        frozen = _disable_frozen_norm_updates(model)
        assert frozen > 0

        norms = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        assert not any(m.training for m in norms)

    def test_is_idempotent(self, model_cache):
        model = model_cache("densenet121")
        backbone, head = _split_backbone_head_params(model)
        config.TUNING_MODE = "head_only"
        _apply_tuning_mode(model, 1, backbone, head)
        model.train()
        assert _disable_frozen_norm_updates(model) == _disable_frozen_norm_updates(model)

    def test_leaves_trainable_batchnorm_alone(self, model_cache):
        model = model_cache("densenet121")
        backbone, head = _split_backbone_head_params(model)
        config.TUNING_MODE = "full"
        _apply_tuning_mode(model, 1, backbone, head)
        model.train()
        assert _disable_frozen_norm_updates(model) == 0


class TestMonitoredMetric:
    def test_loss_is_minimized_the_others_maximized(self):
        metrics = {"loss": 0.4, "auroc": 0.8, "auprc": 0.3}

        config.CHECKPOINT_METRIC = "val_loss"
        assert _get_monitored_metric(metrics) == (0.4, True)

        config.CHECKPOINT_METRIC = "val_auroc"
        assert _get_monitored_metric(metrics) == (0.8, False)

        config.CHECKPOINT_METRIC = "val_auprc"
        assert _get_monitored_metric(metrics) == (0.3, False)

    def test_rejects_an_unknown_checkpoint_metric(self):
        config.CHECKPOINT_METRIC = "nonsense"
        with pytest.raises(ValueError, match="Unsupported checkpoint metric"):
            _get_monitored_metric({"loss": 0.1, "auroc": 0.5, "auprc": 0.5})
