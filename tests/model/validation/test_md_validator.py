"""Tests for boltz.model.validation.md — the head losses of MDValidator."""

import math
from types import SimpleNamespace

import pytest
import torch

from boltz.model.loss.bfactor import bfactor_loss_fn
from boltz.model.loss.distogramv2 import distogram_loss
from boltz.model.validation.md import MDValidator

BINS = 16
BOUNDARIES = torch.logspace(math.log10(5.0), math.log10(3000.0), BINS - 1)


class _Model:
    """Stands in for Boltz2: holds what the head metrics read and records the logs."""

    aggregate_distogram = True
    bfactor_module = SimpleNamespace(boundaries=BOUNDARIES)

    def __init__(self, predict_bfactor=True):
        self.predict_bfactor = predict_bfactor
        self.logged = {}

    def log(self, name, value, prog_bar=False):
        self.logged[name] = value


def _batch_and_out(small_feats_dict, with_bfactor=True):
    n_tok, frames = 8, 3
    target = torch.nn.functional.one_hot(torch.randint(0, BINS, (1, n_tok, n_tok, frames)), BINS)
    batch = {**small_feats_dict, "disto_target": target.float()}
    out = {"pdistogram": torch.randn(1, n_tok, n_tok, 1, BINS)}
    if with_bfactor:
        out["pbfactor"] = torch.randn(1, n_tok, BINS)
    return batch, out


def test_head_metrics_match_the_training_losses(small_feats_dict):
    batch, out = _batch_and_out(small_feats_dict)
    validator, model = MDValidator(["md_val"]), _Model()

    validator._process_heads(model, batch, out)
    validator.on_epoch_end(model)

    disto_loss, _ = distogram_loss(out, batch, aggregate_distogram=True)
    assert torch.isclose(model.logged["val/disto_loss"], disto_loss)
    assert torch.isclose(model.logged["val/bfactor_loss"], bfactor_loss_fn(out, batch, BOUNDARIES))


def test_bfactor_metric_is_skipped_without_the_head(small_feats_dict):
    batch, out = _batch_and_out(small_feats_dict, with_bfactor=False)
    validator, model = MDValidator(["md_val"]), _Model(predict_bfactor=False)

    validator._process_heads(model, batch, out)
    validator.on_epoch_end(model)

    assert "val/disto_loss" in model.logged
    assert "val/bfactor_loss" not in model.logged


def test_missing_bfactor_output_raises_when_the_head_is_on(small_feats_dict):
    batch, out = _batch_and_out(small_feats_dict, with_bfactor=False)

    with pytest.raises(KeyError):
        MDValidator(["md_val"])._process_heads(_Model(predict_bfactor=True), batch, out)
