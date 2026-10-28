"""Tests for boltz.model.loss.bfactor — bfactor_loss_fn."""

import math

import torch
import pytest

BINS = 20
BOUNDARIES = torch.logspace(math.log10(5.0), math.log10(3000.0), BINS - 1)


class TestBfactorLoss:
    """Tests for bfactor_loss_fn."""

    def test_perfect_prediction(self, small_feats_dict):
        """Matching bins → loss ≈ 0."""
        from boltz.model.loss.bfactor import bfactor_loss_fn

        B, N_tok, N_atom = 1, 8, 16

        # Compute what the target bins would be
        bfactor_atom = small_feats_dict["bfactor"].unsqueeze(-1)
        token_to_rep_atom = small_feats_dict["token_to_rep_atom"].float()
        bfactor_token = torch.bmm(token_to_rep_atom, bfactor_atom)
        target_bin = (bfactor_token > BOUNDARIES).sum(dim=-1).long()

        # Create logits that perfectly match targets
        logits = torch.full((B, N_tok, BINS), -10.0)
        for b in range(B):
            for t in range(N_tok):
                logits[b, t, target_bin[b, t]] = 10.0

        output = {"pbfactor": logits}
        loss = bfactor_loss_fn(output, small_feats_dict, BOUNDARIES)
        assert loss.item() < 0.1

    def test_zero_masked(self, small_feats_dict):
        """Atoms with bfactor=0 are masked out."""
        from boltz.model.loss.bfactor import bfactor_loss_fn

        B, N_tok, N_atom = 1, 8, 16

        # Set all bfactors to 0 (they get masked by token_mask = bfactor > 1e-5)
        feats = {**small_feats_dict, "bfactor": torch.zeros(B, N_atom)}
        logits = torch.randn(B, N_tok, BINS)
        output = {"pbfactor": logits}

        loss = bfactor_loss_fn(output, feats, BOUNDARIES)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_finiteness(self, small_feats_dict):
        """Random inputs → finite loss."""
        from boltz.model.loss.bfactor import bfactor_loss_fn

        B, N_tok = 1, 8
        logits = torch.randn(B, N_tok, BINS)
        output = {"pbfactor": logits}

        loss = bfactor_loss_fn(output, small_feats_dict, BOUNDARIES)
        assert torch.isfinite(loss)


class TestBFactorModule:
    """Tests for the log-spaced bin edges held by BFactorModule."""

    def test_boundaries_are_log_spaced_and_saved(self):
        from boltz.model.modules.trunkv2 import BFactorModule

        module = BFactorModule(token_s=4, num_bins=BINS, b_min=5.0, b_max=3000.0)

        assert module.boundaries.shape == (BINS - 1,)
        assert module.boundaries[0].item() == pytest.approx(5.0, rel=1e-5)
        assert module.boundaries[-1].item() == pytest.approx(3000.0, rel=1e-5)
        ratios = module.boundaries[1:] / module.boundaries[:-1]
        assert torch.allclose(ratios, ratios[0].expand_as(ratios))
        assert "boundaries" in module.state_dict()
