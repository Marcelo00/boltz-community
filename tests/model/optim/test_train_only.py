"""Tests for boltz.model.optim.train_only — TrainOnly."""

import pytest
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from boltz.model.optim.train_only import TrainOnly


class _Toy(pl.LightningModule):
    """A frozen trunk with dropout and a trainable head, recording the modes it runs in."""

    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5))
        self.head = nn.Linear(4, 1)
        self.modes = []

    def forward(self, x):
        return self.head(self.trunk(x))

    def training_step(self, batch, batch_idx):
        self.modes.append(("train", self.trunk.training, self.head.training, self.training))
        return self(batch[0]).pow(2).mean()

    def validation_step(self, batch, batch_idx):
        self.modes.append(("val", self.trunk.training, self.head.training, self.training))

    def configure_optimizers(self):
        return torch.optim.SGD([p for p in self.parameters() if p.requires_grad], lr=0.1)


def test_setup_freezes_all_but_the_trainable_modules():
    model = _Toy()

    TrainOnly(["head"]).setup(None, model, "fit")

    assert all(not p.requires_grad for p in model.trunk.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
    assert not model.trunk.training
    assert model.head.training
    assert model.training


def test_setup_rejects_unknown_module():
    with pytest.raises(AssertionError, match="Unknown modules"):
        TrainOnly(["nope"]).setup(None, _Toy(), "fit")


def test_setup_rejects_nothing_to_train():
    with pytest.raises(AssertionError, match="nothing to train"):
        TrainOnly([]).setup(None, _Toy(), "fit")


def test_setup_rejects_parameters_outside_child_modules():
    model = _Toy()
    model.scale = nn.Parameter(torch.ones(1))

    with pytest.raises(AssertionError, match="outside child modules"):
        TrainOnly(["head"]).setup(None, model, "fit")


def test_frozen_modules_stay_frozen_through_training_and_validation():
    model = _Toy()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    data = DataLoader(TensorDataset(torch.randn(8, 4)), batch_size=2)
    trainer = pl.Trainer(
        accelerator="cpu",
        max_steps=3,
        limit_train_batches=3,
        val_check_interval=1,  # validate after every training batch
        num_sanity_val_steps=1,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        callbacks=[TrainOnly(["head"])],
    )

    trainer.fit(model, data, data)

    train_modes = [m for m in model.modes if m[0] == "train"]
    assert len(train_modes) == 3
    assert any(m[0] == "val" for m in model.modes)
    # The trunk never runs in train mode, also after the sanity check and each validation
    assert all(not trunk for _, trunk, _, _ in model.modes)
    assert all(head and top for _, _, head, top in train_modes)

    after = model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before if k.startswith("trunk"))
    assert not torch.equal(before["head.weight"], after["head.weight"])
    optimized = {id(p) for g in trainer.optimizers[0].param_groups for p in g["params"]}
    assert optimized == {id(p) for p in model.head.parameters()}
