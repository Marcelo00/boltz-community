"""Tests for the fixed validation noise of the MD validator."""

from types import SimpleNamespace

import pytest
import torch

from boltz.model.modules.utils import random_rotations
from boltz.model.validation.md import MDValidator

DEVICES = ["cpu", *(["cuda"] if torch.cuda.is_available() else [])]


class _Model(torch.nn.Module):
    """Draw the same kinds of random numbers as the diffusion sampler."""

    validation_args = SimpleNamespace(
        recycling_steps=1, sampling_steps=1, diffusion_samples=1
    )

    def forward(self, batch, **kwargs):
        device = batch["coords"].device
        noise = torch.randn(4, device=device)
        rotation = random_rotations(1, device=device).flatten()
        return torch.cat([noise, rotation])


def _validate(validator: MDValidator, key: int, device: str) -> torch.Tensor:
    batch = {
        "coords": torch.zeros(1, 1, 3, device=device),
        "record_key": torch.tensor([key]),
    }
    return validator.run_model(_Model(), batch, 0)


def _train_draw(device: str) -> torch.Tensor:
    return torch.randn(4, device=device)


@pytest.mark.parametrize("device", DEVICES)
def test_validation_noise_is_fixed_per_record(device):
    validator = MDValidator(["md_val"], seed=0)
    first = _validate(validator, 7, device)
    _train_draw(device)
    assert torch.equal(first, _validate(validator, 7, device))


@pytest.mark.parametrize("device", DEVICES)
def test_validation_noise_differs_between_records_and_seeds(device):
    base = _validate(MDValidator(["md_val"], seed=0), 7, device)
    assert not torch.equal(base, _validate(MDValidator(["md_val"], seed=0), 8, device))
    assert not torch.equal(base, _validate(MDValidator(["md_val"], seed=1), 7, device))


@pytest.mark.parametrize("device", DEVICES)
def test_training_stream_is_unchanged_by_validation(device):
    validator = MDValidator(["md_val"], seed=0)
    torch.manual_seed(123)
    expected = [_train_draw(device) for _ in range(3)]

    torch.manual_seed(123)
    actual = [_train_draw(device)]
    _validate(validator, 7, device)
    actual.append(_train_draw(device))
    _validate(validator, 8, device)
    actual.append(_train_draw(device))

    for want, got in zip(expected, actual):
        assert torch.equal(want, got)


@pytest.mark.parametrize("device", DEVICES)
def test_training_noise_still_varies_after_validation(device):
    validator = MDValidator(["md_val"], seed=0)
    torch.manual_seed(123)
    _validate(validator, 7, device)
    first = _train_draw(device)
    _validate(validator, 7, device)
    second = _train_draw(device)
    assert not torch.equal(first, second)


def test_batch_of_several_records_is_rejected():
    batch = {"coords": torch.zeros(2, 1, 3), "record_key": torch.tensor([7, 8])}
    with pytest.raises(ValueError, match="only one element"):
        MDValidator(["md_val"]).run_model(_Model(), batch, 0)
