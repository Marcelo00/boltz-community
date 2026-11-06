"""Tests for scripts/train/export_ema.py."""

import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).parents[1] / "scripts" / "train" / "export_ema.py"
spec = importlib.util.spec_from_file_location("export_ema", SCRIPT)
export_ema = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_ema)


def _checkpoint(ema_weights):
    raw = {"w": torch.zeros(3), "steps": torch.tensor(5)}
    return {
        "state_dict": raw,
        "hyper_parameters": {"token_s": 4},
        "optimizer_states": [{}],
        "lr_schedulers": [{}],
        "loops": {},
        "ema": {"cur_step": 7, "ema_weights": ema_weights},
        "callbacks": {"EMA": {"cur_step": 7, "ema_weights": ema_weights}},
    }


def test_export_replaces_the_state_dict_and_drops_the_run_state(tmp_path):
    ema = {"w": torch.ones(3), "steps": torch.tensor(5)}
    torch.save(_checkpoint(ema), tmp_path / "run.ckpt")

    export_ema.export_ema(tmp_path / "run.ckpt", tmp_path / "ema.ckpt")

    exported = torch.load(tmp_path / "ema.ckpt", weights_only=False)
    assert torch.equal(exported["state_dict"]["w"], torch.ones(3))
    assert exported["state_dict"]["steps"].dtype == torch.long
    assert exported["hyper_parameters"] == {"token_s": 4}
    for key in ("optimizer_states", "lr_schedulers", "callbacks", "ema", "loops"):
        assert key not in exported


def test_export_needs_the_ema_weights(tmp_path):
    checkpoint = _checkpoint({})
    del checkpoint["callbacks"]
    torch.save(checkpoint, tmp_path / "run.ckpt")

    with pytest.raises(ValueError, match="No EMA weights"):
        export_ema.export_ema(tmp_path / "run.ckpt", tmp_path / "ema.ckpt")


def test_export_rejects_different_keys(tmp_path):
    torch.save(_checkpoint({"w": torch.ones(3)}), tmp_path / "run.ckpt")

    with pytest.raises(ValueError, match="different keys"):
        export_ema.export_ema(tmp_path / "run.ckpt", tmp_path / "ema.ckpt")
