"""Tests for the shipped training configurations."""

from pathlib import Path

import omegaconf
import pytest

from boltz.model.validation import md_metrics as mm
from boltz.model.validation.md import MEAN_METRICS, RECORD_TYPES

CONFIG_DIR = Path(__file__).parents[1] / "scripts" / "train" / "configs_v3"


def logged_metrics() -> set[str]:
    """Every metric name the MD validator can log."""
    names = set(MEAN_METRICS) | set(mm.METRIC_NAMES)
    names |= {f"{name}_{label}" for name in mm.METRIC_NAMES for label in RECORD_TYPES}
    return {f"val/{name}" for name in names}


@pytest.mark.parametrize("config", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_the_monitored_metric_is_logged_by_the_validator(config):
    cfg = omegaconf.OmegaConf.load(config)
    assert cfg.monitor in logged_metrics()
    assert cfg.monitor_mode in ("min", "max")
