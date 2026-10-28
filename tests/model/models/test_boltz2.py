from types import SimpleNamespace

import pytest
from torch import nn

from boltz.model.models import boltz2


class _DummyModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


def _minimal_boltz2(validate_structure, **kwargs):
    return boltz2.Boltz2(
        atom_s=2,
        atom_z=2,
        token_s=2,
        token_z=2,
        num_bins=4,
        training_args=kwargs.pop("training_args", {}),
        validation_args={},
        embedder_args={},
        msa_args={},
        pairformer_args={},
        score_model_args={
            "atom_encoder_depth": 1,
            "atom_encoder_heads": 1,
            "token_transformer_depth": 1,
            "token_transformer_heads": 1,
            "atom_decoder_depth": 1,
            "atom_decoder_heads": 1,
            "conditioning_transition_layers": 1,
        },
        diffusion_process_args={},
        diffusion_loss_args={},
        confidence_prediction=False,
        validate_structure=validate_structure,
        validators=[],
        **kwargs,
    )


@pytest.fixture
def dummy_submodules(monkeypatch):
    for module_name in [
        "InputEmbedder",
        "RelativePositionEncoder",
        "ContactConditioning",
        "MSAModule",
        "PairformerModule",
        "DiffusionConditioning",
        "AtomDiffusion",
        "DistogramModule",
    ]:
        monkeypatch.setattr(boltz2, module_name, _DummyModule)


def test_boltz2_stores_validate_structure(dummy_submodules):
    for validate_structure in [False, True]:
        model = _minimal_boltz2(validate_structure)

        assert model.validate_structure is validate_structure


def _bfactor_model():
    # on_load_checkpoint also resets the lr and weight decay from the training args
    training_args = SimpleNamespace(max_lr=1e-4, weight_decay=0.0)
    return _minimal_boltz2(False, predict_bfactor=True, training_args=training_args)


def _checkpoint_with(model, boundaries=None):
    """A checkpoint holding the bfactor head, with or without stored edges."""
    state_dict = {
        key: value.clone()
        for key, value in model.state_dict().items()
        if key.startswith("bfactor_module.")
    }
    if boundaries is None:
        del state_dict["bfactor_module.boundaries"]
    else:
        state_dict["bfactor_module.boundaries"] = boundaries
    return {"state_dict": state_dict}


def test_load_checkpoint_drops_bfactor_head_without_boundaries(dummy_submodules):
    model = _bfactor_model()
    checkpoint = _checkpoint_with(model)

    model.on_load_checkpoint(checkpoint)

    assert not any(k.startswith("bfactor_module.") for k in checkpoint["state_dict"])


def test_load_checkpoint_keeps_bfactor_head_with_matching_boundaries(
    dummy_submodules,
):
    model = _bfactor_model()
    checkpoint = _checkpoint_with(model, model.bfactor_module.boundaries.clone())

    model.on_load_checkpoint(checkpoint)

    assert "bfactor_module.bfactor.weight" in checkpoint["state_dict"]


def test_load_checkpoint_rejects_different_boundaries(dummy_submodules):
    model = _bfactor_model()
    checkpoint = _checkpoint_with(model, model.bfactor_module.boundaries * 2)

    with pytest.raises(AssertionError, match="bin edges"):
        model.on_load_checkpoint(checkpoint)
