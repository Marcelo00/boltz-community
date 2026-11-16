"""Tests for the ensemble metrics of the MD validator on a small synthetic record."""

import math

import pytest
import torch

from boltz.data import const
from boltz.data.feature.featurizerv2 import convert_atom_name
from boltz.model.validation import md_metrics as mm
from boltz.model.validation.md import (
    NUM_REFERENCE_FRAMES,
    MDValidator,
    _reference_stride,
    build_system_index,
)

NA_ATOMS = ("P", "OP1", "OP2", "O5'", "C5'", "C4'", "C3'", "O3'", "C2'", "C1'", "N9", "C2", "C4", "C6")
ELEMENTS = {"P": 15, "O": 8, "C": 6, "N": 7}
# Eight nucleotides 5 A apart leave 11 non-neighbouring pairs within the 20 A cap, enough for
# the pair-distance metrics; the atoms of one nucleotide span less than that spacing
NUM_NT, NT_SPACING, ATOM_SPACING = 8, 5.0, 0.25
LIGAND_CHAIN_ATOMS, NUM_FRAMES, NUM_PAD = 3, 11, 2
# The structures of a record differ by a bend of the chain rather than by atom noise, so that
# long-range distances vary (the lDDT metrics need that) while bonds and links stay intact
BEND_PER_NT = 0.4
DNA, RNA, LIGAND = (const.chain_type_ids[name] for name in ("DNA", "RNA", "NONPOLYMER"))


class _Model:
    """Records the logged values."""

    def __init__(self):
        self.logged = {}

    def log(self, name, value, prog_bar=False):
        self.logged[name] = value


def _bend(coords: torch.Tensor, amplitude: torch.Tensor, nt_of_atom: torch.Tensor):
    """Displace every nucleotide perpendicular to the chain, the more the further along it is.

    Returns one structure per amplitude.
    """
    bent = coords.expand(len(amplitude), *coords.shape).clone()
    bent[..., 1] += amplitude[:, None] * BEND_PER_NT * nt_of_atom
    return bent


def _nt_of_atom(num_ligand_chains: int):
    """Nucleotide index of each atom; the ligand and the padding do not move with the chain."""
    per_nt = [float(k) for k in range(NUM_NT) for _ in NA_ATOMS]
    return torch.tensor(per_nt + [0.0] * (num_ligand_chains * LIGAND_CHAIN_ATOMS + NUM_PAD))


def _frame0(num_ligand_chains: int):
    """Nucleotides in a row with O3'(i)-P(i+1) = 1.6 A, and ligand chains 4.5 A above them."""
    coords = []
    for k in range(NUM_NT):
        for j, name in enumerate(NA_ATOMS):
            if name == "P":
                coords.append([NT_SPACING * k, 0.0, 0.0])
            elif name == "O3'":
                coords.append([NT_SPACING * (k + 1) - 1.6, 0.0, 0.0])
            else:
                angle = torch.tensor(float(j))
                coords.append([
                    NT_SPACING * k + ATOM_SPACING * j,
                    1.5 * angle.cos().item(),
                    1.5 * angle.sin().item(),
                ])
    for c in range(num_ligand_chains * LIGAND_CHAIN_ATOMS):
        coords.append([8.0 + 1.5 * c, 4.5, 0.0])
    coords += [[0.0, 0.0, 0.0]] * NUM_PAD
    return torch.tensor(coords)


def _batch(mol_type: int = DNA, key: int = 1, ordered: bool = True, num_ligand_chains: int = 2):
    """Features of one record: NUM_NT nucleotides (the first P unresolved) and bonded ligand chains."""
    num_lig = num_ligand_chains * LIGAND_CHAIN_ATOMS
    num_na = NUM_NT * len(NA_ATOMS)
    num_atoms, num_tokens = num_na + num_lig + NUM_PAD, NUM_NT + num_lig
    ligand_names = ["C1", "CA", "O3", "C4A", "C5A", "N6"][:num_lig]
    names = list(NA_ATOMS) * NUM_NT + ligand_names + [""] * NUM_PAD

    atom_token = [k for k in range(NUM_NT) for _ in NA_ATOMS] + list(range(NUM_NT, num_tokens))
    atom_to_token = torch.zeros(num_atoms, num_tokens)
    atom_to_token[torch.arange(len(atom_token)), torch.tensor(atom_token)] = 1
    rep = torch.zeros(num_tokens, num_atoms)
    center = torch.zeros(num_tokens, num_atoms)
    for k in range(NUM_NT):
        rep[k, k * len(NA_ATOMS) + NA_ATOMS.index("C4")] = 1
        center[k, k * len(NA_ATOMS) + NA_ATOMS.index("C1'")] = 1
    for t in range(num_lig):
        rep[NUM_NT + t, num_na + t] = 1
        center[NUM_NT + t, num_na + t] = 1

    pad = torch.ones(num_atoms)
    pad[-NUM_PAD:] = 0
    resolved = pad.clone()
    resolved[0] = 0  # 5' phosphate of the first nucleotide

    chars = torch.tensor([convert_atom_name(name) for name in names])
    elements = torch.tensor([ELEMENTS.get(name[:1], 0) for name in names])
    generator = torch.Generator().manual_seed(key)
    noise = 0.05 * torch.randn(NUM_FRAMES, num_atoms, 3, generator=generator)
    bend = torch.linspace(-1.0, 1.0, NUM_FRAMES)
    frames = _bend(_frame0(num_ligand_chains), bend, _nt_of_atom(num_ligand_chains)) + noise
    feats = {
        "coords": frames,
        "ensemble_ref_idxs": torch.arange(NUM_FRAMES) if ordered else torch.randperm(NUM_FRAMES),
        "atom_pad_mask": pad,
        "atom_resolved_mask": resolved,
        "atom_to_token": atom_to_token,
        "token_to_rep_atom": rep,
        "token_to_center_atom": center,
        "mol_type": torch.tensor([mol_type] * NUM_NT + [LIGAND] * num_lig),
        "asym_id": torch.tensor(
            [0] * NUM_NT + [c + 1 for c in range(num_ligand_chains) for _ in range(LIGAND_CHAIN_ATOMS)]
        ),
        "ref_atom_name_chars": torch.nn.functional.one_hot(chars, 64),
        "ref_element": torch.nn.functional.one_hot(elements, const.num_elements),
        "record_key": torch.tensor(key),
    }
    return {name: value[None] for name, value in feats.items()}


def _samples(batch, num_samples: int = 4, bend_amplitude: float = 0.8):
    """Structures spanning the same kind of bend as the record, as a stand-in for model samples.

    They stay well inside the gate thresholds, so a clean record passes them.
    """
    num_ligand_chains = (batch["mol_type"][0] == LIGAND).sum().item() // LIGAND_CHAIN_ATOMS
    generator = torch.Generator().manual_seed(0)
    noise = 0.05 * torch.randn((num_samples, *batch["coords"].shape[2:]), generator=generator)
    bend = torch.linspace(-bend_amplitude, bend_amplitude, num_samples)
    base = _frame0(num_ligand_chains)
    return _bend(base, bend, _nt_of_atom(num_ligand_chains)) + noise


def _scored(batch, samples):
    """Score a batch directly, without the validator."""
    system, atoms, _ = build_system_index(batch)
    frames = batch["coords"][0][:, atoms]
    params = mm.MetricParams()
    reference = mm.build_reference(frames, system, params)
    return mm.score_ensemble(samples[:, atoms], frames, frames[:-1:5], reference, system, params)


def test_system_index_finds_links_ligands_and_masked_atoms():
    system, atoms, record_type = build_system_index(_batch())

    assert record_type == "dna"
    assert len(atoms) == NUM_NT * len(NA_ATOMS) - 1 + 2 * LIGAND_CHAIN_ATOMS
    assert system.num_nucleotides == NUM_NT
    assert len(system.links) == NUM_NT - 1
    assert len(system.backbone) == NUM_NT * len(mm.BACKBONE_ATOM_NAMES) - 1
    # the two bonded ligand chains are one ligand
    assert [len(ligand) for ligand in system.ligands] == [2 * LIGAND_CHAIN_ATOMS]
    # purines (they have N9) order the ring atoms (C2, C6, C4), here consecutive in NA_ATOMS
    offsets = system.base_atoms - system.base_atoms[:, :1]
    assert offsets.tolist() == [[0, 2, 1]] * NUM_NT
    assert system.radii[0].item() == pytest.approx(const.vdw_radii[ELEMENTS["O"] - 1])  # OP1, P masked


def test_every_metric_has_a_value_on_a_complete_record():
    """No metric may be silently NaN on a record that has a ligand, links and enough pairs."""
    batch = _batch()
    values = _scored(batch, _samples(batch))

    assert tuple(values) == mm.METRIC_NAMES
    undefined = [name for name, value in values.items() if math.isnan(value)]
    assert undefined == []


def test_ensemble_metrics_are_logged_as_medians_per_record_type():
    """The RNA record gets samples that span a narrower range, so it covers its frames worse."""
    validator, model = MDValidator(["md_val"]), _Model()
    for mol_type, key, amplitude in ((DNA, 1, 0.8), (RNA, 2, 0.25)):
        batch = _batch(mol_type, key)
        validator._process_ensemble(batch, _samples(batch, bend_amplitude=amplitude))
    validator.on_epoch_end(model)

    for name in mm.METRIC_NAMES:
        assert f"val/{name}_dna" in model.logged
        assert f"val/{name}_rna" in model.logged
        assert f"val/{name}_hybrid" not in model.logged
        assert model.logged[f"val/{name}_n"] == 2
    assert 0 < model.logged["val/cov_rna"] < model.logged["val/cov_dna"] <= 1
    # the median of two records is the mean of them, not the lower one
    pair = torch.stack([model.logged["val/cov_dna"], model.logged["val/cov_rna"]])
    assert torch.isclose(model.logged["val/cov"], pair.mean())
    assert model.logged["val/cov"] > pair.min()


def test_a_single_record_is_logged_like_any_other():
    """A smoke run scores one record, for which the collected values are a scalar."""
    validator, model = MDValidator(["md_val"]), _Model()
    batch = _batch()
    validator._process_ensemble(batch, _samples(batch))
    validator.on_epoch_end(model)

    assert model.logged["val/cov_n"] == 1
    assert model.logged["val/cov"] == model.logged["val/cov_dna"]
    assert "val/cov_rna" not in model.logged


def test_metrics_undefined_for_a_record_are_left_out_of_the_median():
    """A record without a ligand must not drag the ligand metrics to NaN for the others."""
    validator, model = MDValidator(["md_val"]), _Model()
    for key, num_ligand_chains in ((1, 2), (2, 0)):
        batch = _batch(key=key, num_ligand_chains=num_ligand_chains)
        validator._process_ensemble(batch, _samples(batch))
    validator.on_epoch_end(model)

    assert model.logged["val/cov_n"] == 2
    assert model.logged["val/lig_cov_mean_n"] == 1
    assert not torch.isnan(model.logged["val/lig_cov_mean"])
    assert not torch.isnan(model.logged["val/cov"])


def test_a_stretched_link_and_a_buried_ligand_fail_the_gates():
    batch = _batch()
    system, atoms, _ = build_system_index(batch)
    samples = _samples(batch)
    assert _scored(batch, samples)["chain_break"] == 0
    assert _scored(batch, samples)["clash_free_lig_na"] == 1

    broken = samples.clone()
    o3, p = system.links[0]
    broken[:, atoms[p]] += torch.tensor([4.0, 0.0, 0.0])  # pull the next nucleotide's P away
    assert _scored(batch, broken)["chain_break"] == 1

    buried = samples.clone()
    buried[:, atoms[system.ligands[0]]] = samples[:, atoms[system.na_atoms[0]], None].squeeze(2)
    assert _scored(batch, buried)["clash_free_lig_na"] == 0


def test_reference_is_computed_once_per_record():
    validator = MDValidator(["md_val"])
    batch = _batch()
    validator._process_ensemble(batch, _samples(batch))
    reference = validator._references[int(batch["record_key"])][1]
    validator._process_ensemble(batch, _samples(batch))
    assert len(validator._references) == 1
    assert validator._references[int(batch["record_key"])][1] is reference


@pytest.mark.parametrize("change", ["atoms", "coords"])
def test_reference_is_rebuilt_for_a_different_record_under_the_same_key(change):
    """Neither a different atom selection nor different coordinates may reuse the reference."""
    validator = MDValidator(["md_val"])
    batch = _batch()
    validator._process_ensemble(batch, _samples(batch))
    reference = validator._references[int(batch["record_key"])][1]

    other = {name: value.clone() for name, value in batch.items()}
    if change == "atoms":
        other["atom_resolved_mask"][0, -NUM_PAD - 1] = 0  # drop one ligand atom
    else:
        other["coords"][0] = _batch(key=7)["coords"][0]  # same selection, other structure
    validator._process_ensemble(other, _samples(other))
    assert len(validator._references) == 1
    assert validator._references[int(batch["record_key"])][1] is not reference


def test_the_reference_subset_adapts_to_the_trajectory_length():
    """A record with fewer or more frames still gets about as many references as samples."""
    assert _reference_stride(101) == 5  # the length this metric set was calibrated on
    for num_frames in (21, 51, 101, 200):
        references = len(range(0, num_frames - 1, _reference_stride(num_frames)))
        assert NUM_REFERENCE_FRAMES <= references <= 1.5 * NUM_REFERENCE_FRAMES


def test_frames_out_of_order_are_rejected():
    batch = _batch(ordered=False)
    with pytest.raises(ValueError, match="trajectory order"):
        MDValidator(["md_val"])._process_ensemble(batch, _samples(batch))
