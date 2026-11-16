"""Tests for the eRMSD base-geometry metric.

The reference implementation below transcribes barnaba's ``calc_lcs``/``calc_gmat``
(Bottaro et al.) in numpy, with its nanometre lengths converted to Angstrom.
"""

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from boltz.model.validation import md_metrics as mm

NUM_BASES = 12


def reference_gmat(coords: np.ndarray, cutoff: float, scale: np.ndarray) -> np.ndarray:
    """G-vectors of one structure, following barnaba's formula. coords: (R, 3, 3) ring atoms."""
    origin = coords.sum(axis=1) / 3.0
    x = coords[:, 0] - origin
    x /= np.linalg.norm(x, axis=1)[:, None]
    z = np.cross(x, coords[:, 1] - origin)
    z /= np.linalg.norm(z, axis=1)[:, None]
    y = np.cross(z, x)

    num = len(coords)
    out = np.zeros((num, num, 4))
    for i in range(num):
        for j in range(num):
            if i == j:
                continue
            dotp = np.array([(origin[j] - origin[i]) @ axis for axis in (x[i], y[i], z[i])]) * scale
            norm = np.linalg.norm(dotp)
            if norm >= cutoff:
                continue
            ff = np.pi * norm / cutoff
            out[i, j, :3] = dotp * np.sin(ff) / ff
            out[i, j, 3] = (1.0 + np.cos(ff)) * cutoff / np.pi
    return out


def _random_bases(num_bases: int = NUM_BASES, seed: int = 0):
    """Stacked bases 4 A apart with a random in-plane offset, and their atom indices."""
    rng = np.random.default_rng(seed)
    coords = []
    for k in range(num_bases):
        centre = np.array([0.0, 0.0, 4.0 * k]) + rng.normal(scale=0.5, size=3)
        coords += [centre + np.array([1.4, 0.0, 0.0]), centre + np.array([-0.7, 1.2, 0.0]),
                   centre + np.array([-0.7, -1.2, 0.0])]
    base_atoms = torch.arange(3 * num_bases).reshape(num_bases, 3)
    return torch.tensor(np.array(coords), dtype=torch.float64)[None], base_atoms


def test_g_vectors_match_the_reference_formula():
    coords, base_atoms = _random_bases()
    got = mm.g_vectors(coords, base_atoms)[0].numpy()
    expected = reference_gmat(
        coords[0].numpy().reshape(NUM_BASES, 3, 3), mm.ERMSD_CUTOFF, np.array(mm.ERMSD_SCALE)
    )
    assert np.allclose(got, expected, atol=1e-10)
    assert (np.abs(got).sum((-1, -2)) > 0).all(), "every base should see a neighbour"


def test_ermsd_is_zero_for_identical_structures():
    coords, base_atoms = _random_bases()
    g = mm.g_vectors(coords, base_atoms)
    assert mm.ermsd_matrix(g, g).abs().max().item() < 1e-9


def test_ermsd_is_invariant_under_rigid_motion():
    coords, base_atoms = _random_bases()
    rotation = torch.tensor(Rotation.random(random_state=3).as_matrix(), dtype=coords.dtype)
    moved = coords @ rotation.T + torch.tensor([5.0, -3.0, 2.0], dtype=coords.dtype)
    other, _ = _random_bases(seed=1)
    g, g_moved = mm.g_vectors(coords, base_atoms), mm.g_vectors(moved, base_atoms)
    g_other = mm.g_vectors(other, base_atoms)
    assert torch.allclose(mm.ermsd_matrix(g, g_other), mm.ermsd_matrix(g_moved, g_other), atol=1e-9)


def test_ermsd_is_unchanged_when_both_structures_flip_the_base_frames():
    """Swapping the last two ring atoms mirrors every base frame. barnaba does this for DNA
    purines; applied to both structures the sign changes cancel in the distance."""
    coords, base_atoms = _random_bases()
    other, _ = _random_bases(seed=1)
    flipped = base_atoms[:, [0, 2, 1]]
    plain = mm.ermsd_matrix(
        mm.g_vectors(coords, base_atoms), mm.g_vectors(other, base_atoms)
    )
    mirrored = mm.ermsd_matrix(
        mm.g_vectors(coords, flipped), mm.g_vectors(other, flipped)
    )
    assert torch.allclose(plain, mirrored, atol=1e-9)


def test_ermsd_grows_with_distortion():
    coords, base_atoms = _random_bases()
    g = mm.g_vectors(coords, base_atoms)
    previous = 0.0
    for noise in (0.1, 0.5, 2.0):
        torch.manual_seed(0)
        distorted = coords + noise * torch.randn(coords.shape, dtype=coords.dtype)
        value = mm.ermsd_matrix(mm.g_vectors(distorted, base_atoms), g).item()
        assert value > previous
        previous = value


def test_bases_beyond_the_cutoff_are_ignored():
    coords, base_atoms = _random_bases(num_bases=2)
    far = coords.clone()
    far[0, 3:] += torch.tensor([0.0, 0.0, 100.0], dtype=coords.dtype)
    assert mm.g_vectors(far, base_atoms).abs().max().item() == pytest.approx(0.0)
