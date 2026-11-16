"""Tests for boltz.model.validation.md_metrics with known answers."""

import math

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.stats import rankdata, spearmanr, wasserstein_distance

from boltz.model.loss.confidencev2 import lddt_dist
from boltz.model.validation import md_metrics as mm

torch.manual_seed(0)
rng = np.random.default_rng(0)


def random_rigid(coords):
    rot = torch.tensor(Rotation.random(random_state=1).as_matrix(), dtype=torch.float32)
    return coords @ rot.T + torch.tensor([3.0, -2.0, 5.0])


def test_kabsch():
    x = torch.randn(1, 50, 3) * 5
    y = random_rigid(x[0])
    out = mm.kabsch_align(x, y, torch.arange(50))
    assert torch.allclose(out[0], y, atol=1e-4)
    # fit on a subset still recovers the rigid motion for all atoms
    out = mm.kabsch_align(x, y, torch.arange(10))
    assert torch.allclose(out[0], y, atol=1e-4)
    # reflection is not allowed
    mirrored = x[0] * torch.tensor([-1.0, 1.0, 1.0])
    out = mm.kabsch_align(x, mirrored, torch.arange(50))
    assert not torch.allclose(out[0], mirrored, atol=1e-2)


def test_lddt_matches_boltz():
    n = 60
    true = torch.randn(n, 3) * 6
    for noise in (0.3, 1.0, 3.0):
        pred = true + torch.randn(n, 3) * noise
        mask = 1 - torch.eye(n)
        expected, _ = lddt_dist(torch.cdist(pred, pred)[None], torch.cdist(true, true)[None], mask[None])
        i, j = torch.triu_indices(n, n, offset=1)
        pairs = torch.stack([i, j])
        got = mm.lddt_matrix(mm.pair_distances(pred[None], pairs), mm.pair_distances(true[None], pairs))
        assert abs(got.item() - expected.item()) < 1e-5, (got, expected)


def test_lddt_pairs():
    frames = torch.tensor([[[0.0, 0, 0], [10, 0, 0], [30, 0, 0]], [[0.0, 0, 0], [20, 0, 0], [24, 0, 0]]])
    pairs = mm.lddt_pairs(frames, 15.0)
    assert sorted(map(tuple, pairs.T.tolist())) == [(0, 1), (1, 2)]


def test_wasserstein():
    for n, m in ((20, 101), (20, 81), (5, 7)):
        a = rng.normal(size=(30, n))
        b = rng.normal(loc=0.5, scale=2, size=(30, m))
        a[0, :3] = a[0, 0]  # ties
        got = mm.wasserstein_1d(torch.tensor(a), torch.tensor(b))
        expected = [wasserstein_distance(a[k], b[k]) for k in range(30)]
        assert np.allclose(got.numpy(), expected, atol=1e-10)


def _correlated_pair(ties: int = 5):
    """Two correlated vectors, the second with a few tied values."""
    x = torch.tensor(rng.normal(size=40))
    y = x + torch.tensor(rng.normal(size=40)) * 0.5
    y[:ties] = y[0]
    return x, y


def test_spearman_matches_scipy_with_ties():
    x, y = _correlated_pair()
    assert abs(mm.spearman(x, y) - spearmanr(x, y)[0]) < 1e-10


def test_spearman_is_nan_for_a_constant_vector():
    x, _ = _correlated_pair()
    assert math.isnan(mm.spearman(x, torch.ones(40)))


def test_partial_spearman_matches_a_rank_regression():
    x, y = _correlated_pair()
    z = torch.tensor(rng.normal(size=40))
    rank_x, rank_y, rank_z = rankdata(x), rankdata(y), rankdata(z)
    design = np.stack([np.ones(40), rank_z], 1)

    def residual(values):
        return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]

    expected = np.corrcoef(residual(rank_x), residual(rank_y))[0, 1]
    assert abs(mm.partial_spearman(x, y, z) - expected) < 1e-10


def test_partial_spearman_is_nan_when_the_control_explains_everything():
    """The rank residuals are zero, so there is no correlation left to report."""
    z = torch.tensor(rng.normal(size=40))
    assert math.isnan(mm.partial_spearman(z * 2 + 1, z, z))


def test_rmsf_uses_the_unbiased_variance():
    base = torch.zeros(4, 3, 3)
    base[:, 1, 0] = torch.tensor([1.0, -1.0, 1.0, -1.0])  # one atom moves, two stay put
    assert torch.allclose(mm.rmsf(base), torch.tensor([0.0, (4 / 3) ** 0.5, 0.0]))


def test_median_averages_the_two_middle_values():
    assert abs(mm.median(torch.tensor([1.0, 2.0, 3.0, 10.0])).item() - 2.5) < 1e-12


def test_segment_mean():
    seg = mm.segment_mean(torch.tensor([1.0, 3.0, 5.0]), torch.tensor([0, 0, 1]), 2)
    assert torch.allclose(seg, torch.tensor([2.0, 5.0]))


def test_pose_rmsd():
    pocket = torch.randn(30, 3) * 4
    ligand = torch.randn(8, 3) + 6
    ref = torch.cat([pocket, ligand])[None]
    shifted = torch.cat([pocket, ligand + torch.tensor([1.5, 0, 0])])
    pred = random_rigid(shifted)[None]
    got = mm.pose_rmsd_matrix(pred, ref, torch.arange(30), torch.arange(30, 38))
    assert abs(got.item() - 1.5) < 1e-3


def test_min_dist_to_groups():
    src = torch.randn(3, 5, 3)
    dst = torch.randn(3, 9, 3)
    groups = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2])
    got = mm.min_dist_to_groups(src, dst, groups, 3)
    d = torch.cdist(src, dst)
    expected = torch.stack([d[:, :, groups == g].amin((1, 2)) for g in range(3)], 1)
    assert torch.allclose(got, expected)


def test_contact_mae():
    """One nucleotide in contact in every structure, one far away in every structure."""
    d_ref = torch.tensor([[3.0, 9.0], [3.2, 9.0], [3.1, 9.0], [3.0, 9.0]])
    assert mm.contact_mae(d_ref, d_ref, 4.0, 0.25, 0.1) == 0.0
    d_far = torch.full((4, 2), 9.0)
    assert mm.contact_mae(d_far, d_ref, 4.0, 0.25, 0.1) > 0.9
    assert math.isnan(mm.contact_mae(d_far, d_far, 4.0, 0.25, 0.1))


def test_clash_ratio():
    a = torch.tensor([[[0.0, 0, 0]]])
    b = torch.tensor([[[2.0, 0, 0]]])
    ratio = mm.clash_ratio(a, b, torch.tensor([1.7]), torch.tensor([1.52]))
    assert abs(ratio.item() - 2.0 / 3.22) < 1e-6


def test_top_fraction():
    mask = mm._top_fraction(torch.tensor([0.1, 0.5, 0.3, 0.9]), 0.25)
    assert mask.tolist() == [False, False, False, True]


def _chain_system(num_nt=4):
    """Nucleotides with atoms P, C4', O3' (indices 3k, 3k+1, 3k+2), linked O3'(k)-P(k+1)."""
    n = 3 * num_nt
    return mm.SystemIndex(
        na_atoms=torch.arange(n),
        na_residue=torch.arange(num_nt).repeat_interleave(3),
        rep_atoms=torch.arange(num_nt) * 3 + 1,
        c4_atoms=torch.arange(num_nt) * 3 + 1,
        base_atoms=torch.full((num_nt, 3), -1),
        backbone=torch.arange(n),
        links=torch.stack([torch.arange(num_nt - 1) * 3 + 2, torch.arange(1, num_nt) * 3], 1),
        link_neighbours=torch.zeros(0, 2, dtype=torch.long),
        radii=torch.full((n,), 1.7),
        ligands=(),
    )


def test_c4_distances():
    system = _chain_system()
    coords = torch.zeros(1, 12, 3)
    coords[0, system.c4_atoms, 0] = torch.tensor([0.0, 6.0, 12.0, 18.0])
    assert torch.allclose(mm.c4_distances(coords, system), torch.full((1, 3), 6.0))


def test_radius_of_gyration():
    coords = torch.tensor([[[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0]]])
    assert abs(mm.radius_of_gyration(coords).item() - 1.0) < 1e-6
    assert abs(mm.radius_of_gyration(coords * 1.05 + 3).item() - 1.05) < 1e-6

