"""Metrics comparing a set of structures with an MD ensemble.

The functions work on plain coordinate tensors of shape (n, N, 3) and atom index tensors. All
atom indices refer to the atom axis of the coordinates passed in.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

# Atoms the structures are superposed on before their fluctuation is measured; MDAnalysis
# "nucleicbackbone", the set the stored RMSF targets were fitted with
BACKBONE_ATOM_NAMES = ("P", "O5'", "C5'", "C3'", "O3'")

# Atom pairs at most two bonds apart across an O3'(i)-P(i+1) link (atom of i, atom of i+1).
# Their distance follows from the bond geometry, so they are never counted as clashing.
LINK_EXCLUDED_PAIRS = (("O3'", "P"), ("O3'", "OP1"), ("O3'", "OP2"), ("O3'", "O5'"), ("C3'", "P"))

# Ring atoms that define the local frame of a base, in the order that makes the frames of
# purines and pyrimidines point to the same face
BASE_RING_ATOMS = {True: ("C2", "C6", "C4"), False: ("C2", "C4", "C6")}

# Part of the lDDT definition, not a choice: inclusion radius and the four error thresholds
LDDT_CUTOFF = 15.0
LDDT_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)

# Pairs within LDDT_CUTOFF + this margin in any MD frame are scored, so that the pairs
# included for one frame are exactly those closer than LDDT_CUTOFF in it
LDDT_PAIR_MARGIN = 2.0

# Part of the eRMSD definition (Bottaro et al.): the ellipsoidal cutoff in scaled units, and
# the scaling per Angstrom, which weights displacement out of the base plane more strongly
ERMSD_CUTOFF = 2.4
ERMSD_SCALE = (1 / 5.0, 1 / 5.0, 1 / 3.0)

# Fluctuation below this (Angstrom) is no fluctuation: the structures of the set are copies of
# one another, so the ratio of their fluctuation to the reference's has no value
STATIC_TOLERANCE = 1e-3

# The same for diversity, which lDDT measures in steps of 0.25 / number of pairs, so anything
# below this is exactly zero. lDDT only reacts to pair distances that differ by at least 0.5 A,
# so a set whose structures are merely very similar already counts as having no diversity.
MIN_LDDT_DIVERSITY = 1e-6

# The partial correlation of the pair-distance metrics needs four pairs, one more than its two
# ranked variables and its control; below that none of the pair metrics is reported
MIN_PAIRS = 4

METRIC_NAMES = (
    "lddt_nn",
    "cov",
    "div_ratio",
    "lig_rmsd_f0",
    "lig_prec_mean",
    "lig_cov_mean",
    "rmsf_spearman_na",
    "rmsf_spearman_na_null",
    "rmsf_logratio_na",
    "rmsf_logratio_lig",
    "lig_contact_mae",
    "clash_free_na",
    "clash_free_lig_na",
    "chain_break",
    "chain_break_severe",
    "w1_na_mobile",
    "w1_lig_na_mobile",
    "dfluct_logratio_na",
    "dfluct_pspearman_na",
    "ermsd_cov",
    "c4_dist_w1",
    "rg_reldiff",
    "rg_w1",
)


@dataclass(frozen=True)
class SystemIndex:
    """Atom indices of one nucleic acid + ligand system.

    Attributes
    ----------
    na_atoms : Tensor
        (A,) indices of the nucleic-acid heavy atoms.
    na_residue : Tensor
        (A,) nucleotide index (0..R-1) of each nucleic-acid atom.
    rep_atoms : Tensor
        (R,) representative atom of each nucleotide.
    c4_atoms : Tensor
        (R,) C4' atom of each nucleotide, -1 if missing.
    base_atoms : Tensor
        (R, 3) the base ring atoms that define the local frame, ordered (C2, C6, C4) for
        purines and (C2, C4, C6) for pyrimidines; -1 where an atom is missing.
    backbone : Tensor
        (B,) backbone atoms used for superposition.
    links : Tensor
        (K, 2) O3'(i), P(i+1) atom pairs, including links between chains.
    link_neighbours : Tensor
        (E, 2) atom pairs at most two bonds apart across a link, excluded from clashes.
    radii : Tensor
        (N,) van der Waals radius of every atom.
    ligands : tuple[Tensor, ...]
        Heavy-atom indices of each ligand.

    """

    na_atoms: Tensor
    na_residue: Tensor
    rep_atoms: Tensor
    c4_atoms: Tensor
    base_atoms: Tensor
    backbone: Tensor
    links: Tensor
    link_neighbours: Tensor
    radii: Tensor
    ligands: tuple[Tensor, ...]

    @property
    def num_nucleotides(self) -> int:
        """Number of nucleotides."""
        return len(self.rep_atoms)


@dataclass(frozen=True)
class MetricParams:
    """The chosen thresholds of the metrics.

    Values that follow from a definition (lDDT, eRMSD) or are implementation details are
    module constants instead.
    """

    pocket_cutoff: float = 4.5
    contact_cutoff: float = 4.0
    contact_width: float = 0.25
    contact_set_min: float = 0.1
    clash_factor: float = 0.75
    chain_break_cutoff: float = 2.0
    chain_break_severe_cutoff: float = 2.5
    pair_cap: Optional[float] = 20.0
    mobile_fraction: float = 0.25


@dataclass(frozen=True)
class PairSet:
    """Atom or nucleotide pairs for distance-distribution metrics.

    Attributes
    ----------
    pairs : Tensor
        (2, P) pairs.
    mobile : Tensor
        (P,) mask of the pairs with the largest MD distance std.

    """

    pairs: Tensor
    mobile: Tensor


@dataclass(frozen=True)
class Reference:
    """MD-derived quantities that are fixed per system.

    Attributes
    ----------
    frame0 : Tensor
        (N, 3) first MD frame, the superposition target.
    lddt_pairs : Tensor
        (2, P) atom pairs scored by lDDT.
    pockets : tuple[Tensor, ...]
        Fit atoms of each ligand's pocket.
    na_pairs : PairSet
        Nucleotide pairs (indices into ``rep_atoms``).
    lig_pairs : tuple[PairSet, ...]
        Per ligand, (ligand atom, nucleotide) pairs.
    clash_pairs_na : Tensor
        (A, A) nucleic-acid atom pairs counted as clashes when too close: different
        nucleotides, not across a link, and never that close in the MD frames.
    clash_pairs_lig : tuple[Tensor, ...]
        (L, A) per ligand, the ligand-nucleic acid atom pairs counted as clashes.

    """

    frame0: Tensor
    lddt_pairs: Tensor
    pockets: tuple[Tensor, ...]
    na_pairs: PairSet
    lig_pairs: tuple[PairSet, ...]
    clash_pairs_na: Tensor
    clash_pairs_lig: tuple[Tensor, ...]


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def kabsch_align(mobile: Tensor, target: Tensor, fit_idx: Tensor) -> Tensor:
    """Superpose each structure onto a target using a subset of atoms.

    Parameters
    ----------
    mobile : Tensor
        (n, N, 3) structures to move.
    target : Tensor
        (N, 3) or (n, N, 3) targets.
    fit_idx : Tensor
        Atoms used for the fit.

    Returns
    -------
    Tensor
        (n, N, 3) superposed structures.

    """
    mob = mobile.double()
    tgt = target.double().expand_as(mob)
    p, q = mob[:, fit_idx], tgt[:, fit_idx]
    p_mean, q_mean = p.mean(1, keepdim=True), q.mean(1, keepdim=True)
    h = (p - p_mean).transpose(1, 2) @ (q - q_mean)
    u, _, vt = torch.linalg.svd(h)
    v, ut = vt.transpose(1, 2), u.transpose(1, 2)
    sign = torch.ones(len(mob), 3, dtype=mob.dtype)
    sign[:, 2] = torch.sign(torch.linalg.det(v @ ut))
    rot = v @ torch.diag_embed(sign) @ ut
    return ((mob - p_mean) @ rot.transpose(1, 2) + q_mean).to(mobile.dtype)


def pair_distances(coords: Tensor, pairs: Tensor) -> Tensor:
    """Get the (n, P) distances of atom pairs given as a (2, P) tensor."""
    return (coords[:, pairs[0]] - coords[:, pairs[1]]).norm(dim=-1)


def base_ring_atoms(atoms_by_name: dict[str, int]) -> list[int]:
    """Get the ring atoms that define a base frame, -1 where one is missing.

    Purines are told apart by their N9; see ``BASE_RING_ATOMS`` for the order.
    """
    return [atoms_by_name.get(name, -1) for name in BASE_RING_ATOMS["N9" in atoms_by_name]]


def na_atom_position(system: SystemIndex) -> Tensor:
    """Get the (N,) position of each atom within ``na_atoms``, -1 for all other atoms."""
    position = torch.full((len(system.radii),), -1, dtype=torch.long)
    position[system.na_atoms] = torch.arange(len(system.na_atoms))
    return position


def linked_nucleotides(system: SystemIndex) -> Tensor:
    """Get the (K, 2) nucleotide pairs joined by an O3'-P link."""
    return system.na_residue[na_atom_position(system)[system.links]].reshape(-1, 2)


def clash_candidate_pairs(system: SystemIndex) -> Tensor:
    """Get the (A, A) nucleic-acid atom pairs whose closeness no bond explains.

    Pairs within one nucleotide and the pairs across an O3'-P link are left out; the upper
    triangle is kept, so each pair appears once.
    """
    residue = system.na_residue
    keep = torch.triu(residue[:, None] != residue[None], diagonal=1)
    if len(system.link_neighbours):
        position = na_atom_position(system)
        a, b = position[system.link_neighbours[:, 0]], position[system.link_neighbours[:, 1]]
        keep[torch.minimum(a, b), torch.maximum(a, b)] = False
    return keep


def lddt_pairs(frames: Tensor, cutoff: float) -> Tensor:
    """Get the atom pairs closer than a cutoff in at least one frame, as a (2, P) tensor."""
    num_atoms = frames.shape[1]
    keep = torch.zeros(num_atoms, num_atoms, dtype=torch.bool)
    for frame in frames:
        keep |= torch.cdist(frame, frame) < cutoff
    return torch.triu(keep, diagonal=1).nonzero().T


def lddt_matrix(pred: Tensor, ref: Tensor) -> Tensor:
    """Get the global lDDT of every prediction against every reference.

    Parameters
    ----------
    pred : Tensor
        (n, P) pair distances of the predictions.
    ref : Tensor
        (m, P) pair distances of the references on the same pairs.

    Returns
    -------
    Tensor
        (n, m) lDDT values.

    """
    mask = (ref < LDDT_CUTOFF).float()
    norm = mask.sum(-1).clamp(min=1)
    out = torch.empty(len(pred), len(ref))
    for i, dist in enumerate(pred):
        error = (dist[None] - ref).abs()
        score = sum((error < t).float() for t in LDDT_THRESHOLDS) / len(LDDT_THRESHOLDS)
        out[i] = (score * mask).sum(-1) / norm
    return out


def pocket_atoms(frame0: Tensor, ligand: Tensor, system: SystemIndex, cutoff: float) -> Tensor:
    """Get all atoms of the nucleotides with an atom within a cutoff of the ligand."""
    dist = torch.cdist(frame0[system.na_atoms], frame0[ligand]).min(1).values
    residues = system.na_residue[dist < cutoff].unique()
    return system.na_atoms[torch.isin(system.na_residue, residues)]


def pose_rmsd_matrix(pred: Tensor, ref: Tensor, fit_idx: Tensor, ligand: Tensor) -> Tensor:
    """Get the ligand RMSD of every prediction against every reference after a pocket fit.

    Returns
    -------
    Tensor
        (n, m) RMSD values.

    """
    n, m = len(pred), len(ref)
    atoms = torch.cat([fit_idx, ligand])
    num_fit = len(fit_idx)
    mobile = pred[:, atoms][:, None].expand(n, m, -1, -1).reshape(n * m, len(atoms), 3)
    target = ref[:, atoms][None].expand(n, m, -1, -1).reshape(n * m, len(atoms), 3)
    aligned = kabsch_align(mobile, target, torch.arange(num_fit))
    sq = (aligned[:, num_fit:] - target[:, num_fit:]).pow(2).sum(-1).mean(-1)
    return sq.sqrt().reshape(n, m)


def rmsf(coords: Tensor) -> Tensor:
    """Get the (N,) root mean square fluctuation of already superposed structures.

    Uses the unbiased variance (n - 1), like the distance standard deviations, so sets of
    different size can be compared.
    """
    sq = (coords - coords.mean(0)).pow(2).sum(-1)
    return (sq.sum(0) / (len(coords) - 1)).sqrt()


def median(x: Tensor) -> Tensor:
    """Get the median, averaging the two middle values for an even length."""
    return torch.quantile(x.double(), 0.5)


def segment_mean(values: Tensor, segments: Tensor, num_segments: int) -> Tensor:
    """Average values (first axis) over segment ids."""
    total = torch.zeros(num_segments, *values.shape[1:], dtype=values.dtype)
    total.index_add_(0, segments, values)
    count = torch.zeros(num_segments, dtype=values.dtype)
    count.index_add_(0, segments, torch.ones(len(values), dtype=values.dtype))
    return total / count.reshape(-1, *([1] * (values.dim() - 1)))


def _rank(x: Tensor) -> Tensor:
    """Rank values, giving ties their average rank."""
    order = torch.argsort(x)
    _, inverse, counts = torch.unique_consecutive(
        x[order], return_inverse=True, return_counts=True
    )
    ends = counts.cumsum(0).double()
    mean_rank = ends - (counts.double() + 1) / 2
    ranks = torch.empty(len(x), dtype=torch.float64)
    ranks[order] = mean_rank[inverse]
    return ranks


def _pearson(x: Tensor, y: Tensor) -> float:
    x, y = x - x.mean(), y - y.mean()
    tol = 1e-8 * len(x)
    if x.norm() < tol or y.norm() < tol:
        return float("nan")
    return (x @ y).item() / (x.norm() * y.norm()).item()


def spearman(x: Tensor, y: Tensor) -> float:
    """Get the Spearman correlation of two vectors (NaN if one is constant)."""
    if len(x) < 3:
        return float("nan")
    return _pearson(_rank(x.double()), _rank(y.double()))


def partial_spearman(x: Tensor, y: Tensor, z: Tensor) -> float:
    """Get the Spearman correlation of x and y, with the linear effect of z on the ranks removed."""
    if len(x) < 4:
        return float("nan")
    rz = _rank(z.double())
    design = torch.stack([torch.ones_like(rz), rz], 1)

    def residual(v: Tensor) -> Tensor:
        rv = _rank(v.double())
        coef = torch.linalg.lstsq(design, rv[:, None]).solution
        return rv - (design @ coef)[:, 0]

    return _pearson(residual(x), residual(y))


def wasserstein_1d(a: Tensor, b: Tensor) -> Tensor:
    """Get the exact 1D Wasserstein-1 distance between empirical distributions, row by row.

    Parameters
    ----------
    a : Tensor
        (P, n) samples of the first distributions.
    b : Tensor
        (P, m) samples of the second distributions.

    Returns
    -------
    Tensor
        (P,) distances.

    """
    n, m = a.shape[1], b.shape[1]
    a_sorted, b_sorted = a.double().sort(1).values, b.double().sort(1).values
    # Both quantile functions are constant between consecutive breakpoints k/n and k/m
    steps = torch.cat([torch.arange(1, n + 1) / n, torch.arange(1, m + 1) / m]).double()
    # Rounded so that breakpoints the two grids share merge into one interval
    bounds = torch.unique(
        torch.cat([torch.zeros(1, dtype=torch.float64), steps]).round(decimals=12)
    )
    widths = bounds[1:] - bounds[:-1]
    mid = (bounds[1:] + bounds[:-1]) / 2
    idx_a = (mid * n).floor().long().clamp(max=n - 1)
    idx_b = (mid * m).floor().long().clamp(max=m - 1)
    return ((a_sorted[:, idx_a] - b_sorted[:, idx_b]).abs() * widths).sum(1)


def min_dist_to_groups(src: Tensor, dst: Tensor, groups: Tensor, num_groups: int) -> Tensor:
    """Get the minimum distance between any source atom and the atoms of each group.

    Parameters
    ----------
    src : Tensor
        (n, L, 3) source atoms (e.g. a ligand).
    dst : Tensor
        (n, A, 3) target atoms.
    groups : Tensor
        (A,) group id of each target atom.
    num_groups : int
        Number of groups.

    Returns
    -------
    Tensor
        (n, num_groups) minimum distances.

    """
    per_atom = torch.cdist(src, dst).min(1).values
    out = torch.full((len(src), num_groups), float("inf"), dtype=per_atom.dtype)
    return out.scatter_reduce(1, groups[None].expand_as(per_atom), per_atom, reduce="amin")


def ligand_contact_distances(coords: Tensor, ligand: Tensor, system: SystemIndex) -> Tensor:
    """Get the (n, R) minimum heavy-atom distance between a ligand and each nucleotide."""
    return min_dist_to_groups(
        coords[:, ligand], coords[:, system.na_atoms], system.na_residue, system.num_nucleotides
    )


def contact_mae(
    d_pred: Tensor, d_ref: Tensor, cutoff: float, width: float, set_min: float
) -> float:
    """Get the mean absolute error of soft contact occupancies over the contacted groups."""
    occ_pred = torch.sigmoid((cutoff - d_pred) / width).mean(0)
    occ_ref = torch.sigmoid((cutoff - d_ref) / width).mean(0)
    contacted = (occ_pred > set_min) | (occ_ref > set_min)
    if not contacted.any():
        return float("nan")
    return (occ_pred - occ_ref)[contacted].abs().mean().item()


def clash_ratio(coords_a: Tensor, coords_b: Tensor, radii_a: Tensor, radii_b: Tensor) -> Tensor:
    """Get distance / (r_i + r_j) for all atom pairs of two sets, shape (n, A, B)."""
    return torch.cdist(coords_a, coords_b) / (radii_a[:, None] + radii_b[None])


def min_clash_ratio(frames: Tensor, idx_a: Tensor, idx_b: Tensor, radii: Tensor) -> Tensor:
    """Get the minimum over frames of distance / (r_i + r_j), shape (A, B)."""
    out = None
    for frame in frames:
        ratio = clash_ratio(frame[None, idx_a], frame[None, idx_b], radii[idx_a], radii[idx_b])[0]
        out = ratio if out is None else torch.minimum(out, ratio)
    return out


def clash_counts(
    pred: Tensor, reference: Reference, system: SystemIndex, factor: float
) -> tuple[Tensor, Tensor]:
    """Count the clashes of every structure.

    The factor must be the one the reference excluded its bonded pairs with, otherwise the
    exclusions and the threshold disagree.

    Returns
    -------
    tuple[Tensor, Tensor]
        (n,) nucleic-acid clashes and (n,) ligand-nucleic acid clashes summed over ligands.

    """
    na, radii = system.na_atoms, system.radii
    na_counts = torch.zeros(len(pred))
    lig_counts = torch.zeros(len(pred))
    for s, coords in enumerate(pred):
        ratio = clash_ratio(coords[None, na], coords[None, na], radii[na], radii[na])[0]
        na_counts[s] = ((ratio < factor) & reference.clash_pairs_na).sum()
        for ligand, keep in zip(system.ligands, reference.clash_pairs_lig):
            ratio = clash_ratio(coords[None, ligand], coords[None, na], radii[ligand], radii[na])[0]
            lig_counts[s] += ((ratio < factor) & keep).sum()
    return na_counts, lig_counts


def base_frames(coords: Tensor, base_atoms: Tensor) -> tuple[Tensor, Tensor]:
    """Get the local coordinate system of each base from its ring atoms.

    Parameters
    ----------
    coords : Tensor
        (n, N, 3) structures.
    base_atoms : Tensor
        (R, 3) ring atoms per base, ordered so that the frames of purines and pyrimidines
        point to the same face of the base.

    Returns
    -------
    tuple[Tensor, Tensor]
        (n, R, 3) base centres and (n, R, 3, 3) orthonormal axes as rows.

    """
    a0, a1 = coords[:, base_atoms[:, 0]], coords[:, base_atoms[:, 1]]
    origin = coords[:, base_atoms].mean(2)
    x = torch.nn.functional.normalize(a0 - origin, dim=-1)
    z = torch.nn.functional.normalize(torch.linalg.cross(x, a1 - origin), dim=-1)
    return origin, torch.stack([x, torch.linalg.cross(z, x), z], dim=-2)


def g_vectors(coords: Tensor, base_atoms: Tensor) -> Tensor:
    """Get the G-vectors of every ordered pair of bases (Bottaro et al., eRMSD).

    The position of each base centre in the local frame of another base is scaled
    anisotropically; pairs outside the resulting ellipsoid get a zero vector.

    Returns
    -------
    Tensor
        (n, R, R, 4) G-vectors.

    """
    origin, axes = base_frames(coords, base_atoms)
    diff = origin[:, None, :, :] - origin[:, :, None, :]  # [:, i, j] = centre j minus centre i
    scale = torch.tensor(ERMSD_SCALE, dtype=coords.dtype)
    # [n, i, j, d] = the i -> j vector projected on axis d of base i's frame
    scaled = torch.einsum("nijc,nicd->nijd", diff, axes.transpose(-2, -1)) * scale
    norm = scaled.norm(dim=-1, keepdim=True).clamp(min=1e-9)

    angle = torch.pi * norm / ERMSD_CUTOFF
    tail = (1 + torch.cos(angle)) * ERMSD_CUTOFF / torch.pi
    g = torch.cat([scaled * torch.sin(angle) / angle, tail], dim=-1)
    inside = (norm < ERMSD_CUTOFF) & ~torch.eye(len(base_atoms), dtype=torch.bool)[None, :, :, None]
    return g * inside


def ermsd_set_matrix(pred: Tensor, ref: Tensor, system: SystemIndex) -> Optional[Tensor]:
    """Get the (n, m) eRMSD of every prediction against every reference, None if too few bases."""
    bases = system.base_atoms[(system.base_atoms >= 0).all(1)]
    if len(bases) < 2:
        return None
    return ermsd_matrix(g_vectors(pred, bases), g_vectors(ref, bases))


def ermsd_matrix(g_pred: Tensor, g_ref: Tensor) -> Tensor:
    """Get the eRMSD of every prediction against every reference from their G-vectors.

    Parameters
    ----------
    g_pred : Tensor
        (n, R, R, 4) G-vectors of the predictions.
    g_ref : Tensor
        (m, R, R, 4) G-vectors of the references.

    Returns
    -------
    Tensor
        (n, m) eRMSD values.

    """
    num_bases = g_pred.shape[1]
    flat_pred = g_pred.reshape(len(g_pred), -1)
    flat_ref = g_ref.reshape(len(g_ref), -1)
    return torch.cdist(flat_pred, flat_ref) / num_bases**0.5


def c4_distances(coords: Tensor, system: SystemIndex) -> Tensor:
    """Get the (n, L) C4'(i)-C4'(i+1) distances of the linked nucleotides."""
    links = linked_nucleotides(system)
    c4 = system.c4_atoms
    links = links[(c4[links] >= 0).all(1)]
    return (coords[:, c4[links[:, 0]]] - coords[:, c4[links[:, 1]]]).norm(dim=-1)


def radius_of_gyration(coords: Tensor) -> Tensor:
    """Get the (n,) unweighted radius of gyration."""
    return (coords - coords.mean(1, keepdim=True)).pow(2).sum(-1).mean(1).sqrt()


def ligand_pair_distances(coords: Tensor, pairs: Tensor, system: SystemIndex) -> Tensor:
    """Get the (n, P) distances of (ligand atom, nucleotide representative atom) pairs."""
    return (coords[:, pairs[0]] - coords[:, system.rep_atoms[pairs[1]]]).norm(dim=-1)


def pair_sets(
    frames: Tensor, system: SystemIndex, cap: Optional[float], fraction: float
) -> tuple[PairSet, tuple[PairSet, ...]]:
    """Get the nucleotide and ligand-nucleotide pair sets with their mobile pairs.

    Nucleotide pairs skip linked neighbours; both sets keep the pairs with a mean MD distance
    below ``cap`` (all if None). The mobile pairs are the ``fraction`` with the largest MD std.
    """
    num_nt = system.num_nucleotides
    rep = frames[:, system.rep_atoms]
    i, j = torch.triu_indices(num_nt, num_nt, offset=1)
    linked = torch.zeros(num_nt, num_nt, dtype=torch.bool)
    nt_links = linked_nucleotides(system)
    linked[nt_links[:, 0], nt_links[:, 1]] = True
    linked |= linked.T.clone()
    na = torch.stack([i[~linked[i, j]], j[~linked[i, j]]])

    def select(pairs: Tensor, dist: Tensor) -> PairSet:
        keep = torch.ones(dist.shape[1], dtype=torch.bool) if cap is None else dist.mean(0) < cap
        return PairSet(pairs[:, keep], _top_fraction(dist[:, keep].std(0), fraction))

    na_set = select(na, pair_distances(rep, na))
    lig_sets = []
    for ligand in system.ligands:
        a, r = torch.meshgrid(ligand, torch.arange(num_nt), indexing="ij")
        pairs = torch.stack([a.flatten(), r.flatten()])
        lig_sets.append(select(pairs, ligand_pair_distances(frames, pairs, system)))
    return na_set, tuple(lig_sets)


def _top_fraction(values: Tensor, fraction: float) -> Tensor:
    """Get a mask of the largest values, at least one."""
    mask = torch.zeros(len(values), dtype=torch.bool)
    if len(values):
        mask[values.topk(max(1, round(len(values) * fraction))).indices] = True
    return mask


def build_reference(frames: Tensor, system: SystemIndex, params: MetricParams) -> Reference:
    """Compute the MD-derived quantities that are fixed per system.

    Atom pairs that clash in some MD frame are treated as bonded (e.g. covalent ligand links)
    and are never counted as clashes.

    Parameters
    ----------
    frames : Tensor
        (K, N, 3) MD frames, frame 0 first.
    system : SystemIndex
        The atom indices.
    params : MetricParams
        The thresholds.

    Returns
    -------
    Reference
        The reference quantities.

    """
    frame0 = frames[0]
    na, factor = system.na_atoms, params.clash_factor
    keep = clash_candidate_pairs(system) & (min_clash_ratio(frames, na, na, system.radii) >= factor)
    na_pairs, lig_pairs = pair_sets(frames, system, params.pair_cap, params.mobile_fraction)
    return Reference(
        frame0=frame0,
        lddt_pairs=lddt_pairs(frames, LDDT_CUTOFF + LDDT_PAIR_MARGIN),
        pockets=tuple(
            pocket_atoms(frame0, ligand, system, params.pocket_cutoff) for ligand in system.ligands
        ),
        na_pairs=na_pairs,
        lig_pairs=lig_pairs,
        clash_pairs_na=keep,
        clash_pairs_lig=tuple(
            min_clash_ratio(frames, ligand, na, system.radii) >= factor for ligand in system.ligands
        ),
    )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _offdiag_mean(x: Tensor) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    return ((x.sum() - x.diagonal().sum()) / (n * (n - 1))).item()


def _nanmean(values: list[float]) -> float:
    """Average the defined values, NaN if there are none."""
    defined = [v for v in values if not math.isnan(v)]
    return sum(defined) / len(defined) if defined else float("nan")


def _log_ratio(pred: Tensor, ref: Tensor) -> float:
    """Get log(median(pred) / median(ref)), NaN if either set does not fluctuate."""
    if len(pred) == 0 or pred.max() < STATIC_TOLERANCE or ref.max() < STATIC_TOLERANCE:
        return float("nan")
    return torch.log(median(pred) / median(ref)).item()


def score_ensemble(
    pred: Tensor,
    ref_frames: Tensor,
    ref_subset: Tensor,
    reference: Reference,
    system: SystemIndex,
    params: MetricParams,
) -> dict[str, float]:
    """Score a set of structures against an MD ensemble.

    Parameters
    ----------
    pred : Tensor
        (n, N, 3) structures to score (model samples or a control).
    ref_frames : Tensor
        (m, N, 3) MD frames used for distribution statistics.
    ref_subset : Tensor
        (k, N, 3) MD frames used for set-to-set comparisons.
    reference : Reference
        The fixed MD-derived quantities.
    system : SystemIndex
        The atom indices.
    params : MetricParams
        The thresholds.

    Returns
    -------
    dict[str, float]
        The metric values, NaN where a metric is undefined (e.g. no ligand).

    """
    out = {}
    out.update(_set_metrics(pred, ref_subset, reference))
    out.update(_pose_metrics(pred, ref_subset, reference, system))
    out.update(_rmsf_metrics(pred, ref_frames, reference, system))
    out.update(_contact_metrics(pred, ref_frames, system, params))
    out.update(_validity_metrics(pred, reference, system, params))
    out.update(pair_set_metrics(pred, ref_frames, reference.na_pairs, reference.lig_pairs, system))
    out.update(_backbone_metrics(pred, ref_frames, system))
    out.update(_ermsd_metrics(pred, ref_subset, system))
    return {name: out.get(name, float("nan")) for name in METRIC_NAMES}


def _set_metrics(pred: Tensor, ref_subset: Tensor, reference: Reference) -> dict[str, float]:
    """Precision, coverage and diversity from lDDT matrices."""
    pairs = reference.lddt_pairs
    d_pred = pair_distances(pred, pairs)
    d_ref = pair_distances(ref_subset, pairs)
    l_pr = lddt_matrix(d_pred, d_ref)
    d_pp = _offdiag_mean(1 - lddt_matrix(d_pred, d_pred))
    d_rr = _offdiag_mean(1 - lddt_matrix(d_ref, d_ref))
    return {
        "lddt_nn": l_pr.max(1).values.mean().item(),
        "cov": l_pr.max(0).values.mean().item(),
        # A set of identical structures has no diversity, so the ratio has no finite value
        "div_ratio": (
            math.log(d_pp / d_rr)
            if min(d_pp, d_rr) > MIN_LDDT_DIVERSITY
            else float("nan")
        ),
    }


def _pose_metrics(
    pred: Tensor, ref_subset: Tensor, reference: Reference, system: SystemIndex
) -> dict[str, float]:
    """Ligand pose accuracy, precision and coverage after a pocket fit, averaged over ligands."""
    per_ligand = []
    for ligand, pocket in zip(system.ligands, reference.pockets):
        if len(pocket) < 3:
            continue
        to_frame0 = pose_rmsd_matrix(pred, reference.frame0[None], pocket, ligand)[:, 0]
        matrix = pose_rmsd_matrix(pred, ref_subset, pocket, ligand)
        per_ligand.append({
            "lig_rmsd_f0": to_frame0.mean().item(),
            "lig_prec_mean": matrix.min(1).values.mean().item(),
            "lig_cov_mean": matrix.min(0).values.mean().item(),
        })
    if not per_ligand:
        return {}
    return {key: _nanmean([v[key] for v in per_ligand]) for key in per_ligand[0]}


def _rmsf_metrics(
    pred: Tensor, ref_frames: Tensor, reference: Reference, system: SystemIndex
) -> dict[str, float]:
    """Where (Spearman) and how much (log ratio) the structures fluctuate."""
    rmsf_pred = rmsf(kabsch_align(pred, reference.frame0, system.backbone))
    rmsf_ref = rmsf(kabsch_align(ref_frames, reference.frame0, system.backbone))
    num_nt = system.num_nucleotides
    nt_pred = segment_mean(rmsf_pred[system.na_atoms], system.na_residue, num_nt)
    nt_ref = segment_mean(rmsf_ref[system.na_atoms], system.na_residue, num_nt)

    # Score of a model that only knows the distance from the fit centre
    centroids = segment_mean(reference.frame0[system.na_atoms].double(), system.na_residue, num_nt)
    centre_dist = (centroids - reference.frame0[system.backbone].double().mean(0)).norm(dim=-1)

    static = nt_pred.max() < STATIC_TOLERANCE  # identical structures: the ranks are numerical noise
    return {
        "rmsf_spearman_na": float("nan") if static else spearman(nt_pred, nt_ref),
        "rmsf_spearman_na_null": spearman(centre_dist, nt_ref),
        "rmsf_logratio_na": _log_ratio(nt_pred, nt_ref),
        "rmsf_logratio_lig": _nanmean([
            _log_ratio(rmsf_pred[ligand], rmsf_ref[ligand]) for ligand in system.ligands
        ]),
    }


def _contact_metrics(
    pred: Tensor, ref_frames: Tensor, system: SystemIndex, params: MetricParams
) -> dict[str, float]:
    """Ligand-nucleotide contact occupancy, averaged over ligands."""
    if not system.ligands:
        return {}
    mae = [
        contact_mae(
            ligand_contact_distances(pred, ligand, system),
            ligand_contact_distances(ref_frames, ligand, system),
            params.contact_cutoff,
            params.contact_width,
            params.contact_set_min,
        )
        for ligand in system.ligands
    ]
    return {"lig_contact_mae": _nanmean(mae)}


def _validity_metrics(
    pred: Tensor, reference: Reference, system: SystemIndex, params: MetricParams
) -> dict[str, float]:
    """Clash-free and chain-break fractions over the structures."""
    na_counts, lig_counts = clash_counts(pred, reference, system, params.clash_factor)
    out = {"clash_free_na": (na_counts == 0).float().mean().item()}
    if system.ligands:
        out["clash_free_lig_na"] = (lig_counts == 0).float().mean().item()
    if len(system.links):
        link = pair_distances(pred, system.links.T)
        out["chain_break"] = (link > params.chain_break_cutoff).any(1).float().mean().item()
        severe = link > params.chain_break_severe_cutoff
        out["chain_break_severe"] = severe.any(1).float().mean().item()
    return out


def pair_set_metrics(
    pred: Tensor,
    ref_frames: Tensor,
    na_pairs: PairSet,
    lig_pairs: tuple[PairSet, ...],
    system: SystemIndex,
) -> dict[str, float]:
    """Distance-distribution (W1) and distance-fluctuation metrics on given pair sets."""
    out = {}
    if na_pairs.pairs.shape[1] >= MIN_PAIRS:
        nt_pairs = system.rep_atoms[na_pairs.pairs]
        d_pred, d_ref = pair_distances(pred, nt_pairs), pair_distances(ref_frames, nt_pairs)
        mobile = na_pairs.mobile
        std_pred, std_ref = d_pred.std(0), d_ref.std(0)
        out["w1_na_mobile"] = wasserstein_1d(d_pred[:, mobile].T, d_ref[:, mobile].T).mean().item()
        out["dfluct_logratio_na"] = _log_ratio(std_pred[mobile], std_ref[mobile])
        out["dfluct_pspearman_na"] = partial_spearman(std_pred, std_ref, d_ref.mean(0))

    w1 = []
    for lig_set in lig_pairs:
        if lig_set.pairs.shape[1] < MIN_PAIRS:
            continue
        d_pred = ligand_pair_distances(pred, lig_set.pairs, system)[:, lig_set.mobile]
        d_ref = ligand_pair_distances(ref_frames, lig_set.pairs, system)[:, lig_set.mobile]
        w1.append(wasserstein_1d(d_pred.T, d_ref.T).mean().item())
    if w1:
        out["w1_lig_na_mobile"] = _nanmean(w1)
    return out


def _ermsd_metrics(pred: Tensor, ref_subset: Tensor, system: SystemIndex) -> dict[str, float]:
    """Base-geometry coverage: how well the samples cover the reference base geometries."""
    matrix = ermsd_set_matrix(pred, ref_subset, system)
    if matrix is None:
        return {}
    return {"ermsd_cov": matrix.min(0).values.mean().item()}


def _backbone_metrics(pred: Tensor, ref_frames: Tensor, system: SystemIndex) -> dict[str, float]:
    """Local C4' backbone geometry and nucleic-acid size, compared as distributions."""
    out = {}
    dist_pred, dist_ref = c4_distances(pred, system), c4_distances(ref_frames, system)
    if dist_pred.numel():
        out["c4_dist_w1"] = wasserstein_1d(dist_pred.reshape(1, -1), dist_ref.reshape(1, -1)).item()
    rg_pred = radius_of_gyration(pred[:, system.na_atoms])
    rg_ref = radius_of_gyration(ref_frames[:, system.na_atoms])
    out["rg_reldiff"] = ((rg_pred.mean() - rg_ref.mean()) / rg_ref.mean()).item()
    out["rg_w1"] = wasserstein_1d(rg_pred[None], rg_ref[None]).item()
    return out
