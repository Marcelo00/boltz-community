"""Metrics comparing a set of structures with an MD ensemble.

The functions work on plain coordinate tensors of shape (n, N, 3) and atom index tensors. All
atom indices refer to the atom axis of the coordinates passed in.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

LDDT_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)


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
    backbone : Tensor
        (B,) backbone atoms used for superposition.
    links : Tensor
        (K, 2) O3'(i), P(i+1) atom pairs, including links between chains.
    link_neighbours : Tensor
        (E, 2) atom pairs on both sides of a link, excluded from clashes.
    radii : Tensor
        (N,) van der Waals radius of every atom.
    ligands : tuple[Tensor, ...]
        Heavy-atom indices of each ligand.

    """

    na_atoms: Tensor
    na_residue: Tensor
    rep_atoms: Tensor
    c4_atoms: Tensor
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
    """Thresholds of the metrics."""

    lddt_cutoff: float = 15.0
    lddt_pair_margin: float = 2.0
    pocket_cutoff: float = 4.5
    contact_cutoff: float = 4.0
    contact_width: float = 0.25
    contact_set_min: float = 0.1
    bound_cutoff: float = 4.0
    bound_reference_min: float = 0.5
    bound_fraction: float = 0.5
    clash_factor: float = 0.75
    chain_break_cutoff: float = 2.0
    chain_break_severe_cutoff: float = 2.5
    pair_cap: Optional[float] = 20.0
    mobile_fraction: float = 0.25
    rmsf_floor: float = 1e-3


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
    clash_keep_na : Tensor
        (A, A) nucleic-acid atom pairs that can clash (different nucleotides, not across a link).
    clash_min_ratio_na : Tensor
        (A, A) minimum over the exclusion frames of distance / (r_i + r_j).
    clash_min_ratio_lig : tuple[Tensor, ...]
        (L, A) per ligand, the same for ligand and nucleic-acid atoms.

    """

    frame0: Tensor
    lddt_pairs: Tensor
    pockets: tuple[Tensor, ...]
    na_pairs: PairSet
    lig_pairs: tuple[PairSet, ...]
    clash_keep_na: Tensor
    clash_min_ratio_na: Tensor
    clash_min_ratio_lig: tuple[Tensor, ...]


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


def linked_nucleotides(system: SystemIndex) -> Tensor:
    """Get the (K, 2) nucleotide pairs joined by an O3'-P link."""
    position = torch.full((len(system.radii),), -1, dtype=torch.long)
    position[system.na_atoms] = torch.arange(len(system.na_atoms))
    return system.na_residue[position[system.links]].reshape(-1, 2)


def lddt_pairs(frames: Tensor, cutoff: float) -> Tensor:
    """Get the atom pairs closer than a cutoff in at least one frame, as a (2, P) tensor."""
    num_atoms = frames.shape[1]
    keep = torch.zeros(num_atoms, num_atoms, dtype=torch.bool)
    for frame in frames:
        keep |= torch.cdist(frame, frame) < cutoff
    return torch.triu(keep, diagonal=1).nonzero().T


def lddt_matrix(pred: Tensor, ref: Tensor, cutoff: float = 15.0) -> Tensor:
    """Get the global lDDT of every prediction against every reference.

    Parameters
    ----------
    pred : Tensor
        (n, P) pair distances of the predictions.
    ref : Tensor
        (m, P) pair distances of the references on the same pairs.
    cutoff : float
        Inclusion radius on the reference distances.

    Returns
    -------
    Tensor
        (n, m) lDDT values.

    """
    mask = (ref < cutoff).float()
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
    """Get the Spearman correlation of x and y after removing the linear effect of z on the ranks."""
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
    bounds = torch.unique(torch.cat([torch.zeros(1, dtype=torch.float64), steps]).round(decimals=12))
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


def contact_mae(d_pred: Tensor, d_ref: Tensor, cutoff: float, width: float, set_min: float) -> float:
    """Get the mean absolute error of soft contact occupancies over the contacted groups."""
    occ_pred = torch.sigmoid((cutoff - d_pred) / width).mean(0)
    occ_ref = torch.sigmoid((cutoff - d_ref) / width).mean(0)
    contacted = (occ_pred > set_min) | (occ_ref > set_min)
    if not contacted.any():
        return float("nan")
    return (occ_pred - occ_ref)[contacted].abs().mean().item()


def bound_fraction_diff(
    d_pred: Tensor, d_ref: Tensor, cutoff: float, reference_min: float, fraction: float
) -> float:
    """Get the absolute difference of the bound fractions.

    A structure is bound when it keeps at least ``fraction`` of the reference contacts, the
    groups in contact in at least ``reference_min`` of the reference structures.
    """
    reference_contacts = (d_ref < cutoff).float().mean(0) >= reference_min
    if not reference_contacts.any():
        return float("nan")

    def bound(d: Tensor) -> float:
        kept = (d[:, reference_contacts] < cutoff).float().mean(1)
        return (kept >= fraction).float().mean().item()

    return abs(bound(d_pred) - bound(d_ref))


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

    Returns
    -------
    tuple[Tensor, Tensor]
        (n,) nucleic-acid clashes and (n,) ligand-nucleic acid clashes summed over ligands.

    """
    na, radii = system.na_atoms, system.radii
    keep_na = reference.clash_keep_na & (reference.clash_min_ratio_na >= factor)
    na_counts = torch.zeros(len(pred))
    lig_counts = torch.zeros(len(pred))
    for s, coords in enumerate(pred):
        ratio = clash_ratio(coords[None, na], coords[None, na], radii[na], radii[na])[0]
        na_counts[s] = ((ratio < factor) & keep_na).sum()
        for ligand, min_ratio in zip(system.ligands, reference.clash_min_ratio_lig):
            ratio = clash_ratio(coords[None, ligand], coords[None, na], radii[ligand], radii[na])[0]
            lig_counts[s] += ((ratio < factor) & (min_ratio >= factor)).sum()
    return na_counts, lig_counts


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
    and are not counted as clashes.

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

    res = system.na_residue
    keep = torch.triu(res[:, None] != res[None], diagonal=1)
    position = torch.full((len(system.radii),), -1, dtype=torch.long)
    position[system.na_atoms] = torch.arange(len(system.na_atoms))
    if len(system.link_neighbours):
        a, b = position[system.link_neighbours[:, 0]], position[system.link_neighbours[:, 1]]
        keep[torch.minimum(a, b), torch.maximum(a, b)] = False

    na_pairs, lig_pairs = pair_sets(frames, system, params.pair_cap, params.mobile_fraction)
    return Reference(
        frame0=frame0,
        lddt_pairs=lddt_pairs(frames, params.lddt_cutoff + params.lddt_pair_margin),
        pockets=tuple(
            pocket_atoms(frame0, ligand, system, params.pocket_cutoff) for ligand in system.ligands
        ),
        na_pairs=na_pairs,
        lig_pairs=lig_pairs,
        clash_keep_na=keep,
        clash_min_ratio_na=min_clash_ratio(frames, system.na_atoms, system.na_atoms, system.radii),
        clash_min_ratio_lig=tuple(
            min_clash_ratio(frames, ligand, system.na_atoms, system.radii) for ligand in system.ligands
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
    values = [v for v in values if v == v]
    return sum(values) / len(values) if values else float("nan")


def _log_ratio(pred: Tensor, ref: Tensor, floor: float) -> float:
    """Get log(median(pred) / median(ref)), with both medians floored."""
    return torch.log(median(pred).clamp(min=floor) / median(ref).clamp(min=floor)).item()


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
        The metric values.

    """
    out = {}
    out.update(_set_metrics(pred, ref_subset, reference, params))
    out.update(_pose_metrics(pred, ref_subset, reference, system))
    out.update(_rmsf_metrics(pred, ref_frames, reference, system, params))
    out.update(_contact_metrics(pred, ref_frames, system, params))
    out.update(_validity_metrics(pred, reference, system, params))
    out.update(pair_set_metrics(pred, ref_frames, reference.na_pairs, reference.lig_pairs, system, params))
    out.update(_backbone_metrics(pred, ref_frames, system))
    return out


def _set_metrics(pred, ref_subset, reference, params) -> dict[str, float]:
    """Precision, coverage and diversity from lDDT matrices."""
    pairs = reference.lddt_pairs
    d_pred = pair_distances(pred, pairs)
    d_ref = pair_distances(ref_subset, pairs)
    l_pr = lddt_matrix(d_pred, d_ref, params.lddt_cutoff)
    d_pp = _offdiag_mean(1 - lddt_matrix(d_pred, d_pred, params.lddt_cutoff))
    d_rr = _offdiag_mean(1 - lddt_matrix(d_ref, d_ref, params.lddt_cutoff))
    return {
        "lddt_nn": l_pr.max(1).values.mean().item(),
        "cov": l_pr.max(0).values.mean().item(),
        "div_ratio": math.log(max(d_pp, 1e-6) / d_rr) if d_rr > 0 else float("nan"),
    }


def _pose_metrics(pred, ref_subset, reference, system) -> dict[str, float]:
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


def _rmsf_metrics(pred, ref_frames, reference, system, params) -> dict[str, float]:
    """Where (Spearman) and how much (log ratio) the structures fluctuate."""
    rmsf_pred = rmsf(kabsch_align(pred, reference.frame0, system.backbone))
    rmsf_ref = rmsf(kabsch_align(ref_frames, reference.frame0, system.backbone))
    num_nt = system.num_nucleotides
    nt_pred = segment_mean(rmsf_pred[system.na_atoms], system.na_residue, num_nt)
    nt_ref = segment_mean(rmsf_ref[system.na_atoms], system.na_residue, num_nt)

    # Score of a model that only knows the distance from the fit centre
    centroids = segment_mean(reference.frame0[system.na_atoms].double(), system.na_residue, num_nt)
    centre_dist = (centroids - reference.frame0[system.backbone].double().mean(0)).norm(dim=-1)

    floor = params.rmsf_floor
    static = nt_pred.max() < floor  # identical structures: the ranks would be numerical noise
    return {
        "rmsf_spearman_na": float("nan") if static else spearman(nt_pred, nt_ref),
        "rmsf_spearman_na_null": spearman(centre_dist, nt_ref),
        "rmsf_logratio_na": _log_ratio(nt_pred, nt_ref, floor),
        "rmsf_logratio_lig": _nanmean([
            _log_ratio(rmsf_pred[ligand], rmsf_ref[ligand], floor) for ligand in system.ligands
        ]),
    }


def _contact_metrics(pred, ref_frames, system, params) -> dict[str, float]:
    """Ligand-nucleotide contact occupancy and bound fraction, averaged over ligands."""
    mae, bound = [], []
    for ligand in system.ligands:
        d_pred = ligand_contact_distances(pred, ligand, system)
        d_ref = ligand_contact_distances(ref_frames, ligand, system)
        mae.append(contact_mae(d_pred, d_ref, params.contact_cutoff, params.contact_width, params.contact_set_min))
        bound.append(bound_fraction_diff(
            d_pred, d_ref, params.bound_cutoff, params.bound_reference_min, params.bound_fraction
        ))
    if not system.ligands:
        return {}
    return {"lig_contact_mae": _nanmean(mae), "bound_frac_diff": _nanmean(bound)}


def _validity_metrics(pred, reference, system, params) -> dict[str, float]:
    """Clash-free and chain-break fractions over the structures."""
    na_counts, lig_counts = clash_counts(pred, reference, system, params.clash_factor)
    out = {"clash_free_na": (na_counts == 0).float().mean().item()}
    if system.ligands:
        out["clash_free_lig_na"] = (lig_counts == 0).float().mean().item()
    if len(system.links):
        link = pair_distances(pred, system.links.T)
        out["chain_break"] = (link > params.chain_break_cutoff).any(1).float().mean().item()
        out["chain_break_severe"] = (link > params.chain_break_severe_cutoff).any(1).float().mean().item()
    return out


def pair_set_metrics(
    pred: Tensor,
    ref_frames: Tensor,
    na_pairs: PairSet,
    lig_pairs: tuple[PairSet, ...],
    system: SystemIndex,
    params: MetricParams,
) -> dict[str, float]:
    """Distance-distribution (W1) and distance-fluctuation metrics on given pair sets."""
    out = {}
    if na_pairs.pairs.shape[1] >= 4:
        nt_pairs = system.rep_atoms[na_pairs.pairs]
        d_pred, d_ref = pair_distances(pred, nt_pairs), pair_distances(ref_frames, nt_pairs)
        mobile = na_pairs.mobile
        std_pred, std_ref = d_pred.std(0), d_ref.std(0)
        out["w1_na_mobile"] = wasserstein_1d(d_pred[:, mobile].T, d_ref[:, mobile].T).mean().item()
        out["dfluct_logratio_na"] = _log_ratio(std_pred[mobile], std_ref[mobile], params.rmsf_floor)
        out["dfluct_pspearman_na"] = partial_spearman(std_pred, std_ref, d_ref.mean(0))

    w1 = []
    for lig_set in lig_pairs:
        if lig_set.pairs.shape[1] < 4:
            continue
        d_pred = ligand_pair_distances(pred, lig_set.pairs, system)[:, lig_set.mobile]
        d_ref = ligand_pair_distances(ref_frames, lig_set.pairs, system)[:, lig_set.mobile]
        w1.append(wasserstein_1d(d_pred.T, d_ref.T).mean().item())
    if w1:
        out["w1_lig_na_mobile"] = _nanmean(w1)
    return out


def _backbone_metrics(pred, ref_frames, system) -> dict[str, float]:
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
