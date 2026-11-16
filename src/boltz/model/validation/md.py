import torch
from torch import Tensor, nn
from torchmetrics import CatMetric, MeanMetric

from boltz.data import const
from boltz.model.loss.bfactor import bfactor_loss_fn
from boltz.model.loss.distogramv2 import distogram_loss
from boltz.model.loss.validation import factored_lddt_loss, weighted_minimum_rmsd
from boltz.model.validation import md_metrics as mm

# Scores of the single structures and the losses of the heads, averaged over the records
MEAN_METRICS = [
    "lddt",
    "lddt_best",
    "rmsd",
    "rmsd_best",
    "disto_loss",
    "bfactor_loss",
    *(f"lddt_{name}{suffix}" for name in const.out_types for suffix in ("", "_best")),
]

# Covalently bonded heavy atoms are closer than this in MD, non-bonded ones never are
BOND_MAX_DISTANCE = 2.0

# How many MD frames of a record the set-to-set metrics compare the samples with. Taking about
# as many as there are samples keeps the comparison symmetric; they are spread evenly over the
# trajectory, so the number of frames a record has does not change how many are used.
NUM_REFERENCE_FRAMES = 20

NA_TYPES = {const.chain_type_ids["DNA"]: "dna", const.chain_type_ids["RNA"]: "rna"}
RECORD_TYPES = ("dna", "rna", "hybrid")


class MDValidator(nn.Module):
    """Score sampled structures against the MD frames of each record.

    The lDDT and RMSD metrics compare with MD frame 0 and are averaged over records. The
    ensemble metrics compare with all frames (``md_metrics``) and are summarised by their
    median over records, overall and per record type (DNA, RNA, hybrid).
    """

    def __init__(self, val_names: list[str], seed: int = 0) -> None:
        """Initialize the validator.

        Parameters
        ----------
        val_names : list[str]
            The labels of the validation groups to score.
        seed : int
            The base seed of the sampling noise; the record key is added to it.

        """
        super().__init__()
        self.val_names = val_names
        self.seed = seed
        self.mean_metrics = nn.ModuleDict({name: MeanMetric() for name in MEAN_METRICS})
        # One value per record, reduced to a median at the end of the round. "disable" keeps
        # the NaNs of records where a metric is undefined, so the values stay aligned with
        # record_type_codes.
        self.ensemble_metrics = nn.ModuleDict(
            {name: CatMetric(nan_strategy="disable") for name in mm.METRIC_NAMES}
        )
        # The type of each scored record, so the medians can also be reported per type
        self.record_type_codes = CatMetric()
        self.params = mm.MetricParams()
        # Used for caching the reference structure for calculating metrics
        self._references = {}

    def run_model(self, model: nn.Module, batch: dict[str, Tensor], idx_dataset: int):
        """Sample structures for the batch with the validation settings.

        The noise is fixed per record, and the global RNG is restored afterwards so
        the training noise is unaffected.

        Parameters
        ----------
        model : nn.Module
            The model to evaluate.
        batch : dict[str, Tensor]
            The input features.
        idx_dataset : int
            The index of the validation group.

        Returns
        -------
        dict[str, Tensor]
            The model output.

        """
        args = model.validation_args
        with torch.random.fork_rng():
            # Batch Size for validation is 1. We can directly access the record key
            torch.manual_seed(self.seed + int(batch["record_key"]))
            return model(
                batch,
                recycling_steps=args.recycling_steps,
                num_sampling_steps=args.sampling_steps,
                diffusion_samples=args.diffusion_samples,
            )

    def process(
        self,
        model: nn.Module,
        batch: dict[str, Tensor],
        out: dict[str, Tensor],
        idx_dataset: int,  # noqa: ARG002
    ) -> None:
        """Accumulate all validation metrics of one record.

        Parameters
        ----------
        model : nn.Module
            The evaluated model.
        batch : dict[str, Tensor]
            The input features.
        out : dict[str, Tensor]
            The model output.
        idx_dataset : int
            The index of the validation group.

        """
        samples = out["sample_atom_coords"]
        self._process_frame0(batch, samples)
        self._process_ensemble(batch, samples)
        self._process_heads(model, batch, out)

    def _process_frame0(self, batch: dict[str, Tensor], samples: Tensor) -> None:
        """Accumulate the lDDT and RMSD of the sampled structures against MD frame 0."""
        num_samples = samples.shape[0]
        true_coords = batch["coords"][:, 0].repeat_interleave(num_samples, 0)
        atom_mask = batch["atom_resolved_mask"].repeat_interleave(num_samples, 0)

        lddt, total = factored_lddt_loss(
            true_coords, samples, batch, atom_mask, multiplicity=num_samples
        )
        for name, value in lddt.items():
            weight = total[name][0]
            if weight > 0:
                self._update(f"lddt_{name}", value.mean(), weight)
                self._update(f"lddt_{name}_best", value.max(), weight)

        # Pair-weighted lDDT over all modalities, one value per sample
        pairs = torch.stack([total[name] for name in lddt]).sum(0)
        overall = sum(lddt[name] * total[name] for name in lddt) / pairs.clamp(min=1)
        self._update("lddt", overall.mean())
        self._update("lddt_best", overall.max())

        rmsd, best_rmsd = weighted_minimum_rmsd(samples, batch, multiplicity=num_samples)
        self._update("rmsd", rmsd.mean())
        self._update("rmsd_best", best_rmsd.mean())

    def _process_ensemble(self, batch: dict[str, Tensor], samples: Tensor) -> None:
        """Accumulate the ensemble metrics of the sampled structures against all MD frames."""
        frame_idx = batch["ensemble_ref_idxs"][0].cpu()
        if not torch.equal(frame_idx, torch.arange(len(frame_idx))):
            msg = (
                "The MD validator needs all MD frames of a record in trajectory order; "
                "set data.val_num_ensembles to null"
            )
            raise ValueError(msg)

        # Rebuilding the index is cheap, so a cached reference is only reused when it was
        # built for the same atoms and the same frame 0; a record that is ever cropped
        # differently, or a key shared by two records, gets a fresh reference
        system, atoms, record_type = build_system_index(batch)
        frames = batch["coords"][0][:, atoms].cpu().float()
        key = int(batch["record_key"])
        cached = self._references.get(key)
        if cached is None or not (
            torch.equal(cached[0], atoms) and torch.equal(cached[1].frame0, frames[0])
        ):
            cached = (atoms, mm.build_reference(frames, system, self.params))
            self._references[key] = cached
        reference = cached[1]
        values = mm.score_ensemble(
            samples[:, atoms].detach().cpu().float(),
            frames,
            frames[:-1 : _reference_stride(len(frames))],
            reference,
            system,
            self.params,
        )
        # The metric states follow the module to the accelerator, so the updates must too
        device = self.record_type_codes.device
        for name, value in values.items():
            self.ensemble_metrics[name].update(torch.tensor(value, device=device))
        self.record_type_codes.update(torch.tensor(RECORD_TYPES.index(record_type), device=device))

    def _process_heads(
        self, model: nn.Module, batch: dict[str, Tensor], out: dict[str, Tensor]
    ) -> None:
        """Accumulate the losses of the distogram and bfactor heads."""
        disto_loss, _ = distogram_loss(
            out, batch, aggregate_distogram=model.aggregate_distogram
        )
        self._update("disto_loss", disto_loss)
        if model.predict_bfactor:
            self._update(
                "bfactor_loss",
                bfactor_loss_fn(out, batch, model.bfactor_module.boundaries),
            )

    def on_epoch_end(self, model: nn.Module) -> None:
        """Log and reset the accumulated metrics.

        Parameters
        ----------
        model : nn.Module
            The model used for logging.

        """
        for name, metric in self.mean_metrics.items():
            if metric.update_called:
                model.log(f"val/{name}", metric.compute(), prog_bar=name == "lddt")
                metric.reset()

        if self.record_type_codes.update_called:
            types = self.record_type_codes.compute()
            for name, metric in self.ensemble_metrics.items():
                values = metric.compute()
                # The count says how many records the median covers, since a metric is NaN
                # where it is undefined for a record (no ligand, too few bases or pairs)
                model.log(f"val/{name}_n", (~values.isnan()).sum().float())
                model.log(f"val/{name}", _median(values))
                for code, label in enumerate(RECORD_TYPES):
                    if (types == code).any():
                        model.log(f"val/{name}_{label}", _median(values[types == code]))
                metric.reset()
            self.record_type_codes.reset()

    def _update(self, name: str, value: Tensor, weight: Tensor = 1.0) -> None:
        """Update a metric."""
        self.mean_metrics[name].update(value, weight)


def _reference_stride(num_frames: int) -> int:
    """Get the step that spreads about ``NUM_REFERENCE_FRAMES`` references over a trajectory."""
    return max(1, (num_frames - 1) // NUM_REFERENCE_FRAMES)


def _median(values: Tensor) -> Tensor:
    """Get the median of the defined values, averaging the two middle ones.

    NaN if no value is defined.
    """
    defined = values[~values.isnan()]
    if len(defined) == 0:
        return torch.tensor(float("nan"), device=values.device)
    return defined.double().quantile(0.5).to(values.dtype)


def build_system_index(batch: dict[str, Tensor]) -> tuple[mm.SystemIndex, Tensor, str]:
    """Get the atom indices of a nucleic acid + ligand record from its features.

    Linked nucleotides (O3'-P) and bonded ligand chains are found from their distances in
    MD frame 0, so links between chains are included.

    Parameters
    ----------
    batch : dict[str, Tensor]
        The input features of one record.

    Returns
    -------
    tuple[mm.SystemIndex, Tensor, str]
        The system index on the kept atoms, the kept (present, non-padding) atoms, and the
        record type.

    """
    keys = (
        "atom_pad_mask",
        "atom_resolved_mask",
        "atom_to_token",
        "token_to_rep_atom",
        "token_to_center_atom",
        "mol_type",
        "asym_id",
        "ref_atom_name_chars",
        "ref_element",
        "coords",
    )
    feats = {key: batch[key][0].cpu() for key in keys}
    atoms = (feats["atom_pad_mask"] * feats["atom_resolved_mask"]).bool().nonzero().flatten()
    # Position of each padded atom within the kept selection, -1 if dropped, e.g. [-1, 0, 1, -1, 2]
    compact = torch.full((len(feats["atom_pad_mask"]),), -1, dtype=torch.long)
    compact[atoms] = torch.arange(len(atoms))
    token = feats["atom_to_token"][atoms].argmax(-1)
    mol_type = feats["mol_type"][token]
    names = [_atom_name(codes) for codes in feats["ref_atom_name_chars"][atoms].argmax(-1)]
    frame0 = feats["coords"][0, atoms].float()

    na_atoms = torch.isin(mol_type, torch.tensor(list(NA_TYPES))).nonzero().flatten()
    na_tokens, na_residue = torch.unique(token[na_atoms], return_inverse=True)
    by_name = [{} for _ in na_tokens]
    for atom, residue in zip(na_atoms.tolist(), na_residue.tolist()):
        by_name[residue].setdefault(names[atom], atom)

    # Representative atom, else the centre atom, else the first atom of the nucleotide
    rep = compact[feats["token_to_rep_atom"][na_tokens].argmax(-1)].tolist()
    center = compact[feats["token_to_center_atom"][na_tokens].argmax(-1)].tolist()
    rep_atoms = [
        r if r >= 0 else (c if c >= 0 else min(atom_names.values()))
        for r, c, atom_names in zip(rep, center, by_name)
    ]

    links, link_neighbours = _links(by_name, frame0)
    system = mm.SystemIndex(
        na_atoms=na_atoms,
        na_residue=na_residue,
        rep_atoms=torch.tensor(rep_atoms, dtype=torch.long),
        c4_atoms=torch.tensor([a.get("C4'", -1) for a in by_name], dtype=torch.long),
        base_atoms=torch.tensor(
            [mm.base_ring_atoms(a) for a in by_name], dtype=torch.long
        ).reshape(-1, 3),
        backbone=torch.tensor(
            [a for a in na_atoms.tolist() if names[a] in mm.BACKBONE_ATOM_NAMES], dtype=torch.long
        ),
        links=torch.tensor(links, dtype=torch.long).reshape(-1, 2),
        link_neighbours=torch.tensor(link_neighbours, dtype=torch.long).reshape(-1, 2),
        radii=torch.tensor(const.vdw_radii)[feats["ref_element"][atoms].argmax(-1) - 1],
        ligands=_ligands(mol_type, feats["asym_id"][token], frame0),
    )
    types = {NA_TYPES[t] for t in mol_type[na_atoms].unique().tolist()}
    record_type = types.pop() if len(types) == 1 else "hybrid"
    return system, atoms, record_type


def _links(
    by_name: list[dict[str, int]], frame0: Tensor
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Get the O3'-P links of a record and the atom pairs they explain.

    A link is an O3' and a P of different nucleotides within bonding distance in frame 0, so
    links between chains are found as well.

    Parameters
    ----------
    by_name : list[dict[str, int]]
        Per nucleotide, its atoms by name.
    frame0 : Tensor
        (N, 3) the first MD frame.

    Returns
    -------
    tuple[list, list]
        The (O3', P) atom pairs, and the pairs around them that never count as clashing.

    """
    o3_atoms = [(res, atoms["O3'"]) for res, atoms in enumerate(by_name) if "O3'" in atoms]
    p_atoms = [(res, atoms["P"]) for res, atoms in enumerate(by_name) if "P" in atoms]
    if not (o3_atoms and p_atoms):
        return [], []

    dist = torch.cdist(frame0[[a for _, a in o3_atoms]], frame0[[a for _, a in p_atoms]])
    links, neighbours = [], []
    for i, j in (dist < BOND_MAX_DISTANCE).nonzero().tolist():
        (upstream_res, o3), (downstream_res, p) = o3_atoms[i], p_atoms[j]
        if upstream_res == downstream_res:
            continue
        links.append((o3, p))
        upstream, downstream = by_name[upstream_res], by_name[downstream_res]
        neighbours.extend(
            (upstream[a], downstream[b])
            for a, b in mm.LINK_EXCLUDED_PAIRS
            if a in upstream and b in downstream
        )
    return links, neighbours


def _atom_name(codes: Tensor) -> str:
    """Decode an atom name from its character codes (``convert_atom_name``)."""
    return "".join(chr(code + 32) for code in codes.tolist() if code > 0)


def _ligands(mol_type: Tensor, asym_id: Tensor, frame0: Tensor) -> tuple[Tensor, ...]:
    """Get the atoms of each ligand; ligand chains bonded to each other form one ligand."""
    ligand_atoms = (mol_type == const.chain_type_ids["NONPOLYMER"]).nonzero().flatten()
    chains = [
        ligand_atoms[asym_id[ligand_atoms] == chain] for chain in asym_id[ligand_atoms].unique()
    ]
    ligands = []
    for atoms in chains:
        bonded = [
            i for i, ligand in enumerate(ligands)
            if torch.cdist(frame0[atoms], frame0[ligand]).min() < BOND_MAX_DISTANCE
        ]
        merged = torch.cat([atoms, *(ligands[i] for i in bonded)]).sort().values
        ligands = [ligand for i, ligand in enumerate(ligands) if i not in bonded] + [merged]
    return tuple(ligands)
