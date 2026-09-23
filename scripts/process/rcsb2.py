"""Process nucleic-acid (RNA/DNA + ligand) mmCIF ensembles into Boltz-2 targets.

This is the StructureV2 / Boltz-2 analogue of ``rcsb.py``. It is meant for our
protein-free MD dataset, where each ``.cif`` is a multi-model ensemble (all
trajectory frames as gemmi models, per-atom RMSF in the B-factors). Those
ensembles only survive in a ``StructureV2`` (which carries the ``coords`` +
``ensemble`` tables), so this script parses with the Boltz-2 parser under
``src/boltz`` rather than the legacy V1 parser next to ``rcsb.py``.

Differences vs. ``rcsb.py``:

  * parses with ``boltz.data.parse.mmcif.parse_mmcif`` (StructureV2), the same
    parser ``create_cluster_file.py`` uses to build ``clustering.json``;
  * CCD molecules come from a local ``--moldir`` of ``<CCD>.pkl`` files (loaded
    lazily by the parser) -- no Redis;
  * no protein / MSA handling: every chain gets ``msa_id = ""``;
  * cluster ids are resolved by ``hash_sequence(sequence)`` for polymers and by
    the (lowercased) first-residue CCD code for ligands, matching the keys
    ``create_cluster_file.py`` writes.

Consistency contract: this script and ``create_cluster_file.py`` MUST parse with
the same parser and ``use_assembly=False`` and hash the same
``gemmi.one_letter_code`` string, or every ``cluster_id`` falls through to -1.

Run::

    python rcsb2.py \\
        --datadir PATH_TO_MMCIF_DIR \\
        --clusters clustering.json \\
        --moldir ~/.boltz/mols \\
        --outdir YOUR_OUTPUT_DIR
"""

import argparse
import json
import multiprocessing
import traceback
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import rdkit
from cluster import hash_sequence
from boltz.data.parse.mmcif import parse_mmcif
from tqdm import tqdm

from boltz.data import const
from boltz.data.filter.static.filter import StaticFilter
from boltz.data.filter.static.ligand import ExcludedLigands
from boltz.data.filter.static.polymer import (
    ClashingChainsFilter,
    ConsecutiveCA,
    MinimumLengthFilter,
    UnknownFilter,
)
from boltz.data.types import ChainInfo, InterfaceInfo, Record, Target


class PDB:
    """A raw mmCIF file to process."""

    def __init__(self, id: str, path: str) -> None:  # noqa: A002
        self.id = id
        self.path = path


def fetch(datadir: Path, max_file_size: Optional[int] = None) -> list[PDB]:
    """Fetch the mmCIF files under ``datadir``."""
    data = []
    excluded = 0
    for file in datadir.rglob("*.cif*"):
        # The id must match the key convention used by clustering.json:
        # the (lowercased) file stem.
        pdb_id = str(file.stem).lower()

        # Check file size and skip if too large
        if max_file_size is not None and (file.stat().st_size > max_file_size):
            excluded += 1
            continue

        data.append(PDB(id=pdb_id, path=str(file)))

    print(f"Excluded {excluded} files due to size.")  # noqa: T201
    return data


def finalize(outdir: Path) -> None:
    """Aggregate every per-structure record into a single ``manifest.json``."""
    records_dir = outdir / "records"

    failed_count = 0
    records = []
    for record in records_dir.iterdir():
        try:
            with record.open("r") as f:
                records.append(json.load(f))
        except:  # noqa: E722
            failed_count += 1
            print(f"Failed to parse {record}")  # noqa: T201
    if failed_count > 0:
        print(f"Failed to parse {failed_count} entries.")  # noqa: T201
    else:
        print("All entries parsed successfully.")  # noqa: T201

    outpath = outdir / "manifest.json"
    with outpath.open("w") as f:
        json.dump(records, f)


def parse(data: PDB, moldir: str, clusters: dict) -> Target:
    """Parse one mmCIF into a StructureV2 target with cluster-annotated chains."""
    # Parse structure. use_assembly=False is load-bearing: it must match the
    # parse settings in create_cluster_file.py so the hashed sequences (and thus
    # the clustering.json keys) line up. compute_interfaces=True populates the
    # interface table used for interface-level sampling.
    parsed = parse_mmcif(
        str(data.path),
        moldir=moldir,
        use_assembly=False,
        compute_interfaces=True,
    )
    structure = parsed.data
    structure_info = parsed.info

    nonpolymer_id = const.chain_type_ids["NONPOLYMER"]

    # Create chain metadata
    chain_info = []
    for i, chain in enumerate(structure.chains):
        mol_type = int(chain["mol_type"])
        name = str(chain["name"])

        # Resolve the cluster key the same way create_cluster_file.py keys its
        # output: polymers by hash of their 1-letter sequence, ligands by their
        # (lowercased) first-residue CCD code. No protein/MSA branch here -- the
        # dataset is protein-free, so msa_id is always empty.
        sequence = parsed.sequences.get(name)
        if sequence is not None:
            cluster_key = hash_sequence(sequence)
        elif mol_type == nonpolymer_id:
            res_start = int(chain["res_idx"])
            cluster_key = str(structure.residues[res_start]["name"]).lower()
        else:
            cluster_key = None

        cluster_id = clusters.get(cluster_key, -1) if cluster_key is not None else -1

        chain_info.append(
            ChainInfo(
                chain_id=i,
                chain_name=name,
                msa_id="",
                mol_type=mol_type,
                cluster_id=cluster_id,
                num_residues=int(chain["res_num"]),
                entity_id=int(chain["entity_id"]),
            )
        )

    # Get interface metadata
    interface_info = []
    for interface in structure.interfaces:
        interface_info.append(
            InterfaceInfo(
                chain_1=int(interface["chain_1"]),
                chain_2=int(interface["chain_2"]),
            )
        )

    # Create record
    record = Record(
        id=data.id,
        structure=structure_info,
        chains=chain_info,
        interfaces=interface_info,
    )

    return Target(structure=structure, record=record)


def process_structure(
    data: PDB,
    moldir: str,
    outdir: Path,
    filters: list[StaticFilter],
    clusters: dict,
) -> None:
    """Parse, filter, and dump one structure + record."""
    struct_path = outdir / "structures" / f"{data.id}.npz"
    record_path = outdir / "records" / f"{data.id}.json"

    if struct_path.exists() and record_path.exists():
        return

    try:
        # Parse the target
        target: Target = parse(data, moldir, clusters)
        structure = target.structure

        # Apply the filters -> per-chain validity mask
        mask = structure.mask
        if filters is not None:
            for f in filters:
                mask = mask & f.filter(structure)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print(f"Failed to parse {data.id}")  # noqa: T201
        return

    # Write the validity mask back into chains and interfaces
    chains = []
    for i, chain in enumerate(target.record.chains):
        chains.append(replace(chain, valid=bool(mask[i])))

    interfaces = []
    for interface in target.record.interfaces:
        chain_1 = bool(mask[interface.chain_1])
        chain_2 = bool(mask[interface.chain_2])
        interfaces.append(replace(interface, valid=(chain_1 and chain_2)))

    structure = replace(structure, mask=mask)
    record = replace(target.record, chains=chains, interfaces=interfaces)

    # Dump StructureV2 (asdict -> atoms, bonds, residues, chains, interfaces,
    # mask, coords, ensemble, pocket; no V1 `connections`). Loadable directly by
    # StructureV2.load.
    np.savez_compressed(struct_path, **asdict(structure))

    with record_path.open("w") as f:
        json.dump(asdict(record), f)


def process(args) -> None:
    """Run the data processing task."""
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "records").mkdir(parents=True, exist_ok=True)
    (args.outdir / "structures").mkdir(parents=True, exist_ok=True)

    # Load clusters (lowercased keys/values, matching create_cluster_file.py's
    # lowercased ligand keys).
    with Path(args.clusters).open("r") as f:
        clusters: dict[str, str] = json.load(f)
        clusters = {k.lower(): v.lower() for k, v in clusters.items()}

    if not Path(args.moldir).is_dir():
        raise SystemExit(f"--moldir {args.moldir} is not a directory")
    moldir = str(args.moldir)

    # Static filters (all StructureV2-safe: they read only chains/residues/atoms).
    filters = [
        ExcludedLigands(),
        MinimumLengthFilter(min_len=4, max_len=5000),
        UnknownFilter(),
        ConsecutiveCA(max_dist=10.0),
        ClashingChainsFilter(freq=0.3, dist=1.7),
    ]

    # Keep RDKit properties when pickling mols.
    rdkit.Chem.SetDefaultPickleProperties(rdkit.Chem.PropertyPickleOptions.AllProps)

    print("Fetching data...")  # noqa: T201
    data = fetch(args.datadir, args.max_file_size)

    max_processes = multiprocessing.cpu_count()
    num_processes = max(1, min(args.num_processes, max_processes, len(data)))
    parallel = num_processes > 1

    print("Processing data...")  # noqa: T201
    if parallel:
        # p_tqdm is a processing-only extra (see requirements.txt); import it
        # lazily so serial runs work without it.
        from p_tqdm import p_umap

        # Each worker builds its own per-call CCD cache from moldir; no shared
        # mols dict is passed across processes.
        fn = partial(
            process_structure,
            moldir=moldir,
            outdir=args.outdir,
            clusters=clusters,
            filters=filters,
        )
        p_umap(fn, data, num_cpus=num_processes)
    else:
        for item in tqdm(data):
            process_structure(
                item,
                moldir=moldir,
                outdir=args.outdir,
                clusters=clusters,
                filters=filters,
            )

    finalize(args.outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process NA mmCIF ensembles into Boltz-2 StructureV2 targets."
    )
    parser.add_argument(
        "--datadir",
        type=Path,
        required=True,
        help="Directory containing the mmCIF (.cif) files.",
    )
    parser.add_argument(
        "--clusters",
        type=Path,
        required=True,
        help="Path to clustering.json (from create_cluster_file.py).",
    )
    parser.add_argument(
        "--moldir",
        type=Path,
        required=True,
        help="Boltz CCD molecules directory (<code>.pkl); needed by parse_mmcif.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default="data",
        help="The output directory.",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=multiprocessing.cpu_count(),
        help="The number of processes.",
    )
    parser.add_argument(
        "--max-file-size",
        type=int,
        default=None,
        help="Skip .cif files larger than this many bytes.",
    )
    args = parser.parse_args()
    process(args)
