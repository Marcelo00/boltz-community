"""Process nucleic-acid (RNA/DNA + ligand) mmCIF ensembles into Boltz-2 targets.

This is the StructureV2 / Boltz-2 analogue of ``rcsb.py``. It is meant for our
protein-free MD dataset, where each ``.cif`` is a multi-model ensemble (all
trajectory frames as gemmi models, per-atom RMSF in the B-factors). Those
ensembles only survive in a ``StructureV2`` (which carries the ``coords`` +
``ensemble`` tables), so this script parses with the Boltz-2 parser under
``src/boltz`` rather than the legacy V1 parser next to ``rcsb.py``.

Differences vs. ``rcsb.py``:

  * parses with ``boltz.data.parse.mmcif.parse_mmcif`` (StructureV2), the same
    parser ``clusterv2.py`` uses to build ``clustering.json``;
  * CCD molecules come from a local ``--moldir`` of ``<CCD>.pkl`` files (loaded
    lazily by the parser) -- no Redis;
  * no protein / MSA handling: every chain gets ``msa_id = ""``;
  * cluster ids are resolved by ``hash_sequence(sequence)`` for polymers and by
    the (lowercased) first-residue CCD code for ligands, matching the keys
    ``clusterv2.py`` writes.

Consistency contract: this script and ``clusterv2.py`` MUST parse with
the same parser and ``use_assembly=False`` and hash the same
``gemmi.one_letter_code`` string, or every ``cluster_id`` falls through to -1.

A file that fails to parse is skipped and reported in ``rcsbv2_report.csv`` in
``--outdir`` (same columns as ``build_mmcif_report.csv``); the run only aborts when more than
``--max-failure-fraction`` of the files fail. Re-running resumes: finished
structures are kept and the manifest covers ``--datadir`` only.

Run::

    python rcsbv2.py \\
        --datadir PATH_TO_MMCIF_DIR \\
        --clusters clustering.json \\
        --moldir ~/.boltz/mols \\
        --outdir YOUR_OUTPUT_DIR
"""

import argparse
import csv
import json
import multiprocessing
from collections import Counter
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import rdkit
from clusterv2 import hash_sequence
from tqdm import tqdm

from boltz.data import const
from boltz.data.filter.static.filter import StaticFilter
from boltz.data.filter.static.ligand import ExcludedLigands
from boltz.data.filter.static.polymer import (
    ClashingChainsFilter,
    MinimumLengthFilter,
    UnknownFilter,
)
from boltz.data.parse.mmcif import parse_mmcif
from boltz.data.types import ChainInfo, InterfaceInfo, Record, StructureV2


@dataclass(frozen=True)
class TargetV2:
    """Parsed structure + record. ``types.Target`` is typed for the V1 ``Structure``."""

    record: Record
    structure: StructureV2


REPORT_FILENAME = "rcsbv2_report.csv"


@dataclass(frozen=True)
class Outcome:
    """Result of processing one structure: ok, skipped (done before) or error."""

    id: str
    status: str = "ok"
    error: str = ""


class PDB:
    """A raw mmCIF file to process."""

    def __init__(self, id: str, path: str) -> None:  # noqa: A002
        self.id = id
        self.path = path


def fetch(datadir: Path, max_file_size: Optional[int] = None) -> list[PDB]:
    """Fetch the mmCIF files under ``datadir``."""
    data = []
    excluded = 0
    for file in sorted(datadir.rglob("*.cif")):
        # The id must match the key convention used by clustering.json:
        # the (lowercased) file stem.
        pdb_id = str(file.stem).lower()

        # Check file size and skip if too large
        if max_file_size is not None and (file.stat().st_size > max_file_size):
            excluded += 1
            continue

        data.append(PDB(id=pdb_id, path=str(file)))

    print(f"Excluded {excluded} files due to size.")  # noqa: T201

    counts = Counter(pdb.id for pdb in data)
    duplicates = sorted(pdb_id for pdb_id, count in counts.items() if count > 1)
    if duplicates:
        msg = f"Several files share the structure id: {duplicates}"
        raise SystemExit(msg)
    return data


def finalize(outdir: Path, ids: list[str]) -> None:
    """Aggregate the records of ``ids`` into a single ``manifest.json``."""
    records = []
    for pdb_id in ids:
        record_path = outdir / "records" / f"{pdb_id}.json"
        if record_path.exists():
            records.append(json.loads(record_path.read_text()))

    chains = [chain for record in records for chain in record["chains"]]
    unclustered = sum(chain["cluster_id"] == -1 for chain in chains)
    print(f"{unclustered}/{len(chains)} chains have no cluster id.")  # noqa: T201
    if chains and unclustered == len(chains):
        msg = "No chain matches clustering.json; check that it was built from these files."
        raise SystemExit(msg)

    with (outdir / "manifest.json").open("w") as f:
        json.dump(records, f)


def write_report(outdir: Path, outcomes: list[Outcome]) -> None:
    """Write one ``sim_id, status, error`` row per structure."""
    with (outdir / REPORT_FILENAME).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sim_id", "status", "error"])
        writer.writerows((o.id, o.status, o.error) for o in outcomes)


def parse(data: PDB, moldir: str, clusters: dict) -> TargetV2:
    """Parse one mmCIF into a StructureV2 target with cluster-annotated chains."""
    # Parse structure. use_assembly=False is load-bearing: it must match the
    # parse settings in clusterv2.py so the hashed sequences (and thus
    # the clustering.json keys) line up. call_compute_interfaces=True populates the
    # interface table used for interface-level sampling.
    parsed = parse_mmcif(
        str(data.path),
        moldir=moldir,
        use_assembly=False,
        call_compute_interfaces=True,
    )
    structure = parsed.data
    structure_info = parsed.info

    nonpolymer_id = const.chain_type_ids["NONPOLYMER"]

    # Create chain metadata
    chain_info = []
    for i, chain in enumerate(structure.chains):
        mol_type = int(chain["mol_type"])
        name = str(chain["name"])

        # Resolve the cluster key the same way clusterv2.py keys its
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

    return TargetV2(structure=structure, record=record)


def apply_filters(structure: StructureV2, filters: list[StaticFilter]) -> np.ndarray:
    """Combine the structure mask with the per-chain validity of every filter."""
    mask = structure.mask
    for static_filter in filters:
        mask = mask & static_filter.filter(structure)
    return mask


def save_structure(path: Path, structure: StructureV2) -> None:
    """Write the structure so that ``path`` only exists once it is complete."""
    temporary = path.with_name(f"{path.stem}.tmp.npz")
    np.savez_compressed(temporary, **asdict(structure))
    temporary.replace(path)


def save_record(path: Path, record: Record) -> None:
    """Write the record so that ``path`` only exists once it is complete."""
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(asdict(record)))
    temporary.replace(path)


def process_structure(
    data: PDB,
    moldir: str,
    outdir: Path,
    filters: list[StaticFilter],
    clusters: dict,
) -> Outcome:
    """Parse, filter, and dump one structure + record."""
    struct_path = outdir / "structures" / f"{data.id}.npz"
    record_path = outdir / "records" / f"{data.id}.json"

    if struct_path.exists() and record_path.exists():
        return Outcome(data.id, "skipped")

    try:
        target: TargetV2 = parse(data, moldir, clusters)
        mask = apply_filters(target.structure, filters)
    except Exception as exc:  # noqa: BLE001
        return Outcome(data.id, "error", f"{type(exc).__name__}: {exc}")

    # Write the validity mask back into chains and interfaces
    chains = [
        replace(chain, valid=bool(mask[i]))
        for i, chain in enumerate(target.record.chains)
    ]
    interfaces = [
        replace(
            interface,
            valid=bool(mask[interface.chain_1] and mask[interface.chain_2]),
        )
        for interface in target.record.interfaces
    ]
    record = replace(target.record, chains=chains, interfaces=interfaces)

    # Dump StructureV2 (asdict -> atoms, bonds, residues, chains, interfaces,
    # mask, coords, ensemble, pocket; no V1 `connections`). Loadable directly by
    # StructureV2.load. The record goes last: it marks the structure as done.
    save_structure(struct_path, replace(target.structure, mask=mask))
    save_record(record_path, record)
    return Outcome(data.id)


def run_all(fn, data: list[PDB], num_processes: int) -> list[Outcome]:
    """Process every structure, in parallel when ``num_processes`` > 1."""
    if num_processes <= 1:
        return list(tqdm(map(fn, data), total=len(data)))
    with multiprocessing.Pool(processes=num_processes) as pool:
        return list(tqdm(pool.imap_unordered(fn, data), total=len(data)))


def process(args) -> None:
    """Run the data processing task."""
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "records").mkdir(parents=True, exist_ok=True)
    (args.outdir / "structures").mkdir(parents=True, exist_ok=True)

    # Load clusters (lowercased keys/values, matching clusterv2.py's
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
        ClashingChainsFilter(freq=0.3, dist=1.7),
    ]

    # Keep RDKit properties when pickling mols.
    rdkit.Chem.SetDefaultPickleProperties(rdkit.Chem.PropertyPickleOptions.AllProps)

    print("Fetching data...")  # noqa: T201
    data = fetch(args.datadir, args.max_file_size)
    if not data:
        msg = f"No .cif files found under {args.datadir}"
        raise SystemExit(msg)

    num_processes = max(1, min(args.num_processes, multiprocessing.cpu_count(), len(data)))

    print("Processing data...")  # noqa: T201
    # Each worker builds its own per-call CCD cache from moldir; no shared mols
    # dict is passed across processes.
    fn = partial(
        process_structure,
        moldir=moldir,
        outdir=args.outdir,
        clusters=clusters,
        filters=filters,
    )
    outcomes = run_all(fn, data, num_processes)

    failed = [outcome for outcome in outcomes if outcome.status == "error"]
    write_report(args.outdir, outcomes)
    finalize(args.outdir, [pdb.id for pdb in data])
    print(f"Processed {len(data) - len(failed)}/{len(data)} structures.")  # noqa: T201
    if failed:
        print(f"{len(failed)} failed, see {args.outdir / REPORT_FILENAME}")  # noqa: T201
    if len(failed) > args.max_failure_fraction * len(data):
        msg = f"More than {args.max_failure_fraction:.0%} of the structures failed."
        raise SystemExit(msg)


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
        help="Path to clustering.json (from clusterv2.py).",
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
        "--max-failure-fraction",
        type=float,
        default=0.05,
        help="Abort (after writing the manifest) when more files than this fail.",
    )
    parser.add_argument(
        "--max-file-size",
        type=int,
        default=None,
        help="Skip .cif files larger than this many bytes.",
    )
    args = parser.parse_args()
    process(args)
