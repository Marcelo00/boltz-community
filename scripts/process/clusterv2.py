#!/usr/bin/env python3
"""Build the nucleic-acid ``clustering.json`` consumed by the Boltz processor.

This is the production analogue of Boltz's ``scripts/process/cluster.py``, but
focused on nucleic acids: RNA and DNA are actually clustered with mmseqs (the
upstream script gives every unique NA sequence its own id), while proteins are
out of scope.

It reads every ``*.cif`` under a dataset root, parses each with the StructureV2
``parse_mmcif`` (``boltz.data.parse.mmcif``), and from each structure collects:

  * RNA chains and DNA chains (classified directly from the gemmi
    ``PolymerType`` via ``mol_type`` -- no ``U``/``T`` heuristic), using
    ``parsed.sequences[chain]`` (the ``gemmi.one_letter_code`` string), and
  * ligand CCD codes from the non-polymer chains.

RNA and DNA are clustered **separately** with ``mmseqs easy-cluster`` and the
output is keyed exactly like ``cluster.py`` so the downstream record builder can
look chains up by sequence::

    { hash_sequence(seq): representative_hash, ..., ccd_code: ccd_code }

where ``hash_sequence(seq) = sha256(seq).hexdigest()``. Because this producer
and the downstream builder both parse the same CIFs with the same
``parse_mmcif`` and hash the same ``gemmi.one_letter_code`` string, the keys
line up by construction.

Run::

    python clusterv2.py \\
        --cif_path /path/to/dataset_root \\
        --moldir ~/.boltz/mols \\
        --output clustering.json \\
        [--identity 0.9] [--coverage 0.8] [--short_len 10]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from boltz.data import const
from boltz.data.parse.mmcif import parse_mmcif

RNA_ID = const.chain_type_ids["RNA"]
DNA_ID = const.chain_type_ids["DNA"]
LIG_ID = const.chain_type_ids["NONPOLYMER"]

# Sequences shorter than this bypass mmseqs (each unique one clustered alone).
# Feeding sub-k sequences to mmseqs can crash the prefilter ("No k-mer could be
# extracted"); a surviving short sequence would be a singleton anyway, so this
# is semantically free and only removes crash risk.
DEFAULT_SHORT_LEN = 10
DEFAULT_IDENTITY = 0.9
DEFAULT_COVERAGE = 0.8


def hash_sequence(seq: str) -> str:
    """SHA-256 hex digest, matching Boltz ``scripts/process/cluster.py``."""
    return hashlib.sha256(seq.encode()).hexdigest()


def discover_cifs(root: Path) -> list[Path]:
    """Return the CIF files to process.

    Accepts either a single ``.cif`` file, a single system folder, or a dataset
    root; ``*.cif`` files are collected recursively.
    """
    root = root.resolve()
    if root.is_file() and root.suffix == ".cif":
        return [root]
    return sorted(p for p in root.rglob("*.cif") if p.is_file())


def collect_from_cif(
    cif: Path,
    moldir: str,
    rna_seqs: set[str],
    dna_seqs: set[str],
    ligand_codes: set[str],
) -> None:
    """Parse one CIF and add its NA sequences and ligand CCD codes to the sets.

    Parses identically to the downstream record builder (same ``moldir`` and
    ``use_assembly=False``) so the hashed sequences match.
    """
    parsed = parse_mmcif(
        str(cif),
        moldir=moldir,
        call_compute_interfaces=False,
        use_assembly=False,
    )
    structure = parsed.data
    sequences = parsed.sequences  # keyed by chain NAME

    for chain in structure.chains:
        mol_type = int(chain["mol_type"])
        name = str(chain["name"])

        if mol_type == RNA_ID:
            seq = sequences.get(name)
            if seq:
                rna_seqs.add(seq)
        elif mol_type == DNA_ID:
            seq = sequences.get(name)
            if seq:
                dna_seqs.add(seq)
        elif mol_type == LIG_ID:
            start = int(chain["res_idx"])
            end = start + int(chain["res_num"])
            for i in range(start, end):
                code = str(structure.residues[i]["name"]).strip()
                if code:
                    ligand_codes.add(code)


def run_mmseqs(
    sequences: set[str],
    prefix: str,
    identity: float,
    coverage: float,
    mmseqs: str,
    tmp: Path,
) -> dict[str, str]:
    """Cluster ``sequences`` with mmseqs; return ``{member_hash: rep_hash}``.

    The FASTA header of each sequence is ``hash_sequence(seq)`` (its stable id),
    so the ``_cluster.tsv`` comes back already expressed as those hashes -- no
    separate header->sequence bookkeeping needed. Nucleotide mode is
    auto-detected by ``createdb`` from the alphabet, so no ``--search-type`` is
    passed; ``X`` (modified/unknown residue) is mapped to ``N`` on the mmseqs
    input only, while the hash key still uses the exact ``gemmi.one_letter_code``
    string.
    """
    if not sequences:
        return {}

    seq_to_hash = {seq: hash_sequence(seq) for seq in sorted(sequences)}
    fasta = tmp / f"{prefix}.fasta"
    fasta.write_text(
        "".join(
            f">{h}\n{seq.replace('X', 'N')}\n" for seq, h in seq_to_hash.items()
        )
    )

    result_prefix = tmp / prefix
    cmd = [
        mmseqs, "easy-cluster",
        str(fasta), str(result_prefix), str(tmp / f"{prefix}_mmtmp"),
        "--min-seq-id", str(identity),
        "-c", str(coverage),
        "--cov-mode", "0",          # both sequences must meet coverage
        "--spaced-kmer-mode", "0",  # contiguous seeds (reach short NA)
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"mmseqs binary {mmseqs!r} not found on PATH; install it or pass "
            "--mmseqs."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"mmseqs easy-cluster failed for {prefix!r}:\n{exc.stderr}"
        ) from exc

    tsv = result_prefix.with_name(f"{prefix}_cluster.tsv")
    member_to_rep: dict[str, str] = {}
    for line in tsv.read_text().splitlines():
        if not line:
            continue
        rep_id, member_id = line.split("\t")
        member_to_rep[member_id] = rep_id

    # Any sequence mmseqs did not emit is its own representative.
    return {h: member_to_rep.get(h, h) for h in seq_to_hash.values()}


def cluster_na(
    sequences: set[str],
    prefix: str,
    identity: float,
    coverage: float,
    mmseqs: str,
    tmp: Path,
    short_len: int,
) -> dict[str, str]:
    """Cluster one nucleic-acid type; return ``{seq_hash: cluster_label}``."""
    if not sequences:
        return {}

    long_seqs = {s for s in sequences if len(s) >= short_len}
    short_seqs = sequences - long_seqs

    clustering = run_mmseqs(long_seqs, prefix, identity, coverage, mmseqs, tmp)
    for seq in short_seqs:
        h = hash_sequence(seq)
        clustering[h] = h  # each short sequence is its own cluster
    return clustering


def cluster_ligands(ligand_codes: set[str]) -> dict[str, str]:
    """Cluster ligands by CCD-code identity; return ``{ccd: ccd}`` (lowercased)."""
    return {code.lower(): code.lower() for code in ligand_codes}


def build_clusters(
    rna_seqs: set[str],
    dna_seqs: set[str],
    ligand_codes: set[str],
    identity: float,
    coverage: float,
    mmseqs: str,
    tmp: Path,
    short_len: int,
) -> dict[str, str]:
    """Assemble the full ``{key: label}`` map (hash keys for NA, CCD for lig)."""
    clusters: dict[str, str] = {}
    clusters.update(
        cluster_na(rna_seqs, "rna", identity, coverage, mmseqs, tmp, short_len)
    )
    clusters.update(
        cluster_na(dna_seqs, "dna", identity, coverage, mmseqs, tmp, short_len)
    )
    clusters.update(cluster_ligands(ligand_codes))
    return clusters


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the nucleic-acid clustering.json for Boltz processing."
    )
    parser.add_argument(
        "--cif_path",
        type=Path,
        required=True,
        help="A .cif file, a system folder, or a dataset root (searched for *.cif).",
    )
    parser.add_argument(
        "--moldir",
        type=Path,
        required=True,
        help="Boltz CCD molecules directory (<code>.pkl); needed by parse_mmcif.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("clustering.json"),
        help="Destination JSON path (default: clustering.json).",
    )
    parser.add_argument(
        "--identity",
        type=float,
        default=DEFAULT_IDENTITY,
        help="mmseqs --min-seq-id for NA clustering (default %(default)s).",
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=DEFAULT_COVERAGE,
        help="mmseqs -c length coverage for NA clustering (default %(default)s).",
    )
    parser.add_argument(
        "--short_len",
        type=int,
        default=DEFAULT_SHORT_LEN,
        help="NA sequences shorter than this bypass mmseqs (default %(default)s).",
    )
    parser.add_argument(
        "--mmseqs",
        type=str,
        default="mmseqs",
        help="Path to the mmseqs binary (default: mmseqs).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N CIFs (for testing).",
    )
    args = parser.parse_args(argv)

    if not args.moldir.is_dir():
        print(
            f"[error] --moldir {args.moldir} is not a directory.",
            file=sys.stderr,
        )
        return 1

    cifs = discover_cifs(args.cif_path)
    if args.limit is not None:
        cifs = cifs[: args.limit]
    if not cifs:
        print(f"[error] no .cif files found under {args.cif_path}.", file=sys.stderr)
        return 1
    print(f"[info] parsing {len(cifs)} CIF(s) from {args.cif_path} ...")

    moldir = str(args.moldir)
    rna_seqs: set[str] = set()
    dna_seqs: set[str] = set()
    ligand_codes: set[str] = set()
    n_failed = 0
    for i, cif in enumerate(cifs, start=1):
        try:
            collect_from_cif(cif, moldir, rna_seqs, dna_seqs, ligand_codes)
        except Exception as exc:  # noqa: BLE001
            n_failed += 1
            print(f"[warn] {cif.name}: parse failed: {exc}", file=sys.stderr)
        if i % 100 == 0 or i == len(cifs):
            print(f"[info] parsed {i}/{len(cifs)} CIFs")

    if not (rna_seqs or dna_seqs or ligand_codes):
        print("[error] no sequences or ligands collected; nothing to cluster.",
              file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as td:
        clusters = build_clusters(
            rna_seqs,
            dna_seqs,
            ligand_codes,
            identity=args.identity,
            coverage=args.coverage,
            mmseqs=args.mmseqs,
            tmp=Path(td),
            short_len=args.short_len,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(clusters, indent=2, sort_keys=True))

    n_labels = len(set(clusters.values()))
    print(
        f"[done] keys={len(clusters)} "
        f"(rna_seq={len(rna_seqs)} dna_seq={len(dna_seqs)} lig={len(ligand_codes)}) "
        f"clusters={n_labels} cif_failed={n_failed} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
