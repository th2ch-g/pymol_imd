"""Measure recorded QBP or chorismate geometry without driving a simulation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mdtraj
import numpy as np


def fit(moving, target):
    origin = moving.mean(axis=0)
    destination = target.mean(axis=0)
    u, _, vt = np.linalg.svd((moving - origin).T @ (target - destination))
    rotation = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
    return origin, destination, rotation


def rmsd(moving, target):
    origin, destination, rotation = fit(moving, target)
    return float(
        np.sqrt(np.mean(np.sum(((moving - origin) @ rotation + destination - target) ** 2, axis=1)))
    )


def atoms_by_residue(topology, name):
    atoms = [atom for atom in topology.atoms if atom.name == name]
    result = {atom.residue.resSeq: atom.index for atom in atoms}
    if len(result) != len(atoms):
        raise ValueError("QBP analysis requires one chain with unique residue numbers")
    return result


def qbp_measurements(initial, args):
    topology = initial.topology
    cg = any(atom.name == "BB" for atom in topology.atoms)
    indices = atoms_by_residue(topology, "BB" if cg else "CA")
    references = {}
    for label, filename in (("open", args.open_reference), ("closed", args.closed_reference)):
        reference = mdtraj.load_pdb(str(filename))
        mapping = atoms_by_residue(reference.topology, "CA")
        references[label] = {
            residue: reference.xyz[0, index] * 10 for residue, index in mapping.items()
        }
    common = sorted(set(indices) & set(references["open"]) & set(references["closed"]))
    if len(common) < 200:
        raise ValueError("Too few matching QBP backbone residues; verify numbering and topology")

    def domain_a(residue):
        return 5 <= residue <= 84 or 186 <= residue <= 224

    def domain_b(residue):
        return 90 <= residue <= 180

    domains = {"a": domain_a, "b": domain_b}
    backbone = [indices[residue] for residue in common]
    masks = {
        label: np.array([predicate(residue) for residue in common])
        for label, predicate in domains.items()
    }
    particles = {
        label: [
            atom.index
            for atom in topology.atoms
            if predicate(atom.residue.resSeq)
            and (atom.element is None or atom.element.symbol != "H")
        ]
        for label, predicate in domains.items()
    }
    reference_arrays = {
        label: np.array([reference[residue] for residue in common])
        for label, reference in references.items()
    }
    initial_backbone = initial.xyz[0, backbone].astype(float) * 10

    def measure(xyz):
        bb = xyz[backbone]
        values = {
            "distance_particles_A": float(
                np.linalg.norm(xyz[particles["a"]].mean(0) - xyz[particles["b"]].mean(0))
            ),
            "distance_backbone_A": float(
                np.linalg.norm(bb[masks["a"]].mean(0) - bb[masks["b"]].mean(0))
            ),
        }
        for label, reference in reference_arrays.items():
            values[f"rmsd_{label}_A"] = rmsd(bb, reference)
            origin, destination, rotation = fit(bb[masks["a"]], reference[masks["a"]])
            difference = (bb[masks["b"]] - origin) @ rotation + destination - reference[masks["b"]]
            values[f"domain_b_{label}_after_a_fit_A"] = float(
                np.sqrt(np.mean(np.sum(difference**2, axis=1)))
            )
        for label, mask in masks.items():
            values[f"domain_{label}_internal_rmsd_A"] = rmsd(bb[mask], initial_backbone[mask])
        return values

    metadata = {
        "comparison": "CG BB versus reference CA (approximate)"
        if cg
        else "AA CA versus reference CA",
        "particle_distance": "All CG beads (not the paper's heavy-atom distance)"
        if cg
        else "All non-hydrogen atoms",
        "matched_backbone_residues": len(common),
    }
    return measure, metadata


def chorismate_measurements(initial):
    residues = [residue for residue in initial.topology.residues if residue.name == "ISJ"]
    if len(residues) != 1:
        raise ValueError("Expected exactly one ISJ ligand")
    atoms = list(residues[0].atoms)
    indices = {atom.name: atom.index for atom in atoms}
    if len(indices) != len(atoms):
        raise ValueError("Ligand atom names must be unique")
    pairs = [
        ("C1", "O11"),
        ("C3", "C16"),
        ("C1", "C2"),
        ("C2", "C3"),
        ("C12", "C16"),
        ("O11", "C12"),
    ]
    missing = {name for pair in pairs for name in pair} - indices.keys()
    if missing:
        raise ValueError(f"Missing reactive atoms: {sorted(missing)}")

    def measure(xyz):
        return {
            f"{first}_{second}_A": float(np.linalg.norm(xyz[indices[first]] - xyz[indices[second]]))
            for first, second in pairs
        }

    return measure, {
        "ligand_atoms": len(atoms),
        "atom_indices_zero_based": indices,
        "interpretation": "Six geometric distances; bond orders and mechanism are not assigned",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system", choices=("qbp", "chorismate"))
    parser.add_argument(
        "--topology",
        required=True,
        type=Path,
        help="Prepared initial PDB matching trajectory atom order",
    )
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory for metrics.csv and summary.json",
    )
    parser.add_argument(
        "--open-reference", type=Path, default=Path("examples/qbp_open_reference.pdb")
    )
    parser.add_argument(
        "--closed-reference", type=Path, default=Path("examples/qbp_closed_reference.pdb")
    )
    args = parser.parse_args()
    initial = mdtraj.load_pdb(str(args.topology))
    measure, metadata = (
        qbp_measurements(initial, args)
        if args.system == "qbp"
        else chorismate_measurements(initial)
    )
    baseline = measure(initial.xyz[0].astype(float) * 10)
    if (args.output / "metrics.csv").exists() or (args.output / "summary.json").exists():
        parser.error("Output already exists; select a new directory")
    args.output.mkdir(parents=True, exist_ok=True)
    count = 0
    first = last = None
    minima = dict.fromkeys(baseline, float("inf"))
    maxima = dict.fromkeys(baseline, float("-inf"))
    with (args.output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["frame", "time_ps", *baseline])
        writer.writeheader()
        for chunk in mdtraj.iterload(str(args.trajectory), top=initial.topology, chunk=200):
            for xyz, time_ps in zip(chunk.xyz, chunk.time, strict=True):
                if not np.isfinite(xyz).all():
                    raise ValueError(f"Non-finite coordinates in frame {count}")
                values = measure(xyz.astype(float) * 10)
                row = {"frame": count, "time_ps": float(time_ps), **values}
                writer.writerow(row)
                if first is None:
                    first = row
                last = row
                for name, value in values.items():
                    minima[name] = min(minima[name], value)
                    maxima[name] = max(maxima[name], value)
                count += 1
    if count == 0:
        raise ValueError("Trajectory has no frames")
    summary = {
        "system": args.system,
        **metadata,
        "frames": count,
        "initial_pdb": baseline,
        "first": first,
        "last": last,
        "minima": minima,
        "maxima": maxima,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
