"""Carry OpenMM particle masses in prepared PDBs without changing IMDv3."""

from pathlib import Path

import numpy as np
from openmm import unit
from openmm.app import Element, PDBFile

_PREFIX = "REMARK 950 PYMOL_IMD_MASS "


def system_masses(system) -> np.ndarray:
    return np.array(
        [
            system.getParticleMass(i).value_in_unit(unit.dalton)
            for i in range(system.getNumParticles())
        ]
    )


def validate_masses(masses, n_atoms: int) -> np.ndarray:
    values = np.asarray(masses, dtype=float)
    if values.shape != (n_atoms,) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(
            "Particle masses must be finite, nonnegative and match prepared atom order"
        )
    return values


def write_particle_masses(path: Path, masses: np.ndarray) -> None:
    """Store masses in daltons by prepared atom index, including massless sites."""
    lines = path.read_text().splitlines(keepends=True)
    count = sum(line.startswith(("ATOM  ", "HETATM")) for line in lines)
    values = validate_masses(masses, count)
    records = [f"{_PREFIX}{i} {mass:.17g}\n" for i, mass in enumerate(values)]
    path.write_text("".join(records + [line for line in lines if not line.startswith(_PREFIX)]))


def element_masses(symbols) -> np.ndarray:
    return np.array(
        [Element.getBySymbol(symbol).mass.value_in_unit(unit.dalton) for symbol in symbols]
    )


def read_particle_masses(path: Path) -> np.ndarray | None:
    """Read exact masses, or elemental masses for older all-atom prepared PDBs."""
    lines = path.read_text().splitlines()
    atoms = [line for line in lines if line.startswith(("ATOM  ", "HETATM"))]
    records = [line[len(_PREFIX) :].split() for line in lines if line.startswith(_PREFIX)]
    if records:
        if len(records) != len(atoms) or any(int(row[0]) != i for i, row in enumerate(records)):
            raise ValueError("Prepared PDB particle mass records do not match atom order")
        return validate_masses([float(row[1]) for row in records], len(atoms))
    if any(line[12:16].strip() == "BB" for line in atoms):
        return None
    topology = PDBFile(str(path)).topology
    return validate_masses(
        [
            atom.element.mass.value_in_unit(unit.dalton) if atom.element else 0
            for atom in topology.atoms()
        ],
        len(atoms),
    )
