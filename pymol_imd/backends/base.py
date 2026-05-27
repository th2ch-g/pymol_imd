"""Common backend interface shared by AA / CG / ML-MM.

A backend is responsible for turning a user-supplied structure (PDB, optional
ligand SDF, optional CG / ML-MM specific flags) into an OpenMM ``System``
plus matching ``Topology`` and initial ``Positions``. The IMD driver layer
attaches the interactive pulling force and runs the simulation, so backends
never need to know about PyMOL.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import openmm
from openmm.app import Topology
from openmm.unit import Quantity


@dataclass(frozen=True)
class SystemBundle:
    """Output of ``Backend.build_system``.

    Attributes:
        system: Parameterized OpenMM ``System`` (without the IMD pulling force;
            the driver layer attaches that).
        topology: Matching topology — its atom order MUST line up with PyMOL.
        positions: Initial atomic positions as an OpenMM ``Quantity``.
        backend_name: Short label ("aa", "cg", "mlmm") used for logging.
        pbc: Whether the system was set up with periodic boundary conditions.
        prepared_pdb_path: A PDB file written out *after* hydrogen completion
            / ligand merge / CG mapping so PyMOL can be re-loaded against the
            same atom order the OpenMM topology uses. Critical for IMD: if
            PyMOL has the original PDB while OpenMM has hydrogens added, the
            ``index - 1`` lookup in the sync step writes coordinates to the
            wrong atoms and the structure visually collapses.
        extra: Optional backend-specific dict for data the IMD driver may need
            (e.g., ML region atom indices + MLIP backend for ML/MM). The base
            driver ignores it; downstream backends opt in.
    """

    system: openmm.System
    topology: Topology
    positions: Quantity
    backend_name: str
    prepared_pdb_path: Path
    pbc: bool = False
    extra: dict | None = None


class Backend(Protocol):
    """Structural interface for simulation backends.

    Concrete backends live in this package (``aa.py``, ``cg.py``, ``mlmm.py``).
    They are intentionally duck-typed: any object exposing ``build_system``
    with the right signature is a valid backend.

    Each backend also exposes ``recommended_const_k`` so the command layer
    can pick a sensible default pulling-force scale per backend (heavier CG
    beads need a smaller ``const_k`` than fast AA atoms) without spreading
    those numbers across the codebase.

    ``ligand_sdf`` is **AA-only**; CG and ML/MM backends raise if it is
    passed. Backend-specific kwargs (``ff`` / ``elastic_network`` for CG,
    ``ml_selection`` / ``ml_backend`` for ML/MM) flow in via ``**kwargs``.

    ``prepared_pdb_path`` in the returned bundle is *backend-defined*:
    AA / ML-MM write it via ``prep.write_prepared_pdb``, CG copies
    martinize2's ``cg.pdb`` verbatim because martinize already emits
    accurate CONECT records covering chemical bonds + constraints.
    """

    name: str
    recommended_const_k: float

    def build_system(
        self,
        pdb_path: Path,
        *,
        pbc: bool = False,
        **kwargs: object,
    ) -> SystemBundle:
        """Construct the OpenMM ``System`` for the given inputs."""
        ...
