"""Coarse-grained Martini backend.

The pipeline is two-step::

    AA PDB --(vermouth / martinize2)--> CG .top + .itp + .gro
           --(martini_openmm)----------> OpenMM System / Topology / Positions

Most of the heavy lifting is delegated to upstream tools; this module wires
them together and produces a :class:`SystemBundle` that the IMD driver can
consume identically to the AA path.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from openmm import unit
from openmm.app import GromacsGroFile

from ..prep import write_prepared_pdb
from .base import SystemBundle

logger = logging.getLogger(__name__)

MARTINI_FF = "martini30"  # default forcefield label fed to martinize2 (--ff)
DEFAULT_DSSP_BIN = "mkdssp"


class CGBackend:
    name = "cg"

    def build_system(
        self,
        pdb_path: Path,
        *,
        ligand_sdf: Path | None = None,
        pbc: bool = False,
        ff: str = MARTINI_FF,
        elastic_network: bool = True,
        work_dir: Path | None = None,
        **_: object,
    ) -> SystemBundle:
        if ligand_sdf is not None:
            raise NotImplementedError(
                "CG backend does not yet support custom ligand SDFs; "
                "the Martini ligand-mapping pipeline is not wired up. "
                "Open an issue with the ligand SMILES if needed."
            )

        if shutil.which("martinize2") is None:
            raise RuntimeError(
                "martinize2 not found in PATH. Activate the pixi env first "
                "(`pixi shell`) — vermouth provides this binary."
            )

        work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="pymol_imd_cg_"))
        work_dir.mkdir(parents=True, exist_ok=True)
        cg_gro = work_dir / "cg.gro"
        cg_top = work_dir / "cg.top"

        cmd = [
            "martinize2",
            "-f",
            str(pdb_path),
            "-x",
            str(cg_gro),
            "-o",
            str(cg_top),
            "-ff",
            ff,
            "-dssp",
            DEFAULT_DSSP_BIN,
        ]
        if elastic_network:
            cmd += ["-elastic"]
        logger.info("Running %s", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=work_dir)

        gro = GromacsGroFile(str(cg_gro))
        # martini_openmm provides a drop-in replacement for GromacsTopFile that
        # understands Martini non-standard directives.
        from martini_openmm import MartiniTopFile  # type: ignore

        top = MartiniTopFile(
            str(cg_top),
            periodicBoxVectors=gro.getPeriodicBoxVectors() if pbc else None,
        )
        system = top.createSystem(nonbondedCutoff=1.1 * unit.nanometer)

        prepared_pdb = write_prepared_pdb(
            top.topology,
            gro.positions,
            work_dir / "cg_prepared.pdb",
        )

        return SystemBundle(
            system=system,
            topology=top.topology,
            positions=gro.positions,
            backend_name=self.name,
            prepared_pdb_path=prepared_pdb,
            pbc=pbc,
        )
