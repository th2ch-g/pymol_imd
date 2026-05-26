"""Headless smoke test: download a tiny PDB, build AA system, minimize, step.

Runs without PyMOL: imports the backend + IMDSystem directly. The PyMOL ↔
OpenMM coordinate sync is bypassed because there is no PyMOL window to read
from / write back to, so we only exercise the OpenMM half of the loop.

Usage::

    pixi run smoke
    pixi run smoke 1ubq      # override PDB id
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("smoke")


def main(pdb_id: str = "1aki", n_steps: int = 10) -> int:
    from pymol_imd.backends.aa import AABackend
    from pymol_imd.imd import IMDConfig, IMDSystem
    from pymol_imd.prep import fetch_pdb

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pdb_path = fetch_pdb(pdb_id, out_dir=tmp_path)
        log.info("Building AA system from %s", pdb_path)
        bundle = AABackend().build_system(pdb_path, ligand_sdf=None, pbc=False)
        log.info("System built: %d atoms (pbc=%s)", bundle.topology.getNumAtoms(), bundle.pbc)

        cfg = IMDConfig(nsteps_per_frame=1, mouse_force=False)
        imd = IMDSystem(bundle, cfg)

        log.info("Running minimization chunk...")
        imd.simulation.minimizeEnergy(maxIterations=200)
        log.info("Stepping %d MD frames...", n_steps)
        for _ in range(n_steps):
            imd.simulation.step(1)
        log.info("Smoke test OK")
    return 0


if __name__ == "__main__":
    pdb_id = sys.argv[1] if len(sys.argv) > 1 else "1aki"
    raise SystemExit(main(pdb_id))
