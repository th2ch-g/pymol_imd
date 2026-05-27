"""Headless smoke test: download a tiny PDB, build AA and CG systems, minimize, step.

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


def _run_backend(backend, pdb_path: Path, *, label: str, n_steps: int) -> None:
    from pymol_imd.imd import IMDConfig, IMDSystem

    log.info("[%s] Building system from %s", label, pdb_path)
    bundle = backend.build_system(pdb_path)
    log.info(
        "[%s] System built: %d atoms (pbc=%s)", label, bundle.topology.getNumAtoms(), bundle.pbc
    )

    imd = IMDSystem(bundle, IMDConfig(nsteps_per_frame=1, mouse_force=False))
    log.info("[%s] Minimizing...", label)
    imd.simulation.minimizeEnergy(maxIterations=200)
    log.info("[%s] Stepping %d frames...", label, n_steps)
    for _ in range(n_steps):
        imd.simulation.step(1)
    log.info("[%s] OK", label)


def main(pdb_id: str = "1aki", n_steps: int = 10) -> int:
    from pymol_imd.backends.aa import AABackend
    from pymol_imd.backends.cg import CGBackend
    from pymol_imd.prep import fetch_pdb

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pdb_path = fetch_pdb(pdb_id, out_dir=tmp_path)
        _run_backend(AABackend(), pdb_path, label="AA", n_steps=n_steps)
        _run_backend(CGBackend(), pdb_path, label="CG", n_steps=n_steps)
        log.info("Smoke test OK")
    return 0


if __name__ == "__main__":
    pdb_id = sys.argv[1] if len(sys.argv) > 1 else "1aki"
    raise SystemExit(main(pdb_id))
