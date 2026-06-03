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
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("smoke")


def _verify_force_cap(imd, *, label: str) -> None:
    """Inject a fake 1 nm delta on atom 0 and confirm |F| is capped at max_force_kj.

    With const_k=10000 and max_force_kj=5000 (defaults), nominal |F| = 10000
    kJ/mol/nm and the cap should clamp it to 5000.
    """
    nominal_fmag = 1.0 * imd.config.const_k  # delta=1 nm * const_k
    cap = imd.config.max_force_kj
    if nominal_fmag <= cap:
        log.info(
            "[%s] const_k=%.0f below cap=%.0f; cap check would be vacuous, skipping",
            label,
            imd.config.const_k,
            cap,
        )
        return

    def _fake_pymol_positions() -> np.ndarray:
        pos = imd._read_openmm_positions_nm().copy()
        pos[0] = pos[0] + np.array([1.0, 0.0, 0.0])  # 1 nm shove on atom 0
        return pos

    original = imd._read_pymol_positions_nm
    imd._read_pymol_positions_nm = _fake_pymol_positions
    try:
        imd._apply_pull_force_from_pymol()
    finally:
        imd._read_pymol_positions_nm = original

    _, params = imd._pull_force.getParticleParameters(0)
    fmag = math.sqrt(sum(float(p) ** 2 for p in params))
    if abs(fmag - cap) > 1.0:
        raise AssertionError(
            f"[{label}] |F| cap not enforced: got {fmag:.2f} kJ/mol/nm, expected {cap:.2f}"
        )
    log.info("[%s] |F| cap OK: nominal=%.0f -> capped=%.1f kJ/mol/nm", label, nominal_fmag, fmag)


def _run_backend(
    backend,
    pdb_path: Path,
    *,
    label: str,
    n_steps: int,
    integrator: str = "langevin",
    check_force_cap: bool = False,
) -> None:
    from pymol_imd.imd import IMDConfig, IMDSystem

    log.info("[%s] Building system from %s", label, pdb_path)
    bundle = backend.build_system(pdb_path)
    log.info(
        "[%s] System built: %d atoms (pbc=%s)", label, bundle.topology.getNumAtoms(), bundle.pbc
    )

    cfg = IMDConfig(nsteps_per_frame=1, mouse_force=False, integrator=integrator)
    imd = IMDSystem(bundle, cfg)
    log.info("[%s] Integrator: %s", label, integrator)
    log.info("[%s] Minimizing...", label)
    imd.simulation.minimizeEnergy(maxIterations=200)

    if check_force_cap:
        _verify_force_cap(imd, label=label)

    log.info("[%s] Stepping %d frames...", label, n_steps)
    for _ in range(n_steps):
        imd.simulation.step(1)
        if imd._diverged():
            raise AssertionError(f"[{label}] diverged after a single 1-step block")
    log.info("[%s] OK", label)


def _run_pbc_build(backend, pdb_path: Path, *, label: str) -> None:
    """Build with ``pbc=True`` and take a couple of NPT steps.

    Covers the ``pbc=True`` path the rest of the suite skips. A box-vector
    format bug once crashed ``PDBFile.writeHeader`` here on every backend, so
    this asserts the periodic box is set, the System uses PBC, the prepared PDB
    was actually written (writeHeader survived), a ``MonteCarloBarostat`` was
    added for NPT, and the system does not immediately diverge.
    """
    from openmm import MonteCarloBarostat

    from pymol_imd.imd import IMDConfig, IMDSystem

    log.info("[%s] Building pbc=True system...", label)
    bundle = backend.build_system(pdb_path, pbc=True)
    if bundle.topology.getPeriodicBoxVectors() is None:
        raise AssertionError(f"[{label}] pbc=True but no periodic box vectors set")
    if not bundle.system.usesPeriodicBoundaryConditions():
        raise AssertionError(f"[{label}] pbc=True but System reports no PBC")
    if not Path(bundle.prepared_pdb_path).exists():
        raise AssertionError(f"[{label}] prepared PDB not written under pbc=True")

    cfg = IMDConfig(nsteps_per_frame=1, mouse_force=False, integrator="langevin_middle")
    imd = IMDSystem(bundle, cfg)
    has_barostat = any(
        isinstance(bundle.system.getForce(i), MonteCarloBarostat)
        for i in range(bundle.system.getNumForces())
    )
    if not has_barostat:
        raise AssertionError(f"[{label}] pbc=True but no MonteCarloBarostat (NPT) added")
    imd.simulation.minimizeEnergy(maxIterations=50)
    imd.simulation.step(2)
    if imd._diverged():
        raise AssertionError(f"[{label}] pbc=True diverged after 2 steps")
    log.info("[%s] pbc=True OK: box set, prepared PDB written, NPT barostat, steps", label)


def main(pdb_id: str = "1aki", n_steps: int = 10) -> int:
    from pymol_imd.backends.aa import AABackend
    from pymol_imd.backends.cg import CGBackend
    from pymol_imd.prep import fetch_pdb

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pdb_path = fetch_pdb(pdb_id, out_dir=tmp_path)
        # AA runs with LangevinMiddleIntegrator (matches imd_setup_aa default) and
        # also exercises the |F| cap path under a synthetic large delta.
        _run_backend(
            AABackend(),
            pdb_path,
            label="AA",
            n_steps=n_steps,
            integrator="langevin_middle",
            check_force_cap=True,
        )
        _run_backend(CGBackend(), pdb_path, label="CG", n_steps=n_steps, integrator="langevin")
        # pbc=True regression: this path was uncovered and once crashed every
        # backend at prepared-PDB write. AA exercises it (CG always builds a box).
        _run_pbc_build(AABackend(), pdb_path, label="AA-pbc")
        log.info("Smoke test OK")
    return 0


if __name__ == "__main__":
    pdb_id = sys.argv[1] if len(sys.argv) > 1 else "1aki"
    raise SystemExit(main(pdb_id))
