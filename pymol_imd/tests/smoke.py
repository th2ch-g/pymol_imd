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


def _verify_force_limit(imd, *, label: str) -> None:
    """Confirm an over-limit drag skips the whole frame; a sub-limit one applies.

    With const_k=10000 and max_force_kj=5000 (defaults), a 1 nm shove gives a
    nominal |F| = 10000 kJ/mol/nm > the 5000 limit, so the whole frame's pull is
    skipped and atom 0's force must be exactly 0 (the skip-and-warn behavior that
    replaced the old clamp). A sub-limit shove is then applied to confirm a
    within-limit pull still goes through, unclamped.
    """
    cap = imd.config.max_force_kj
    over_fmag = 1.0 * imd.config.const_k  # delta=1 nm * const_k
    if over_fmag <= cap:
        log.info(
            "[%s] const_k=%.0f keeps a 1 nm drag within limit=%.0f; skip check vacuous",
            label,
            imd.config.const_k,
            cap,
        )
        return

    def _shoved_positions(atom: int, dist_nm: float):
        def _fake() -> np.ndarray:
            pos = imd._read_openmm_positions_nm().copy()
            pos[atom] = pos[atom] + np.array([dist_nm, 0.0, 0.0])
            return pos

        return _fake

    def _atom0_fmag() -> float:
        _, params = imd._pull_force.getParticleParameters(0)
        return math.sqrt(sum(float(p) ** 2 for p in params))

    original = imd._read_pymol_positions_nm
    try:
        # (1) Over-limit 1 nm shove -> whole frame skipped -> atom 0 force == 0.
        imd._read_pymol_positions_nm = _shoved_positions(0, 1.0)
        imd._apply_pull_force_from_pymol()
        fmag = _atom0_fmag()
        if fmag > 1e-6:
            raise AssertionError(
                f"[{label}] over-limit drag not skipped: |F|={fmag:.2f} kJ/mol/nm, expected 0"
            )

        # (2) Sub-limit shove -> applied unclamped (no magnitude cap anymore).
        sub_dist = 0.25 * cap / imd.config.const_k
        expected = sub_dist * imd.config.const_k
        imd._read_pymol_positions_nm = _shoved_positions(0, sub_dist)
        imd._apply_pull_force_from_pymol()
        fmag = _atom0_fmag()
        if abs(fmag - expected) > 1.0:
            raise AssertionError(
                f"[{label}] sub-limit drag not applied: |F|={fmag:.2f}, expected {expected:.2f}"
            )
    finally:
        imd._read_pymol_positions_nm = original
    log.info(
        "[%s] force-limit OK: over-limit (1 nm) skipped -> 0; sub-limit applied -> %.1f kJ/mol/nm",
        label,
        fmag,
    )


def _run_backend(
    backend,
    pdb_path: Path,
    *,
    label: str,
    n_steps: int,
    integrator: str = "langevin",
    check_force_limit: bool = False,
    build_kwargs: dict | None = None,
) -> None:
    from pymol_imd.imd import IMDConfig, IMDSystem

    log.info("[%s] Building system from %s", label, pdb_path)
    bundle = backend.build_system(pdb_path, **(build_kwargs or {}))
    log.info(
        "[%s] System built: %d atoms (pbc=%s)", label, bundle.topology.getNumAtoms(), bundle.pbc
    )

    cfg = IMDConfig(nsteps_per_frame=1, mouse_force=False, integrator=integrator)
    imd = IMDSystem(bundle, cfg)
    log.info("[%s] Integrator: %s", label, integrator)
    log.info("[%s] Minimizing...", label)
    imd.simulation.minimizeEnergy(maxIterations=200)

    if check_force_limit:
        _verify_force_limit(imd, label=label)

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


def _check_implicit_solvent_resolver() -> None:
    """Headless unit check of the implicit-solvent knob resolution (no OpenMM)."""
    from pymol_imd import constants as C

    cases = [
        (("auto", False), "implicit/gbn2.xml"),  # default -> GBn2 in a non-periodic run
        (("auto", True), None),  # pbc wins: GB yields to PME
        (("none", False), None),  # explicit vacuo
        (("none", True), None),
        (("obc2", False), "implicit/obc2.xml"),  # explicit model passthrough
    ]
    for (val, pbc), expected in cases:
        got = C.resolve_implicit_solvent(val, pbc=pbc)
        if got != expected:
            raise AssertionError(
                f"resolve_implicit_solvent({val!r}, pbc={pbc}) = {got!r}, expected {expected!r}"
            )
    # explicit GB + pbc (conflict) and an unknown model must both raise
    for val, pbc in (("gbn2", True), ("bogus", False)):
        try:
            C.resolve_implicit_solvent(val, pbc=pbc)
        except ValueError:
            pass
        else:
            raise AssertionError(f"resolve_implicit_solvent({val!r}, pbc={pbc}) should have raised")
    log.info("implicit-solvent resolver OK (auto/none/explicit/conflict)")


def main(pdb_id: str = "1aki", n_steps: int = 10) -> int:
    from pymol_imd.backends.aa import AABackend
    from pymol_imd.backends.cg import CGBackend
    from pymol_imd.prep import fetch_pdb

    _check_implicit_solvent_resolver()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pdb_path = fetch_pdb(pdb_id, out_dir=tmp_path)
        # AA runs with LangevinMiddleIntegrator (matches imd_setup_aa default) and
        # also exercises the force-limit skip path under a synthetic over-limit
        # delta. The default carries GBn2 implicit solvent (the AA / ML-MM default).
        _run_backend(
            AABackend(),
            pdb_path,
            label="AA(GBn2-default)",
            n_steps=n_steps,
            integrator="langevin_middle",
            check_force_limit=True,
        )
        # Vacuo opt-out: implicit_solvent="none" restores the old NoCutoff path.
        _run_backend(
            AABackend(),
            pdb_path,
            label="AA(vacuo)",
            n_steps=n_steps,
            integrator="langevin_middle",
            build_kwargs={"implicit_solvent": "none"},
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
