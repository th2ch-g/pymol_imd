"""Headless smoke + numerical-regression test for the ML/MM (ONIOM) backend.

Runs without PyMOL: builds the MLMM system directly, computes the per-frame
ONIOM delta force ``F_ML - F_MM_subset`` on the ML region at the *initial*
geometry, and checks it is finite. With a saved reference it also asserts the
delta is unchanged to within a tolerance — this is the guard that the Tier-1
lossless refactors (ase.Atoms reuse, index-map precompute, ...) do not alter
numerics.

The device is pinned to CPU so the reference is reproducible across machines
(MPS/CUDA would give bitwise-different results from CPU). A small ML selection
that straddles a residue boundary is used so the link-atom path is exercised
too.

Usage::

    pixi run smoke-mlmm                       # compare vs saved ref (or create it)
    pixi run smoke-mlmm 1ubq                  # override pdb id
    pixi run smoke-mlmm 1aki aimnet2 "resi 1-3"   # pdb / backend / selection
    SMOKE_MLMM_SAVE_REF=1 pixi run smoke-mlmm     # (re)write the reference
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from openmm import unit

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("smoke_mlmm")

# Reference deltas live next to this file; one per backend (they differ
# because each MLIP predicts its own forces).
REF_DIR = Path(__file__).parent / "refs"

# float32-native backends (aimnet2/orb) carry ~1e-5 numerical noise; MACE runs
# float64 by default so it can be held to a tighter bound.
_ATOL_BY_BACKEND = {"mace": 1e-7, "aimnet2": 1e-4, "orb": 1e-4, "uma": 1e-4}


def _compute_ml_delta(imd) -> np.ndarray:
    """Run one ONIOM correction and read the per-particle delta force back.

    The values written into ``_mlmm_force`` are exactly ``F_ML - F_MM_subset``
    (plus redistributed link-atom force on the boundary slots), so reading the
    per-particle parameters is the cleanest probe of the hot path's output.
    """
    imd._apply_mlmm_correction_force()
    force = imd._mlmm_force
    n = force.getNumParticles()
    out = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        _idx, params = force.getParticleParameters(i)
        out[i] = [float(p) for p in params]
    return out


def main(pdb_id: str = "1aki", backend: str = "aimnet2", ml_selection: str = "resi 1-3") -> int:
    from pymol_imd.backends.mlmm import MLMMBackend
    from pymol_imd.imd import IMDConfig, IMDSystem
    from pymol_imd.prep import fetch_pdb

    with tempfile.TemporaryDirectory() as tmp:
        pdb_path = fetch_pdb(pdb_id, out_dir=Path(tmp))
        log.info("Building MLMM system: pdb=%s backend=%s sel=%r", pdb_id, backend, ml_selection)
        bundle = MLMMBackend().build_system(
            pdb_path,
            ml_selection=ml_selection,
            ml_backend=backend,
            ml_device="cpu",  # pin so the reference is machine-independent
            ml_charge=0,
            ml_mult=1,
        )
        extra = bundle.extra or {}
        n_ml = len(extra.get("ml_indices", ()))
        n_link = len(extra.get("link_atoms", ()))
        log.info(
            "System: %d atoms, ML region=%d atoms, link atoms=%d",
            bundle.topology.getNumAtoms(),
            n_ml,
            n_link,
        )

        cfg = IMDConfig(nsteps_per_frame=1, mouse_force=False, timestep_ps=0.0005)
        imd = IMDSystem(bundle, cfg)

        # OpenMM's Modeller.addHydrogens places some rotatable H atoms
        # stochastically, so build_system positions vary run-to-run even though
        # the topology (atom count + order) is deterministic. Pin coordinates to
        # a saved reference so the delta-force regression is reproducible —
        # AIMNet2 is bitwise-deterministic at fixed coordinates (verified).
        REF_DIR.mkdir(parents=True, exist_ok=True)
        pos_ref = REF_DIR / "mlmm_init_positions.npy"
        delta_ref = REF_DIR / f"mlmm_{backend}_delta.npy"
        save_ref = (
            bool(os.environ.get("SMOKE_MLMM_SAVE_REF"))
            or not pos_ref.exists()
            or not delta_ref.exists()
        )
        if not save_ref:
            fixed = np.load(pos_ref)
            if fixed.shape[0] != imd.n_atoms:
                raise AssertionError(
                    f"position ref atom count {fixed.shape[0]} != system {imd.n_atoms}; "
                    "re-save with SMOKE_MLMM_SAVE_REF=1 if intentional."
                )
            imd.simulation.context.setPositions(fixed * unit.nanometer)
            log.info("init positions pinned from %s", pos_ref)

        delta = _compute_ml_delta(imd)
        if not np.all(np.isfinite(delta)):
            raise AssertionError("non-finite ML delta force at the initial geometry")
        fmax = float(np.max(np.linalg.norm(delta, axis=1)))
        log.info("ML delta force finite at t0: |F|max=%.4f kJ/mol/nm", fmax)

        # --- numerical-regression check against the saved reference ----------
        if save_ref:
            np.save(pos_ref, imd._read_openmm_positions_nm())
            np.save(delta_ref, delta)
            log.info("baseline saved: init positions + delta (shape=%s)", delta.shape)
        else:
            ref = np.load(delta_ref)
            if ref.shape != delta.shape:
                raise AssertionError(
                    f"ML delta shape changed vs ref: {delta.shape} != {ref.shape}. "
                    "Re-create the reference with SMOKE_MLMM_SAVE_REF=1 if intentional."
                )
            atol = _ATOL_BY_BACKEND.get(backend, 1e-4)
            max_dev = float(np.max(np.abs(delta - ref)))
            if max_dev > atol:
                raise AssertionError(
                    f"ML delta force CHANGED vs reference: max|Δ|={max_dev:.3e} > atol={atol:.0e}. "
                    "A lossless refactor must not move these numbers."
                )
            log.info("numerics match reference: max|Δ|=%.3e (atol=%.0e)", max_dev, atol)

        # --- a few real frames must not blow up ------------------------------
        log.info("Stepping 5 frames...")
        imd.running = True
        for i in range(5):
            imd.step()
            if not imd.running:
                raise AssertionError(f"diverged at frame {i}")
        delta_after = _compute_ml_delta(imd)
        if not np.all(np.isfinite(delta_after)):
            raise AssertionError("non-finite ML delta force after stepping")
        log.info("Smoke MLMM OK (backend=%s)", backend)
    return 0


if __name__ == "__main__":
    pdb = sys.argv[1] if len(sys.argv) > 1 else "1aki"
    ml_backend = sys.argv[2] if len(sys.argv) > 2 else "aimnet2"
    sel = sys.argv[3] if len(sys.argv) > 3 else "resi 1-3"
    raise SystemExit(main(pdb, ml_backend, sel))
