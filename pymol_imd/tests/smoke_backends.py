"""Exercise a selected ML/MM backend and optionally compare CPU/CUDA forces."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import tempfile
import time
from pathlib import Path

import numpy as np

from pymol_imd import constants
from pymol_imd.backends.mlmm import MLMMBackend, _build_mlip_calculator
from pymol_imd.imd import IMDConfig, IMDSystem
from pymol_imd.tests.smoke_mlmm import _compute_ml_delta
from pymol_imd.tests.smoke_molecules import write_inputs

log = logging.getLogger("smoke-backends")

# Cross-device comparisons allow float32 reduction order differences.
# The saved CPU numerical-regression tolerances remain unchanged.
CROSS_DEVICE_ATOL_EV_A = 2e-4
CROSS_DEVICE_RTOL = 2e-5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=constants.SUPPORTED_ML_BACKENDS, default="aimnet2")
    parser.add_argument("--platform", default="cpu")
    parser.add_argument("--ml-device", default="cpu")
    parser.add_argument("--ml-mult", type=int, default=1)
    parser.add_argument("--molecule", choices=("peptide", "dna", "rna", "popc"), default="peptide")
    parser.add_argument("--compare-cpu", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    import torch

    if args.ml_device == "cuda":
        assert torch.cuda.is_available(), "The selected ML device must be available"
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = write_inputs(root)
        charge = -3 if args.molecule in {"dna", "rna"} else 0
        selection = "resi 1-2" if args.molecule == "peptide" else "all"
        bundle = MLMMBackend().build_system(
            paths[args.molecule],
            ml_backend=args.backend,
            ml_device=args.ml_device,
            ml_selection=selection,
            ml_charge=charge,
            ml_mult=args.ml_mult,
            work_dir=root / "prepared",
        )
        system = IMDSystem(
            bundle,
            IMDConfig(
                platform=args.platform,
                integrator=constants.MLMM_INTEGRATOR,
                timestep_ps=constants.MLMM_TIMESTEP_PS,
                nsteps_per_frame=1,
                mouse_force=False,
            ),
        )
        expected_platform = constants.OPENMM_PLATFORM_NAMES.get(args.platform)
        actual_platform = system.simulation.context.getPlatform().getName()
        if expected_platform is not None:
            assert actual_platform == expected_platform
        assert system._ml_subset_context.getPlatform().getName() == actual_platform
        system.minimize_step(max_iterations=30)
        delta = _compute_ml_delta(system)
        assert np.isfinite(delta).all() and np.max(np.abs(delta)) > 0
        assert system._mlip_atoms.info["spin"] == args.ml_mult
        log.info("Initial ONIOM correction: max |F| = %.4f kJ/mol/nm", np.abs(delta).max())
        comparison = None
        if args.compare_cpu:
            atoms = system._mlip_atoms.copy()
            atoms.calc = _build_mlip_calculator(
                args.backend, device_str="cpu", ml_charge=charge, ml_mult=args.ml_mult
            )
            reference = atoms.get_forces()
            actual = system._mlip_atoms.get_forces()
            np.testing.assert_allclose(
                actual, reference, atol=CROSS_DEVICE_ATOL_EV_A, rtol=CROSS_DEVICE_RTOL
            )
            comparison = float(np.max(np.abs(actual - reference)))
            del atoms
        force_csv, state_csv = root / "forces.csv", root / "state.csv"
        system.set_recording(
            traj=str(root / "trajectory.xtc"),
            force_csv=str(force_csv),
            state_csv=str(state_csv),
        )
        system.running = True
        system.start_recording()
        forces = np.zeros((system.n_atoms, 3))
        forces[0, 0] = 1.0
        system.set_pull_forces(np.array([0]), forces)
        for _ in range(10):
            system.apply_mlmm_if_present()
            alive, state = system.advance_openmm(get_forces=True)
            assert alive
            assert np.isfinite(state.getForces(asNumpy=True)._value).all()
        system.clear_pull_forces()
        system.stop_recording()
        with force_csv.open() as stream:
            force_steps = [int(row["step"]) for row in csv.DictReader(stream)]
        with state_csv.open() as stream:
            state_steps = [int(row["step"]) for row in csv.DictReader(stream)]
        assert force_steps == state_steps == list(range(1, 11))
        assert not system._active_pull_idx_prev
        result = {
            "backend": args.backend,
            "molecule": args.molecule,
            "platform": actual_platform,
            "ml_device": args.ml_device,
            "atoms": system.n_atoms,
            "ml_atoms": len(bundle.extra["ml_indices"]),
            "link_atoms": len(bundle.extra["link_atoms"]),
            "steps": system.simulation.currentStep,
            "cpu_max_force_difference_ev_a": comparison,
            "elapsed_s": time.perf_counter() - started,
            "cuda_peak_bytes": torch.cuda.max_memory_allocated()
            if args.ml_device == "cuda"
            else None,
        }
        print("BACKEND_SMOKE_OK " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
