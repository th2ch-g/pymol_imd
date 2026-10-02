"""Exercise actual OpenMM rollback, bounded recovery and recording atomicity."""

from __future__ import annotations

import csv
import logging
import tempfile
from pathlib import Path

import mdtraj
import numpy as np
import openmm
from openmm import unit
from openmm.app import Element, Topology

from pymol_imd import constants
from pymol_imd.backends.base import SystemBundle
from pymol_imd.imd import IMDConfig, IMDSystem


class ConstantML:
    """Known ML gradient for checking that rollback preserves ONIOM forces."""

    def get_forces(self, atoms):
        return np.tile([0.01, 0.0, 0.0], (len(atoms), 1))


def make_system(folder: Path, extension: str = "xtc", *, mlmm=False) -> IMDSystem:
    topology = Topology()
    residue = topology.addResidue("MOL", topology.addChain())
    topology.addAtom("C", Element.getBySymbol("C"), residue)
    system = openmm.System()
    system.addParticle(12)
    potential = openmm.CustomExternalForce("50*(x*x+y*y+z*z)")
    potential.addParticle(0)
    system.addForce(potential)
    bundle = SystemBundle(
        system=system,
        topology=topology,
        positions=np.array([[0.1, 0, 0]]) * unit.nanometer,
        backend_name="mlmm" if mlmm else "aa",
        prepared_pdb_path=folder / "unused.pdb",
        extra={
            "ml_indices": [0],
            "ml_symbols": ["C"],
            "ml_calculator": ConstantML(),
            "ml_subset_system": openmm.XmlSerializer.deserialize(
                openmm.XmlSerializer.serialize(system)
            ),
        }
        if mlmm
        else None,
    )
    imd = IMDSystem(bundle, IMDConfig(platform="reference", temperature=1, energy_cutoff_kj=100))
    imd.set_recording(
        traj=str(folder / f"trajectory.{extension}"),
        force_csv=str(folder / "forces.csv"),
        state_csv=str(folder / "state.csv"),
    )
    imd.start_recording()
    imd.running = True
    return imd


def check_recording(imd, folder, extension, expected_steps):
    imd.stop_recording()
    with (folder / "state.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["step"]) for row in rows] == expected_steps
    assert all(abs(float(row["potential_energy_kj"])) < 100 for row in rows)
    with (folder / "forces.csv").open() as stream:
        assert not list(csv.DictReader(stream)), "Rejected forces must not be recorded"
    topology = mdtraj.Topology.from_openmm(imd.bundle.topology)
    trajectory = mdtraj.load(str(folder / f"trajectory.{extension}"), top=topology)
    assert trajectory.n_frames == len(expected_steps)
    assert np.isfinite(trajectory.xyz).all()
    assert np.abs(trajectory.xyz).max() < 0.2
    if extension == "xtc":
        np.testing.assert_allclose(trajectory.time, np.asarray(expected_steps) * 0.002)


def check_pull_recovery(root):
    for extension in ("xtc", "dcd", "pdb"):
        folder = root / extension
        imd = make_system(folder, extension)
        imd.config.nsteps_per_frame = 4
        imd.set_pull_forces(np.array([0]), np.array([[1e8, 0, 0]]))
        alive, state = imd.advance_openmm(get_positions=True, get_velocities=True, get_forces=True)
        assert alive and imd._recovery_count == 1
        assert imd.simulation.currentStep == 4
        assert not imd._active_pull_idx_prev
        assert not imd._invalid_state(state)
        assert abs(state.getTime().value_in_unit(unit.picosecond) - 0.008) < 1e-12
        alive, _ = imd.advance_openmm()
        assert alive and imd._recovery_count == 1
        check_recording(imd, folder, extension, list(range(1, 9)))


def check_velocity_recovery(root):
    imd = make_system(root / "velocity")
    imd.simulation.context.setVelocities([[1e5, 0, 0]] * unit.nanometer / unit.picosecond)
    alive, state = imd.advance_openmm(get_positions=True)
    assert alive and imd._recovery_count == 1
    assert abs(state.getPositions(asNumpy=True)._value[0, 0]) < 0.2
    check_recording(imd, root / "velocity", "xtc", [1])


def check_integration_exception(root):
    folder = root / "exception"
    imd = make_system(folder)
    imd.config.nsteps_per_frame = 4
    original = imd.simulation.step
    calls = []

    def fail_once(nsteps):
        calls.append(nsteps)
        if len(calls) == 1:
            original(2)
            raise openmm.OpenMMException("Particle coordinate is NaN")
        original(nsteps)

    imd.simulation.step = fail_once
    alive, _ = imd.advance_openmm()
    assert alive and calls == [4, 4]
    check_recording(imd, folder, "xtc", [1, 2, 3, 4])


def check_unrecoverable(root):
    folder = root / "unrecoverable"
    imd = make_system(folder)
    assert imd.advance_openmm()[0]
    checkpoint_step = imd.simulation.currentStep
    imd.config.energy_cutoff_kj = 1e-12
    alive, _ = imd.advance_openmm(get_positions=True)
    assert not alive and not imd.running
    assert imd.simulation.currentStep == checkpoint_step
    check_recording(imd, folder, "xtc", [1])


def check_local_mlmm_recovery(root):
    folder = root / "mlmm"
    imd = make_system(folder, mlmm=True)
    imd.config.energy_cutoff_kj = 10
    imd._read_pymol_positions_nm = lambda: np.array([[0.2, 0, 0]])
    synced = []
    imd._sync_positions_to_pymol = synced.append
    imd.step()
    assert imd.running and imd._recovery_count == 1
    assert len(synced) == 1 and not imd._invalid_state(synced[0])
    assert not imd._active_pull_idx_prev
    delta = imd._mlmm_force.getParticleParameters(0)[1]
    expected = 0.01 * constants.EV_PER_A_TO_KJ_PER_MOL_NM + 10
    np.testing.assert_allclose(delta, [expected, 0, 0], atol=1e-10)
    check_recording(imd, folder, "xtc", [1])


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    with tempfile.TemporaryDirectory(prefix="imd-recovery-") as directory:
        root = Path(directory)
        check_pull_recovery(root)
        check_velocity_recovery(root)
        check_integration_exception(root)
        check_unrecoverable(root)
        check_local_mlmm_recovery(root)
    print("smoke-recovery: passed")


if __name__ == "__main__":
    main()
