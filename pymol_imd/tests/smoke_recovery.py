"""Verify single-pass OpenMM stepping, optional limits, and failure cleanup."""

from __future__ import annotations

import contextlib
import csv
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

import mdtraj
import numpy as np
import openmm
from openmm import unit
from openmm.app import Element, Topology

from pymol_imd import constants
from pymol_imd.backends.base import SystemBundle
from pymol_imd.imd import IMDConfig, IMDSystem


class ConstantML:
    """Known ML gradient for checking ONIOM force updates."""

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
    imd = IMDSystem(bundle, IMDConfig(platform="reference", temperature=1))
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
    assert all(np.isfinite(float(value)) for row in rows for value in row.values())
    topology = mdtraj.Topology.from_openmm(imd.bundle.topology)
    trajectory = mdtraj.load(str(folder / f"trajectory.{extension}"), top=topology)
    assert trajectory.n_frames == len(expected_steps)
    assert np.isfinite(trajectory.xyz).all()
    if extension == "xtc":
        np.testing.assert_allclose(trajectory.time, np.asarray(expected_steps) * 0.002)


@contextlib.contextmanager
def no_automatic_recovery(imd):
    with contextlib.ExitStack() as stack:
        for owner, name in (
            (imd.simulation.context, "createCheckpoint"),
            (imd.simulation.context, "loadCheckpoint"),
            (imd.simulation.context, "setVelocities"),
            (imd.simulation, "minimizeEnergy"),
        ):
            stack.enter_context(patch.object(owner, name, side_effect=AssertionError(name)))
        yield


def expect_failure(imd, operation, error, message):
    with no_automatic_recovery(imd):
        try:
            operation()
        except error as exc:
            assert message in str(exc), str(exc)
        else:
            raise AssertionError(f"Expected {error.__name__}")
    assert not imd.running and not imd._recording and not imd._active_pull_idx_prev


def check_single_pass(root):
    for extension in ("xtc", "dcd", "pdb"):
        folder = root / extension
        imd = make_system(folder, extension)
        imd.config.nsteps_per_frame = 4
        imd.set_pull_forces(np.array([0]), np.array([[20000, 0, 0]]))
        context = imd.simulation.context
        checkpoint = context.createCheckpoint()
        reporters = imd.simulation.reporters
        imd.simulation.reporters = []
        imd.simulation.step(4)
        expected = context.getState(getPositions=True, getVelocities=True)
        context.loadCheckpoint(checkpoint)
        imd.simulation.reporters = reporters
        with no_automatic_recovery(imd):
            alive, state = imd.advance_openmm(get_positions=True, get_velocities=True)
        assert alive and imd.simulation.currentStep == 4
        np.testing.assert_array_equal(
            state.getPositions(asNumpy=True), expected.getPositions(asNumpy=True)
        )
        np.testing.assert_array_equal(
            state.getVelocities(asNumpy=True), expected.getVelocities(asNumpy=True)
        )
        with (folder / "forces.csv").open() as stream:
            imd._force_csv_file.flush()
            rows = list(csv.DictReader(stream))
        assert [int(row["step"]) for row in rows] == [1, 2, 3, 4]
        assert all(float(row["fmag"]) == 20000 for row in rows)
        imd.clear_pull_forces()
        with no_automatic_recovery(imd):
            assert imd.advance_openmm()[0]
        check_recording(imd, folder, extension, list(range(1, 9)))


def check_unlimited_energy(root):
    imd = make_system(root / "unlimited_energy")
    imd.simulation.context.setVelocities([[1e5, 0, 0]] * unit.nanometer / unit.picosecond)
    with no_automatic_recovery(imd):
        alive, state = imd.advance_openmm(get_positions=True, get_velocities=True)
    assert alive and state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole) > 1e9
    assert imd._state_error(state) is None
    check_recording(imd, root / "unlimited_energy", "xtc", [1])


def check_force_boundary(root):
    imd = make_system(root / "force_boundary")
    requested = np.array([[600000, -800000, 0]])
    imd.set_pull_forces(np.array([0]), requested)
    np.testing.assert_array_equal(imd._pull_force.getParticleParameters(0)[1], requested[0])
    for force in (float("nan"), float("inf")):
        imd.set_pull_forces(np.array([0]), np.array([[100, 0, 0]]))
        imd.set_pull_forces(np.array([0]), np.array([[force, 0, 0]]))
        assert not imd._active_pull_idx_prev
        assert not np.asarray(imd._pull_force.getParticleParameters(0)[1]).any()
    for limit in (5000, 1000):
        imd.config.max_force_kj = limit
        imd.set_pull_forces(np.array([0]), requested)
        np.testing.assert_allclose(
            imd._pull_force.getParticleParameters(0)[1], [0.6 * limit, -0.8 * limit, 0]
        )
        np.testing.assert_array_equal(requested, [[600000, -800000, 0]])
    imd.stop_recording()


def check_integration_exception(root):
    folder = root / "exception"
    imd = make_system(folder)
    assert imd.advance_openmm()[0]
    imd.config.nsteps_per_frame = 4
    imd.set_pull_forces(np.array([0]), np.array([[10, 0, 0]]))
    original = imd.simulation.step
    calls = []
    failed = []

    def fail_once(nsteps):
        calls.append(nsteps)
        original(2)
        failed.append(imd.simulation.context.getState(getPositions=True, getVelocities=True))
        raise openmm.OpenMMException("Particle coordinate is NaN")

    imd.simulation.step = fail_once
    expect_failure(imd, imd.advance_openmm, openmm.OpenMMException, "Particle coordinate is NaN")
    assert calls == [4] and imd.simulation.currentStep == 3
    actual = imd.simulation.context.getState(getPositions=True, getVelocities=True)
    np.testing.assert_array_equal(
        actual.getPositions(asNumpy=True), failed[0].getPositions(asNumpy=True)
    )
    np.testing.assert_array_equal(
        actual.getVelocities(asNumpy=True), failed[0].getVelocities(asNumpy=True)
    )
    check_recording(imd, folder, "xtc", [1])


def check_invalid_state(root, *, nonfinite):
    folder = root / ("nonfinite" if nonfinite else "explicit_cutoff")
    imd = make_system(folder)
    assert imd.advance_openmm()[0]
    if nonfinite:

        def invalid(pending, state_kwargs):
            imd.simulation.context.setPositions([[float("nan"), 0, 0]])
            return imd.simulation.context.getState(**state_kwargs)

        imd._final_frame_state = invalid
        message = "non-finite"
    else:
        imd.config.energy_cutoff_kj = 1e-12
        message = "energy_cutoff_kj"
    expect_failure(imd, imd.advance_openmm, FloatingPointError, message)
    assert imd.simulation.currentStep == 2
    check_recording(imd, folder, "xtc", [1])


def check_local_mlmm(root):
    folder = root / "mlmm"
    imd = make_system(folder, mlmm=True)
    imd._read_pymol_positions_nm = lambda: np.array([[0.101, 0, 0]])
    synced = []
    imd._sync_positions_to_pymol = synced.append
    with no_automatic_recovery(imd):
        imd.step()
    assert imd.running and len(synced) == 1 and not imd._invalid_state(synced[0])
    delta = imd._mlmm_force.getParticleParameters(0)[1]
    expected = 100 * imd._read_openmm_positions_nm()[0]
    expected[0] += 0.01 * constants.EV_PER_A_TO_KJ_PER_MOL_NM
    np.testing.assert_allclose(delta, expected, atol=1e-10)
    check_recording(imd, folder, "xtc", [1])


def check_mlmm_proposed_state(root, *, raises=False):
    class LimitedML(ConstantML):
        def get_forces(self, atoms):
            if atoms.positions[0, 0] > 1.2:
                if raises:
                    raise RuntimeError("MLIP geometry outside supported range")
                return np.full((len(atoms), 3), np.nan)
            return super().get_forces(atoms)

    folder = root / ("mlmm_exception" if raises else "mlmm_proposed")
    imd = make_system(folder, mlmm=True)
    imd._mlip_atoms.calc = LimitedML()
    imd.apply_mlmm_if_present()
    assert imd.advance_openmm()[0]
    imd.set_pull_forces(np.array([0]), np.array([[1e5, 0, 0]]))
    expect_failure(
        imd,
        imd.advance_openmm,
        RuntimeError if raises else FloatingPointError,
        "outside supported range" if raises else "MLIP returned invalid forces",
    )
    assert imd.simulation.currentStep == 2
    assert imd._read_openmm_positions_nm()[0, 0] > 0.12
    check_recording(imd, folder, "xtc", [1])


def check_constraint_failure(root):
    topology = Topology()
    residue = topology.addResidue("MOL", topology.addChain())
    carbon = topology.addAtom("C", Element.getBySymbol("C"), residue)
    hydrogen = topology.addAtom("H", Element.getBySymbol("H"), residue)
    topology.addBond(carbon, hydrogen)
    system = openmm.System()
    system.addParticle(12)
    system.addParticle(1)
    system.addConstraint(0, 1, 0.109)
    folder = root / "constraint"
    bundle = SystemBundle(
        system=system,
        topology=topology,
        positions=np.array([[0, 0, 0], [0.109, 0, 0]]) * unit.nanometer,
        backend_name="aa",
        prepared_pdb_path=folder / "unused.pdb",
    )
    imd = IMDSystem(bundle, IMDConfig(platform="reference"))
    imd.running = True
    imd.set_recording(traj=str(folder / "trajectory.xtc"), state_csv=str(folder / "state.csv"))
    imd.start_recording()
    assert imd.advance_openmm()[0]

    def unconverged(pending, state_kwargs):
        imd.simulation.context.setPositions(np.array([[0, 0, 0], [0.2, 0, 0]]))
        return imd.simulation.context.getState(**state_kwargs)

    imd._final_frame_state = unconverged
    expect_failure(
        imd, imd.advance_openmm, FloatingPointError, "distance constraint did not converge"
    )
    assert imd.simulation.currentStep == 2
    assert imd._read_openmm_positions_nm()[1, 0] == 0.2
    check_recording(imd, folder, "xtc", [1])


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    with tempfile.TemporaryDirectory(prefix="imd-stepping-") as directory:
        root = Path(directory)
        check_single_pass(root)
        check_unlimited_energy(root)
        check_force_boundary(root)
        check_integration_exception(root)
        check_invalid_state(root, nonfinite=True)
        check_invalid_state(root, nonfinite=False)
        check_local_mlmm(root)
        check_mlmm_proposed_state(root)
        check_mlmm_proposed_state(root, raises=True)
        check_constraint_failure(root)
    print("smoke-recovery: passed (single-pass MD, no automatic recovery, clean failure)")


if __name__ == "__main__":
    main()
